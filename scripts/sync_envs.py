"""Sync locked dependencies, then bind each environment to its vLLM source."""

import argparse
import hashlib
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_VLLM = "0.33.0"
SUPPORTED_COMMIT = "4ba787ec9959e566d3b587a796dff07673324da5"
BINARY_PATHS = (
    "csrc",
    "cmake",
    "CMakeLists.txt",
    "pyproject.toml",
    "setup.py",
    "rust",
    "vllm/_custom_ops.py",
    "requirements/common.txt",
    "requirements/cuda.txt",
)


def git(source: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(source), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"Git 检查失败：{source}\n{result.stderr.strip()}")
    return result.stdout.strip()


def command(args: list[str], *, env: dict[str, str], dry_run: bool) -> None:
    print(f"$ {shlex.join(args)}", flush=True)
    if not dry_run:
        subprocess.run(args, cwd=ROOT, env=env, check=True)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_wheel(wheel: dict, path: Path, *, dry_run: bool) -> None:
    expected = wheel["hash"].removeprefix("sha256:")
    if path.is_file() and digest(path) == expected:
        print(f"复用已校验的 vLLM wheel：{path}", flush=True)
        return
    if dry_run:
        print(f"下载并校验 SHA-256：{wheel['url']} → {path}", flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--retry",
                "3",
                "--output",
                str(temporary),
                wheel["url"],
            ],
            check=True,
        )
        if digest(temporary) != expected:
            raise RuntimeError("vLLM wheel 的 SHA-256 与 uv.lock 不一致。")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def sources(target: str, compiled_commit: str) -> list[tuple[str, Path, str]]:
    selected = []
    if target in ("project", "all"):
        relative = git(
            ROOT, "config", "-f", ".gitmodules", "--get", "submodule.vllm-v0.33.path"
        )
        selected.append(("project", ROOT / relative, "qwen-vllm-proj"))
    if target in ("baseline", "all"):
        selected.append(("baseline", ROOT / ".worktrees/vllm-baseline", "qwen-vllm-bl"))

    # Check every selected source before changing either environment.
    for name, source, _ in selected:
        if not (source / "pyproject.toml").is_file():
            raise RuntimeError(
                f"源码目录不存在或未初始化：{source}\n"
                "子模块目录必须与 .gitmodules 一致；"
                "基线工作树需先用 git worktree add 恢复。"
            )
        if (
            Path(git(source, "rev-parse", "--show-toplevel")).resolve()
            != source.resolve()
        ):
            raise RuntimeError(f"{source} 不是独立的 vLLM 工作区。")
        git(source, "cat-file", "-e", f"{compiled_commit}^{{commit}}")
        changed = git(
            source, "diff", "--name-only", compiled_commit, "--", *BINARY_PATHS
        )
        if changed:
            raise RuntimeError(
                f"{source} 的构建配置或依赖与锁定的官方 wheel 不同：\n{changed}\n"
                "请先更新依赖锁定和编译组件；此脚本用于 Python 源码优化。"
            )
        if name == "baseline":
            expected = (ROOT / "configs/baseline.commit").read_text().strip()
            if git(source, "rev-parse", "HEAD") != expected:
                raise RuntimeError("基线 HEAD 与 configs/baseline.commit 不一致。")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=("project", "baseline", "all"))
    parser.add_argument(
        "--dry-run", action="store_true", help="只检查并打印计划，不安装依赖"
    )
    args = parser.parse_args()
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise RuntimeError("当前锁文件仅用于 Linux x86_64、Python 3.12、CUDA 13.0。")
    if not shutil.which("uv") or not shutil.which("curl"):
        raise RuntimeError("请先安装 uv 和 curl。")
    python_version = (ROOT / ".python-version").read_text().strip()
    compiled_commit = (ROOT / "configs/precompiled.commit").read_text().strip()
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    reference = next(
        package for package in lock["package"] if package["name"] == "vllm"
    )
    if (reference["version"], compiled_commit) != (SUPPORTED_VLLM, SUPPORTED_COMMIT):
        raise RuntimeError(
            "升级 vLLM 时需一并更新此安装脚本和 configs/precompiled.commit。"
        )
    wheel = next(
        item
        for item in reference["wheels"]
        if item["url"].endswith("manylinux_2_28_x86_64.whl")
    )
    selected = sources(args.target, compiled_commit)
    wheel_path = ROOT / ".cache/wheels" / Path(urlparse(wheel["url"]).path).name
    environment = os.environ.copy()
    environment.pop("VIRTUAL_ENV", None)
    command(
        ["uv", "lock", "--project", str(ROOT), "--check"],
        env=environment,
        dry_run=args.dry_run,
    )
    prepare_wheel(wheel, wheel_path, dry_run=args.dry_run)

    snapshots = []
    for name, source, prompt in selected:
        venv = source / ".venv"
        python = venv / "bin/python"
        if not venv.exists():
            command(
                [
                    "uv",
                    "venv",
                    "--python",
                    python_version,
                    "--prompt",
                    prompt,
                    str(venv),
                ],
                env=environment,
                dry_run=args.dry_run,
            )
        target_environment = {**environment, "UV_PROJECT_ENVIRONMENT": str(venv)}
        print(f"\n同步 {name} 环境：{venv}", flush=True)
        # vLLM's release metadata supplies the dependency graph. Its actual
        # editable source is selected here rather than hardcoded in uv.lock.
        command(
            [
                "uv",
                "sync",
                "--project",
                str(ROOT),
                "--locked",
                "--python",
                python_version,
                "--no-install-package",
                "vllm",
            ],
            env=target_environment,
            dry_run=args.dry_run,
        )
        build_environment = {
            **target_environment,
            "VLLM_USE_PRECOMPILED": "1",
            "VLLM_PRECOMPILED_WHEEL_LOCATION": str(wheel_path),
        }
        command(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python),
                "--no-deps",
                "--no-build-isolation",
                "--reinstall-package",
                "vllm",
                "--editable",
                str(source),
            ],
            env=build_environment,
            dry_run=args.dry_run,
        )
        command(
            ["uv", "pip", "check", "--python", str(python)],
            env=target_environment,
            dry_run=args.dry_run,
        )
        command(
            [
                str(python),
                "-c",
                "import sys, pathlib, vllm; "
                "print('Python:', sys.executable); "
                "print('vLLM:', vllm.__version__, vllm.__file__); "
                "assert pathlib.Path(vllm.__file__).resolve().is_relative_to("
                "pathlib.Path(sys.argv[1]).resolve())",
                str(source),
            ],
            env=target_environment,
            dry_run=args.dry_run,
        )
        if not args.dry_run:
            snapshot = subprocess.check_output(
                ["uv", "pip", "freeze", "--python", str(python), "--exclude-editable"],
                cwd=ROOT,
                env=target_environment,
                text=True,
            )
            snapshots.append(sorted(snapshot.splitlines()))
    if len(snapshots) == 2 and snapshots[0] != snapshots[1]:
        raise RuntimeError("两套环境的第三方依赖不一致。")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from error
