import os
import subprocess
import sys
from pathlib import Path

# 这些环境变量必须在 import vllm 之前设置。

# WSL2 中启用 pinned memory，使 MRV2 所需的 UVA 可用。
os.environ["VLLM_WSL2_ENABLE_PIN_MEMORY"] = "1"

# 明确使用 Model Runner V2。
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"

# Engine Core 留在当前进程，方便断点调试。
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import vllm
from vllm import LLM, SamplingParams


def print_environment() -> None:
    vllm_source = Path(vllm.__file__).resolve()
    try:
        revision = subprocess.run(
            ["git", "-C", str(vllm_source.parent), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "无法读取 Git 提交"

    print("========== 当前运行环境 ==========")
    print("Python 解释器：", sys.executable)
    print("Python 版本：", sys.version.split()[0])
    print("实际环境目录：", sys.prefix)
    print("vLLM 版本：", vllm.__version__)
    print("vLLM 源码：", vllm_source)
    print("vLLM 提交：", revision)
    print("==================================\n", flush=True)


def main() -> None:
    print_environment()

    model_path = Path(
        "~/vllm-projects/models/Qwen3-0.6B"
    ).expanduser()

    print("模型路径：", model_path, flush=True)
    if not model_path.exists():
        raise FileNotFoundError(f"模型目录不存在：{model_path}")

    llm = LLM(
        model=str(model_path),

        # RTX 3060 支持 FP16。
        dtype="float16",

        # 初期缩短上下文，降低 KV Cache 显存占用。
        max_model_len=1024,

        # 初期只处理一个请求，便于跟踪源码。
        max_num_seqs=1,

        # 给模型、KV Cache 和运行时使用约 70% 显存。
        gpu_memory_utilization=0.70,

        # 初期关闭 CUDA Graph 和 torch.compile，
        # 让普通 Python eager 调用链更容易调试。
        enforce_eager=True,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=32,
    )

    prompts = [
        "请用三句话解释什么是 GPU。",
    ]

    outputs = llm.generate(
        prompts,
        sampling_params,
    )

    for output in outputs:
        print("\n========== 推理结果 ==========")
        print("输入：", output.prompt)
        print("输出：", output.outputs[0].text)


if __name__ == "__main__":
    main()
