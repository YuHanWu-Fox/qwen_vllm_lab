# vLLM Hybrid MoE DBO

这个项目在 vLLM Model Runner V2 中给 GDN 和 Mamba2 混合 MoE 接入 token 粒度的双批重叠（DBO）。在 CUDA DP+EP 下，一次调度可以切成两个微批，切点可以落在一个 prefill 请求内部。两个微批沿用 vLLM 的 MoE 交接顺序，让一批的计算与另一批的专家通信重叠。 [上游 PR #69502](https://github.com/vllm-project/vllm/pull/69502)

## 双 H200 实测

双 H200 NVL、DP=2、EP=2、TP=1，BF16，缓存模式为 `align`。下表使用 DeepEP HT，每 rank 一步处理 32,768 个 prefill token。开关 DBO 的两组使用相同权重和输入。 [实验设置与完整结果](https://github.com/vllm-project/vllm/pull/69502#performance)

| 场景 | 指标 | DBO 关 | DBO 开 |
| --- | --- | ---: | ---: |
| Granite 4.0 Hybrid Tiny 在线服务 | 完成请求吞吐 | 448.1 req/s | **545.5 req/s（+21.8%）** |
| Granite 在线服务 | 一波请求全部收到首 token 的耗时中位数 | 444.6 ms | **342.2 ms（−23.0%）** |
| Qwen3.5-35B-A3B 文本模型，离线 `LLM.generate` | 批次耗时中位数 | 604.8 ms | 520.4 ms |

Granite 在线测试每波 256 条请求，每条输入 256 个 token，输出 4 个 token。prefill 和 decode 阈值分别设为 16,384 和 129。日志确认大 prefill 被切分，随后的 decode 走单批。五个计时波次的 1,280 条输出序列逐项一致。

另用 DeepEP LL 对比 Granite 的单批和双微批 FULL decode。每 rank 为 32、64、128 个请求时，拆批后的吞吐分别下降 43.9%、43.3%、38.2%。在这组负载上，DBO 更适合用在大 prefill，纯 decode 保持单批回放。 [FULL decode 对照](https://github.com/vllm-project/vllm/pull/69502#steady-state-full-decode)

## 切开同一请求后，状态怎么接上

请求被切成 A 和 B 时，两段需要读写同一个卷积与递归状态工作块。`none` 模式下，两份切片表中的请求行本来就指向同一块。`align` 模式则先按整步的 `seq_lens` 选工作块，再按请求行把索引交给各微批。A 写回该层的状态，B 在共享计算流上接着读写，query 长度各自按切片更新。

全 padding 的 eager 微批用一条指向 null block 的 dummy 元数据，保留 DP 对齐后的 token 范围参与专家通信。FULL 单 token decode 按捕获时的请求行布局切分，回放前更新固定地址的元数据。如果任一 rank 预留了内部 prefill checkpoint，整步通过现有 DP 协议回退，沿原路径完成导出。 [工作块定位与切分细节](https://github.com/YuHanWu-Fox/qwen_vllm_lab/blob/243ad8cd80165fba25ea65c8902bc5ffbe9fee64/README.md)

## 验证和复现

实权重对照使用 Granite 4.0 Hybrid Tiny 和补齐官方输出头的 Qwen3.5-35B-A3B 文本权重，覆盖请求内切分、前缀复用、checkpoint 回退、不对称 DP padding 及 FULL 回放。固定 DBO 开启，分别对两种模型的 `none` 和 `align` 切换同步与异步调度，**112 条生成序列和 384 个位置的 top-5 logprob 全部一致**。双卡分布式回归也检查了同一请求在上一步输出尚未被消费时再次进入调度。

| 内容 | 版本与材料 |
| --- | --- |
| 本分支的代码 | `vllm-v0.33` 固定在 Base 开发版本 `214be07d6c`，使用同步调度。[双 H100 的 DeepEP 与 NIXL 验证](experiments/h100_hybrid_dbo/README.md) |
| 上面的双 H200 性能数据 | 同步调度。[性能测试提交 `0e08db76bf`](https://github.com/YuHanWu-Fox/vllm/commit/0e08db76bf35bd88805420153035993b2a99a4c8) |
| 上游投稿实现与异步回归 | [异步支持提交 `5a3ee0916d`](https://github.com/YuHanWu-Fox/vllm/commit/5a3ee0916d6d31ef4334462d8f8888c035d623a1) |

在本仓库用 `git submodule update --init` 检出 Base 子模块。复现双 H200 数据时，按表选择对应提交，从验证包的 `REPRODUCE.md` 开始。异步调度的独立对照见包内的 `async-regression/README.md`。包里也有固定的模型版本、DeepEP 构建环境、benchmark 脚本和逐 rank 日志。 [下载验证包](https://github.com/user-attachments/files/35303311/hybrid-dbo-pr-evidence.zip)
