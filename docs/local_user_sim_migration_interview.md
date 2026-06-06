# 把 user-simulator / judge 从 gpt-5 迁到本地 MoE —— 工程复盘(面试向)

> 背景:τ²-bench Airline × verl 的多轮 RL 训练里,**user simulator** 和 **NL-assertion judge**
> 原本跑在外部 API(gpt-5)。为省钱,把它们换成**本地自部署的 Qwen3.6-35B-A3B-FP8**(MoE),
> 用 A800(实际 A100 80GB)单机服务。下面按"问题 → 排查 → 解法 → 结果/可迁移点"组织,
> 每一节都是一个能独立讲 3–5 分钟的故事。

---

## 0. 一句话电梯版

> "我把一个多轮 agent RL 训练里两个外部 API 依赖(用户模拟器 + 评分裁判)迁到了本地自部署模型。
> 过程里解决了三类硬问题:**镜像限速下的可靠大文件下载**、**CUDA driver/torch 版本兼容性的根因定位**、
> 以及**服务化后的性能与正确性验证**——全程以'量化测量'代替'拍脑袋',最后用一个 35B-MoE(3B 激活)
> 在 2 张卡上稳定扛住了 rollout 峰值并发,把每次调用延迟从 21s 压到 ~5s。"

---

## 1. 选型:从"成本结构"而不是"模型榜单"出发

**问题**:要选一个能在单卡 80GB 上跑、又适合做"客服对话用户模拟"的本地模型。

**关键洞察**:先分析**调用的成本结构**,而不是直接挑"最强模型"。
- **user-sim 是吞吐瓶颈**:每个 assistant 文本轮都要调一次 → 一个 rollout 几十次 × `rollout_n` × 任务数,调用量巨大。
- **judge 是 reward 关键**:每条轨迹只在结束时调一次,量小,但直接决定 RL 的梯度信号质量。

→ 结论:user-sim 选**高吞吐**的;judge 要考虑**保真度**(我当时还提示了"别用同族模型给同族 policy 当裁判,有被 reward-gaming 的风险",后来用户在权衡后决定统一本地)。

**解法**:选 **MoE(3B 激活)** 而非同规模 dense——激活参数小=解码快,正好匹配"每轮都调"的高频场景;
35B 总参 FP8 ~35GB,单卡轻松放下。最终用 `Qwen3.6-35B-A3B-FP8`。

**可迁移点**:*容量规划先看调用频次分布,把钱/算力花在真正的热点上。*

---

## 2. 大文件下载:镜像限速下的"可靠"比"快"更重要

**问题**:模型 ~35GB,本机代理上不了 huggingface.co,只能走 hf-mirror;默认 `hf download` 只有 **~1.4 MB/s**,要 ~7.5 小时。

**排查与试错(真实走过的弯路)**:
1. 装 `hf_transfer`(Rust 并行分块)→ 短时冲到 ~7 MB/s,但**在抽风的镜像上反而更糟**:大量 `ReadTimeout`,40+ 个分片只下完 1 个就**整体卡死**。教训:激进并行对**不稳定**的源是负优化。
2. 换 **aria2c 多连接**(16 连接/文件 + 断点续传 + 自动重试)——hf-mirror 官方推荐方式。稳定不卡死,真实是镜像在按 IP 限总带宽(~3–10 MB/s 波动),客户端再调也突破不了,但**至少能可靠跑完**。
3. 自己加了**按 API 精确大小过滤**的续传逻辑:已完整文件跳过、半截文件删掉重下,避免 aria2 在 hf-client 留下的无控制文件的残片上报错。

**一个值得讲的"险情"**:用 `pkill -f "Qwen3.6..."` 停下载时——
- 模式会匹配到**正在执行这条命令的 shell 自己** → 自杀(exit 144);
- 更危险的是 `pkill -f hf` 会匹配到**正在跑的训练进程**(其 cmdline 含 `hf_merged`)→ 差点误杀几小时的训练。
→ 教训:**共享机器上永远不要用宽泛的 `pkill -f`**,按 PID/进程组精确操作。

**结果**:下完后 56 个文件**逐个比对 API size 校验**通过,35G。

**可迁移点**:*不稳定的数据源,优先选"可恢复"的传输工具;批量操作要有幂等校验;destructive 命令在共享环境要极度克制。*

---

## 3. 版本兼容性侦探:latest 装上了却跑不起来

这是整个过程**最有含金量**的一段。

**问题**:`pip install -U vllm` 装了最新 **0.22.1**,软件层完全支持模型
(`Qwen3_5MoeForConditionalGeneration` 已注册、能解析 config),但 **`torch.cuda.is_available()` 返回 False**——GPU 上根本跑不起来。

**根因定位**:
- 最新 vllm 绑的是 **torch 2.11.0 + cu130(CUDA 13.0)**,而 CUDA 13.0 需要 driver ≥ 580;
- 本机 driver **570.124.06,上限 CUDA 12.8** → cu130 不可用。
- driver 是全机共享、还有别人在训练,**不能升级/重启**。

**约束求解(找一个同时满足两条的 vllm 版本)**:既要 **支持 `qwen3_5_moe`**(版本够新),又要 **仍用 cu128 的 torch**(版本够老)——两个相反方向的约束求交集。
方法:
- 用 `pip install vllm==X --dry-run` **批量映射** 版本 → torch 版本(不真装):`0.12→2.9.0`、`0.15→2.9.1`、`0.18→2.10.0`、`0.20.2/0.21/0.22→2.11.0`。
- torch 的 `+cuXXX` 不在 pip 版本号里,于是**用它依赖的 `nvidia-cuda-runtime-cuXX` 包名反查** CUDA major:`2.9/2.10 → cu12`、`2.11 → cu13`。**一次 dry-run 就定死了 cu128/cu130 的分界线在 0.19↔0.20 之间。**
- 交叉官方信息:vLLM Recipes 写明 Qwen3.6 建议 `vllm>=0.19.0`,且 0.19.0 release note 专门有 **Qwen3.5 FP8 权重加载修复**(正好我是 FP8 模型)。

**结果**:锁定 **vllm 0.19.1**(torch 2.10/cu128,是仍用 CUDA 12.x 的最高 vllm 版本,又恰好 ≥0.19.0)。
装完四项实测全过:`is_available=True`、架构注册、`get_config` 解析、与 transformers 5.10.2 无 #36236 改名坑。

**可迁移点**:*"装上了"≠"能用"。依赖链(driver→CUDA→torch→框架)要当成一个版本矩阵来解;
善用 `--dry-run` + 依赖包名做**不落盘的快速二分**,比反复重装快一个数量级。*

---

## 4. 服务化 + 性能压测:用数据推翻直觉

**问题**:服务起来了(`vllm serve`,TP=2,OpenAI 兼容端点),但能不能扛住 rollout 的并发?

**做的事**:
1. **先发现"思考陷阱"**:Qwen3.6 默认开 thinking,user-sim 第一发回复 200 token **全耗在 `<think>` 里还没说正事**。
   → 传 `chat_template_kwargs={"enable_thinking": false}` 后,回复变成干净的一句 `"Hi, I'd like to cancel my flight booking, please."`(14 token)。
   **教训:把模型当组件用之前,先搞清它的默认行为。**
2. **用真实数据定负载,而不是瞎设参数**:从训练 rollout 日志里统计了 704 条真实 user-sim 回复长度——
   **mean 76 / p95 163 / max 255 token**(顺带发现当前每条轨迹只有 1 个 user 轮,这本身又是个训练诊断线索)。第一版压测我用 `max_tokens=80` 是**低估**的,据此改用真实分布重测。
3. **量化压测推翻直觉**:`enforce-eager`(我为快速启动临时关了 CUDA graph)在 MoE 上是**大坑**——
   长回复 conc=64 时单次延迟 **13–21s**;**开 CUDA graph 后降到 ~5s、吞吐 ×3.7(450 → 1618 tok/s)**。
   结论:2 张卡轻松扛住峰值并发 64,user-sim **不是** rollout 瓶颈(policy 的 24 轮生成才是)。

**可迁移点**:*性能问题先量化再下结论;压测要用**生产数据分布**;一个默认开关(eager/graph)可能就是 3 倍差距。*

---

## 5. 读框架源码来做决策,而不是猜

**两个例子**:
- **"tool 调用的 JSON 会进 user-sim 上下文吗?"** —— 直接读 tau2 的 `user_simulator.py` 和我们的 session 适配层,
  确认 user-sim 只看到 **自然语言对话**(系统指引 + 翻转角色的 user↔agent 文本),agent 的 tool_call / tool 结果走的是
  env 和 reward 路径,**不进** user-sim。设计本意:用户是个真人,看不到 agent 的内部工具。
  → 这直接验证了"上下文很小、不会被 tool JSON 撑爆",从而确定 `max-model-len` 怎么设。
- **集成时的一个隐蔽 bug**:tau2 的 `generate()` 内部 `to_litellm_messages()` 期望 tau2 `Message` 对象,
  我验证脚本图省事传了 dict → litellm 报 `list index out of range`。**真实路径传的是 Message 对象,没问题**——
  教训:复现/验证要走**真实调用路径**,否则会被"测试代码自己的 bug"带偏。

**可迁移点**:*关键决策(尤其涉及第三方框架行为)花十分钟读源码,胜过半天猜测和试错。*

---

## 6. 全面接入:把"swap 模型"做成可回退、可门控

**目标**:项目里所有 gpt-5 调用(user-sim + judge)统一换本地,且**可一键回退**。

**做的事**:
- `.env` 单点切换:`OPENAI_BASE_URL` 指本地 server、模型名加 `openai/` 前缀路由到 OpenAI 兼容端点;真实 key 保留不删。
- thinking 关闭走 **litellm `extra_body` 透传**(tau2 `generate()` 把 `**llm_args` 直接转给 litellm `completion`);
  judge 那条路径没有显式 llm_args 入口,就**monkeypatch 评估器模块的 `DEFAULT_LLM_NL_ASSERTIONS_ARGS`**(合并而非覆盖,保留原有 `temperature=0.0`)。
- 全部用 `TAU2_DISABLE_THINKING` env **门控**,默认开,回退到 gpt-5 时自动不发这个 vendor-specific 字段。
- 验证:端到端跑通两条路径(都路由本地、输出无 `<think>`)+ 14 个离线测试全过。

**一个判断**:发现 `TAU2_AGENT_MODEL=gpt-5` 其实**没被任何代码读**(eval 的 agent 走独立 api_base = 被训模型),
所以"换 agent"无意义、果断不动——**避免了一次会改坏语义的"过度执行"**。

**可迁移点**:*做依赖替换时:单点配置、可回退、按 env 门控、改动前先确认每个引用是否真的被用到。*

---

## 7. 贯穿全程的方法论(面试可以收尾点题)

1. **Measure, don't guess**:下载测速、版本映射、并发压测、回复长度分布——每个结论都有数据。
2. **依赖链当矩阵解**:driver → CUDA → torch → 框架 → 模型架构,任一环错位都会"装上却跑不起来"。
3. **可恢复 > 快**:不稳定环境下,断点续传 + 幂等校验比峰值速度更重要。
4. **读源码定行为**:第三方框架的默认行为/数据流,读比猜快。
5. **共享环境的克制**:精确到 PID 的操作,绝不宽泛 `pkill`;不抢别人的卡。
6. **可回退的改动**:env 门控 + 配置单点 + 保留旧凭证,随时能退回去。

---

## 附:关键数字速查

| 维度 | 数值 |
|---|---|
| 模型 | Qwen3.6-35B-A3B-FP8,MoE 256 专家/8 激活、~3B 激活、原生 256K ctx、hybrid attn+mamba、VL |
| 下载 | hf-mirror 默认 ~1.4 MB/s(~7.5h)→ aria2c 稳定跑完;35G/56 文件全校验 |
| 兼容 | driver 570→CUDA 12.8 上限;vllm 0.20+ = cu130(跑不了),**锁 0.19.1 = torch 2.10/cu128** |
| 服务 | 2×A100 TP=2,~150s 就绪;Ampere 无原生 FP8 → Marlin 反量化 |
| 性能 | conc=64:eager 21s/450 tok/s → **CUDA graph 5.8s/1618 tok/s(×3.7)** |
| 负载 | 真实 user-sim 回复 mean 76 / p95 163 / max 255 token |
| 接入 | litellm `extra_body` 关 thinking;judge monkeypatch;14 测试通过、端到端验证 |
