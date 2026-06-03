# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 这是什么

τ²-bench **Airline** × **verl**:一个多轮工具调用的客服 agent,用 **LoRA + 标准 GRPO
(RL-zero,无 SFT)** 在 Qwen3-8B 上训练(2×A100 80GB)。用户模拟器(user simulator)
和 NL-assertion 评判(judge)跑在外部 API 模型(**gpt-5**)上;reward 用的是 tau2 官方的
`evaluate_simulation`(DB hash + assertions),不是自定义规则。完整设计依据见
`docs/tau3_airline_qwen3_verl_grpo_plan.md`。

## 环境与铁律

- **所有 Python 都跑在 conda env `tau2verl` 里**(一个全新的 py3.12,克隆自可用的
  `verl` 环境)。脚本通过 `conda run -n "$TAU2_ENV"` 调用;临时命令也照此办理。
- **`third_party/verl` 和 `third_party/tau2-bench` 是只读的 git submodule**(已 pin、
  镜像 clone、editable 安装)。绝不修改它们的源码——所有适配都在 `src/tau2_airline_verl/`
  里通过 wrapper/factory 完成。pin 的版本很关键:verl 0.9.0.dev、vllm 0.11.0、
  torch 2.8.0、transformers 4.56.1、numpy 2.2.6(`pip check` 报的 "verl requires
  numpy<2" 是预期内的、无害的警告)。
- **`.env` 是机器路径和密钥的唯一来源**(从 `.env.example` 复制)。每个脚本都会自动
  source 它;Hydra/verl 通过 `${oc.env:VAR}` 读取。关键项:`OPENAI_API_KEY`(gpt-5)、
  `QWEN3_8B_PATH`、`TAU2_ENV`、`TAU2_USER_MODEL`/`TAU2_NL_JUDGE_MODEL`。
- 本机代理无法访问 `huggingface.co` 和 `github.com`;模型下载走 `hf-mirror.com`,
  submodule 走 `gh-proxy.com` 镜像(见 install.sh)。

## 常用命令

```bash
# 一次性安装(submodule + editable 安装进 tau2verl)
bash scripts/setup/install.sh
bash scripts/setup/download_qwen3_8b.sh          # Qwen3-8B 权重(~16GB,hf-mirror)

# 离线测试 —— 无需 GPU、无需 API(对 server/tokenizer/session 打桩)
conda run -n tau2verl python -m pytest tests/ -q
conda run -n tau2verl python -m pytest tests/test_agent_loop.py -q      # 单个文件
conda run -n tau2verl python -m pytest tests/ -q -k mask                # 单个测试
conda run -n tau2verl ruff check src/ tests/                            # lint(line-length 100)

# 构建 verl GRPO 数据集(train=40 / test=10 个 airline 任务,test 按功能场景分层覆盖)
bash scripts/data/build_parquet.sh               # -> data/tau3_airline/{train,test}.parquet

# stage 1 —— base GO/NO-GO rollout(启动本地 vLLM server,跑 tau2 官方 runner)
bash scripts/eval/run_base_rollout.sh 8
bash scripts/eval/run_base_rollout.sh 2 --task-ids 0 1                  # 先 smoke 几个任务

# stage 2 —— GRPO 训练(2×A100)
bash scripts/train/run_grpo_smoke.sh             # 极小的端到端 smoke(2 任务/n=2/1 步)
bash scripts/train/run_grpo.sh                   # 完整训练;每个旋钮都是 env-var 覆盖

# stage 3 —— held-out 评估 + Base vs GRPO 对比报告
LORA_PATH=<trained_adapter_dir> bash scripts/eval/run_eval.sh 8
conda run -n tau2verl python -m tau2_airline_verl.evaluation.report --base base.jsonl --grpo grpo.jsonl
```

## 架构(跨多个文件才能看清的部分)

**数据行只携带 system prompt;轨迹是在 rollout 时构建的。** `data/build_parquet.py`
为每个任务产出一行 verl 数据,其 `prompt` 仅是 airline `policy.md` 的 system message,
任务身份放在 `extra_info.task_id` 里,`agent_name="tau2_airline"` 负责把这一行路由到
我们的自定义 agent loop。多轮对话(greeting、用户请求、工具调用、observation)**不**
存在于数据集中——它是在 rollout 期间实时生成的。

**`Tau2AirlineAgentLoop`(`rollout/agent_loop.py`)是集成的主脊。** 这个版本的 verl 没有
`Interaction` 抽象,所以没法用 verl 惯用的方式挂载 user simulator。取而代之的是,这个
单一的 `@register("tau2_airline")` AgentLoop 拥有整条轨迹,每一轮在两个世界之间搭桥:
- policy(assistant)轮 → verl 的 `server_manager.generate`(可训练的 LLM)
- 工具调用 → tau2 的 `Environment.get_response`(真实地改动 FlightDB)
- 自然语言轮(没有工具调用)→ tau2 的 gpt-5 `UserSimulator`
- 轨迹结束 → tau2 官方 `evaluate_simulation` 算 reward

它逐字镜像了 verl `ToolAgentLoop` 的 token 记账;唯一新增的是 **user-simulator 分支**
(当 policy 输出文本而非工具调用时,由 user 回应而非终止)。

**`Tau2Session`(`rollout/tau2_session.py`)是 tau2 的同步半边。** 它为每条轨迹构建全新的
`Environment` + `UserSimulator`(因此 DB 写入绝不会跨 rollout 泄漏),驱动 tau2 的消息
交换,并组装出供评估器使用的 `SimulationRun`。它的方法全是阻塞式的;AgentLoop 用
`run_in_executor` 把它们卸载出去,这样并发的 rollout 不会卡住 event loop。
`conversion.py` 在 tau2 `Message` 对象与 OpenAI chat dict 之间双向翻译。

**`response_mask` 约定 —— 最容易写错的地方。** assistant(policy)token 的 mask 为
**1**(计入 loss);tool 和 user 的 token 的 mask 为 **0**。token 的累积逐字照搬
`ToolAgentLoop` 的做法:初始 prompt `[system, greeting, first-user]` 只渲染一次,之后
每个后续轮次只通过 `apply_chat_template(add_messages, remove_system_prompt=True)` 追加
*增量* token。`tests/test_agent_loop.py` 在离线状态下钉死了这一点,如果你改动这个 loop,
它就是权威参考。

**reward 路径 —— 没有自定义 reward manager。** AgentLoop 把 tau2 的标量放进
`AgentLoopOutput.reward_score`,verl 会把它落到 `rm_scores[-1]`,因此默认的 `naive`
reward manager 直接透传。`env/reward.py` 包装了
`evaluate_simulation(..., EvaluationType.ALL)`:最终 reward 是该任务自己的
`evaluation_criteria.reward_basis` 上各分量的乘积;各分量子分数(DB、COMMUNICATE、
ACTION、NL_ASSERTION、ENV_ASSERTION)会被暴露出来用于训练曲线。NL judge 模型通过
`set_nl_judge_model()` 从 tau2 默认的 gpt-4.1 monkeypatch 成 gpt-5。KL 的参考策略是
**关掉 adapter 的 base 模型**(LoRA 的特性)——没有单独的 reference model。

**两条评估路径,reward 口径一致。** 训练用上面那个 verl AgentLoop。GO/NO-GO gate 和
held-out 评估(`evaluation/run_tau2_eval.py`)则改用 **tau2 自己的 `run_domain` runner**,
对接一个独立的 vLLM OpenAI server——更简单、不依赖训练栈——但用的是同一个
`evaluate_simulation` reward,所以 base 与 trained 的数字可直接比较。"informative" 任务
(多次 trial 中 reward 既非全 0 也非全 1)正是 GRPO 产生梯度所需的;markdown 的结论判定
就基于这个。

**`env/`、`agents/`、`data/` 是薄薄的 tau2 适配层。** `airline_tool.py` 构建 Environment +
OpenAI 工具 schema + `policy.md`;`airline_interaction.py` 接好 gpt-5 `UserSimulator`;
`agents/qwen3_prompt.py` 把 system prompt 构建为 `policy.md`,而工具是*单独*传给 chat
template 的(Qwen3 原生 / hermes 格式,绝不拼进 prompt 文本)——这种格式保真度正是跳过
SFT(RL-zero)的前提。`data/splits.py` 在本地维护一份**分层 40/10 split**(`airline_split.json`,
不碰只读 submodule):用 gpt-5 把 50 个任务按功能场景(取消/预订/改签/行李·乘客/补偿/保险/其他)打标,
held-out `test`(10)保证每个场景至少 1 条、剩余名额按场景规模分配,`train`(40)与之互不相交。加载时
绕过 tau2 的 `get_tasks_split()`(它读 submodule 旧 split),改为载入全部 50 任务再用本地 id 列表过滤。

## 配置约定

- `scripts/train/run_grpo.sh` 是**权威的、可运行的**训练入口;它在 verl 默认的
  `ppo_trainer.yaml` 之上传入 CLI 覆盖。每个旋钮都是一个 env var(如 `LORA_RANK=`、
  `ROLLOUT_N=`、`MAX_RESPONSE_LENGTH=`),`run_grpo_smoke.sh` 正是借此复用它。改一次
  运行就改脚本或设 env var。
- `configs/grpo_qwen3_airline.yaml` **仅为文档**——它可读地镜像了脚本里真实的 verl
  key/value;它不被任何东西加载。
- `configs/tau2_agent_loop.yaml` 是实际生效的注册表,把 `tau2_airline` 这个名字映射到
  `Tau2AirlineAgentLoop`;`rollout.agent.agent_loop_config_path` 指向它。
