# 基于 τ³-bench Airline 的 Qwen3-8B LoRA verl GRPO Agentic RL 项目计划书

> 版本说明（修订要点）：
> - 训练框架由 **slime 改为 verl**：slime 训练后端是 Megatron-LM，**不支持 LoRA**；verl 的 LoRA 是一等公民（HF peft + FSDP/FSDP2 + vLLM/SGLang rollout），且原生支持多轮工具调用（AgentLoop）。
> - 训练范式由 "SFT warmup + GRPO" 改为 **RL-zero（从 Qwen3-8B-Instruct 直接 GRPO，无 SFT warmup）**，由"阶段1 base rollout"做 GO/NO-GO 闸门把关。SFT 退为可选回退方案。
> - "异步"重新定义为 **(a) 并发 rollout 隐藏 tool/API 延迟（完全 on-policy，无 staleness）**，而不是 off-policy 的训练/生成解耦。后者（one_step_off / fully_async）列入后续深化方向。
> - **第一版只实现最标准的 GRPO**。DAPO/GSPO、turn-level credit、shaped reward、跨域迁移、user-sim 鲁棒性、off-policy 异步等写入"后续深化方向"，后续再实现。

---

## 0. 项目一句话概述

本项目基于 **τ³-bench Airline** 构建一个真实多轮工具调用客服 Agent 的 RL 训练系统：使用 **Qwen3-8B-Instruct** 作为 policy model，在 **2×A100 80GB** 上通过 **LoRA + verl GRPO（RL-zero，无 SFT）** 进行训练；使用外部 API 大模型模拟真实用户反馈（接入 verl 的 Interaction 抽象），使用 τ³-bench 提供的 Airline 工具、数据库、业务规则和 verifier 计算奖励，并通过**并发 rollout 隐藏工具/API 延迟**提高采样效率，最终提升 Agent 在航空客服任务中的任务完成率、工具调用正确率和业务规则遵守能力。

---

## 1. 项目背景

传统 LLM RL 项目常见形式是：

```text
prompt -> response -> reward
```

但真实 Agent 场景更复杂，尤其是客服、订票、改签、退款等业务场景。Agent 需要和用户多轮沟通、调用工具、读取和修改数据库状态，并严格遵守业务规则。

τ³-bench Airline 正好提供了这种环境：

```text
任务目标 + 用户场景 + 业务规则 + 工具 API + 数据库状态 + verifier/reward
```

因此它非常适合做 **agentic RL**：不是训练模型单次回答问题，而是训练模型在环境中完成长程、多轮、可验证的业务任务。

---

## 2. 项目目标

### 2.1 核心目标

构建一个可复现的 Agentic RL 训练系统，使 Qwen3-8B 能够在 τ³-bench Airline 任务中通过多轮对话和工具调用完成航空客服任务。

主要目标包括：

1. 接入 τ³-bench Airline 环境（tasks、policy、tools、db 和 verifier）。
2. 使用 API 大模型作为 user simulator，模拟真实用户反馈（接入 verl Interaction）。
3. 使用 Qwen3-8B-Instruct 作为本地 policy agent，使用其**原生工具调用格式**。
4. 使用 LoRA 降低训练显存，适配 2×A100 80GB 训练环境。
5. 使用 verl 框架实现 **并发多轮 rollout（AgentLoop）+ 标准 GRPO 更新**。
6. 通过并发 rollout 隐藏 tool/API 延迟，并缓存 user-sim 回复以降低成本、提升复现性。
7. 使用 DB reward 和 communicate reward 作为主要 RL 反馈。
8. 评估 pass@1、pass^k、success rate、tool-call accuracy、policy violation rate 等指标。

### 2.2 项目最终效果

第一版目标（RL-zero）：

```text
Base Qwen3-8B  <  Base Qwen3-8B + GRPO (RL-zero)
```

即在不做 SFT 的前提下，仅靠 verifier reward 的 GRPO，让模型在 τ³-bench Airline 上取得更高的任务成功率和更低的工具调用错误率。

> 预期管理：参考已公开的 tau-bench airline 多轮 RL 结果（Qwen3-30B-A3B 58.0%→69.5%、一个 4B 模型 63.8%→66.7%），**RL 的绝对提升通常是个位数到十几个点**，且建立在 base 已有非零成功率之上。本项目以"几个点的 success rate 提升 + tool-format 错误率下降 + policy violation 不上升"为合格线，而非追求 SOTA。

---

## 3. 为什么选择 τ³-bench Airline

### 3.1 为什么不是原 tau-bench

原 tau-bench 的 airline/retail 数据已经有一些任务歧义和评测问题。τ³-bench 是 sierra-research/tau2-bench 仓库升级而来的新版（新增 banking 域与 voice 模态），并对 airline 和 retail 做了大量修复，纳入了 τ²-Bench Verified（Amazon）和 Anthropic 等外部审计的修正。

对于训练项目来说，新版更适合，因为 reward 更干净，减少了"模型做对但被判错"的情况。

### 3.2 为什么先做 Airline

相比 retail、knowledge、SWE、WebArena 等环境，Airline 更适合作为第一版 RL 项目：

| 方向 | 优点 | 缺点 | 是否适合第一版 |
|---|---|---|---|
| τ³-bench Airline | 工具清楚、状态明确、轨迹较短 | 任务数量有限 | 很适合 |
| τ³-bench Retail | 更复杂、更接近电商客服 | 状态和规则更杂，轨迹更长 | 第二阶段 |
| τ-knowledge | 有 RAG 和复杂政策推理 | 上下文更长，变量更多 | 不建议第一版 |
| SWE-bench | reward 清晰，含金量高 | 环境极重，rollout 慢 | 不适合第一版 |
| WebArena | 真实网页交互 | 环境不稳定，成本高 | 不适合第一版 |

Airline 场景包括：

```text
查 reservation / 查用户信息 / 查询航班 / 改签航班 / 取消 reservation
计算费用 / 处理支付或 refund / 查询行李额度 / 拒绝违规请求
```

这些任务既有真实业务复杂度，又不会像 SWE-bench 那样工程成本爆炸。

---

## 4. τ³-bench Airline 数据结构理解

τ³-bench Airline 不是普通的问答数据，而是一套环境定义。典型目录结构类似：

```text
data/tau2/domains/airline/
  tasks.json
  tasks_voice.json
  split_tasks.json
  policy.md
  db.json
  audio_difficulty.json
```

本项目主要使用：`tasks.json`、`policy.md`、`db.json`、`split_tasks.json`。

### 4.1 tasks.json

`tasks.json` 是任务定义文件。每条 task 通常包含：

```json
{
  "id": "8",
  "description": { "purpose": "Booking with extra passenger.", "relevant_policies": null, "notes": null },
  "user_scenario": {
    "persona": null,
    "instructions": {
      "task_instructions": "...",
      "domain": "airline",
      "reason_for_call": "...",
      "known_info": "...",
      "unknown_info": "..."
    }
  },
  "initial_state": null,
  "evaluation_criteria": {
    "actions": [],
    "communicate_info": [],
    "nl_assertions": [],
    "reward_basis": ["DB", "COMMUNICATE"]
  },
  "annotations": null
}
```

### 4.2 user_scenario

`user_scenario` 是 API user simulator 的核心输入。它定义了用户为什么打电话、知道什么、不知道什么、应该如何表现。它会被映射到 verl Interaction 的初始化参数（见 §8）。

### 4.3 policy.md

`policy.md` 是 Airline 业务规则，作为 Agent system prompt 的重要组成部分。它规定：

```text
什么票可以取消 / 什么票可以改签 / basic economy 限制 / 是否允许改 origin/destination
保险、补偿、退款规则 / 行李额度计算 / 支付方式与 certificate/gift card 使用限制
```

Agent 不能只追求完成任务，还必须遵守 policy。

### 4.4 db.json

`db.json` 是后端数据库初始状态。Agent 调用工具时，环境会真实读取或修改数据库，而不是由 LLM 幻想结果。可能包含：users / reservations / flights / payment methods / certificates / gift cards / membership status / baggage information。

### 4.5 evaluation_criteria

`evaluation_criteria` 是 RL reward 的主要依据。核心字段：`actions`、`communicate_info`、`nl_assertions`、`reward_basis`。

需要注意：`actions` 更像参考轨迹，不应简单当成唯一正确答案。Agent 不一定要调用完全相同的工具路径，只要**最终数据库状态正确**且**应沟通信息传达到位**即可。

> 关于 reward 计算：**优先直接复用 τ³-bench 自带的 evaluator**（DB state 比对 + communicate / nl_assertions 检查），而不是自己重写一套，以保证 reward 口径与官方 metric 一致、减少 bug。需要注意 τ³-bench 的 communicate/nl_assertions 检查本身可能是 **LLM 判定**的——这点见 §11.3。

---

## 5. 系统总体架构（verl + 并发 on-policy rollout）

```text
                   ┌───────────────────────┐
                   │ τ³-bench Airline Tasks │
                   └───────────┬───────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ verl HybridEngine（2×A100 80GB colocated，时分复用）                │
│                                                                │
│  ── Rollout 阶段（GPU 跑 SGLang server / AsyncLLMServer）──    │
│     AgentLoop 并发跑 N 条 trajectory：                          │
│        Qwen3-8B Policy (LoRA, 当前版本)                        │
│           ↕  tool_call                                         │
│        Tool   = τ³-bench Airline Tools + DB（async handler）   │
│           ↕  natural language                                 │
│        Interaction = API User Simulator（async, 带缓存）       │
│        某条 trajectory 等 tool/API 时，GPU 去生成其它条        │
│        ↓ 完成后由 τ³-bench verifier 计算 reward                │
│                                                                │
│  ── Training 阶段（GPU 跑 FSDP）──                             │
│     按 task_id 分组 → group-relative advantage → GRPO update   │
│     KL 参考 = 关闭 LoRA adapter 的 base 模型（不额外占显存）   │
│     仅对 assistant token 计 loss                              │
│  → 更新后的 LoRA 权重在下一轮 rollout 前同步给 SGLang          │
└──────────────────────────────────────────────────────────────┘
```

要点：rollout 与 training **时分复用同一批 GPU（HybridEngine）**，每轮训练用的样本都是**当前最新策略**生成的，因此**完全 on-policy、无 staleness**。"异步"仅指 rollout 阶段内部的并发（隐藏 tool/API 延迟），不是训练/生成跨版本重叠。

---

## 6. 模型与训练框架

### 6.1 Policy Model

使用 **Qwen3-8B-Instruct**。

不用 Qwen3-VL-8B 的原因：τ³-bench Airline 是纯文本客服任务，无图片输入，VL 会引入额外工程复杂度，文本版更适合工具调用与 verl 接入。

> thinking 模式：Qwen3 支持 thinking/non-thinking 切换。**第一版关闭 thinking**（多轮 + 工具调用下 thinking 会大幅膨胀 token、复杂化 loss mask 与解析）。是否开启 thinking 列入后续深化方向。

### 6.2 Tool-Call 格式：使用 Qwen3 原生 function calling

不自定义 JSON 协议，**直接使用 Qwen3 的原生工具调用模板（Hermes 风格）**，由 SGLang 的 tool parser 解析。理由：与模型训练时的格式一致，base 表现更好、解析更稳、invalid tool-call 率更低（这也是可以跳过 SFT 的重要前提）。

解析失败时：记录 `invalid_tool_call`，环境返回错误 observation，由 reward / diagnostic penalty 体现。

### 6.3 LoRA 配置

```yaml
lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
```

第二版可升级到 `r: 32 / alpha: 64`。

### 6.4 训练精度与显存优化

```yaml
precision: bf16
gradient_checkpointing: true
flash_attention: true
micro_batch_size: 1
gradient_accumulation_steps: 8-16
context_length: 4096 -> 8192
```

### 6.5 训练框架：verl

选择 verl 而非 slime 的原因：

1. **LoRA 一等公民**：HF peft + FSDP/FSDP2，vLLM 和 SGLang rollout 均支持，并有 multi-gpu LoRA RL 省显存。（slime 后端是 Megatron-LM，不支持 LoRA，这是换框架的根本原因。）
2. **原生多轮 agentic rollout**：AgentLoop（v0.4.2+）+ SGLang multi-turn + Interaction System，工具调用与用户模拟都有一等抽象。
3. **算法覆盖广**：PPO/GRPO/GSPO/DAPO/Dr.GRPO/RLOO 等，方便后续做算法 ablation。
4. **后端灵活**：FSDP/FSDP2/Megatron + vLLM/SGLang/HF，2 卡 colocated HybridEngine 即可跑。

本项目中，verl 负责：

```text
Qwen3-8B LoRA 训练（FSDP）
标准 GRPO update
SGLang server-mode rollout（AgentLoop）
并发多轮 trajectory 生成
KL 参考（关闭 adapter 的 base 模型）
```

自定义模块（接入 verl 抽象）负责：

```text
τ³-bench Airline Tool 适配（→ verl Tool）
API user simulator 适配（→ verl Interaction）
reward 适配（→ 复用 τ³-bench verifier）
失败原因标注 / 评测脚本
```

---

## 7. 为什么需要并发 rollout（隐藏 tool/API 延迟）

在 Airline agentic setting 中，单条 trajectory 的墙钟时间被**外部延迟**主导：

```text
每次 tool call 要执行 τ³-bench 工具
每次自然语言回复要等 API user simulator 返回（网络往返）
不同任务轮数差异大（4 轮 ~ 12 轮）
```

如果 rollout 串行处理 trajectory，GPU 会在等 tool/API 时大量空转。

**解决方案（本项目采用，记为方案 a）：rollout 阶段内部并发。** 用 verl AgentLoop 的 server-mode（AsyncLLMServer）同时跑 N 条 trajectory，把 tool/Interaction handler 写成 **async 协程**：某条 trajectory 在 `await` 外部响应时，GPU 继续为其它 trajectory 生成 token，外部延迟被并发摊掉。

关键性质：**这套是完全 on-policy 的——所有样本都用当前最新策略生成，没有 staleness。** 它只解决"不要串行干等"，不引入 off-policy。

> 与之相对的方案 b（让 optimizer 在 batch 还没生成完时就更新，即 one_step_off / fully_async）会让样本变成旧版本生成（staleness≥1），且其吞吐收益依赖 trainer/rollout **分卡**重叠，在 2×A100 80GB 上收益很小。**本项目第一版不采用方案 b**，列入后续深化方向（§16）。

GRPO 适合同一任务多条 trajectory 的比较：

```text
task_i:  traj_1 r=0  traj_2 r=1  traj_3 r=0  traj_4 r=1 ...
```

组内 advantage：

```python
advantage_i = (reward_i - mean(group_rewards)) / (std(group_rewards) + 1e-6)
```

---

## 8. API User Simulator 设计（→ verl Interaction）

### 8.1 为什么用 API 模拟用户

user simulator 不是被训练对象，只是环境的一部分。用 API 而非本地 72B 的好处：省显存、省部署、质量稳定、便于替换不同强弱用户、天然适合 async 并发。真正训练的是本地 Qwen3-8B policy。

> 接入方式：优先**直接复用 τ³-bench 自带的 user simulator 实现**，将其包成 verl 的 Interaction（多轮）。这样 reward 口径、用户行为都与官方一致。

### 8.2 User Simulator 职责

只负责模拟用户自然语言反馈，**不模拟数据库、不执行工具**。数据库和工具调用由 τ³-bench 本地环境执行。

```text
Agent: 请提供 reservation id。
User: 是 ABC123。
Agent: 这个改签需要补差价 80 美元，可以接受吗？
User: 可以，但我不想坐红眼航班。
```

### 8.3 User Simulator Prompt 模板（如自建）

```text
You are simulating a real customer calling an airline support agent.

Hidden user scenario:
- Domain: airline
- Reason for call: {reason_for_call}
- Known information: {known_info}
- Unknown information: {unknown_info}
- Task instructions: {task_instructions}

Behavior rules:
1. You are the customer, not the assistant.
2. Do not call tools.
3. Do not reveal all hidden information at once.
4. Only answer what the agent asks unless it is natural to add context.
5. If the agent proposes an option that violates your goal/constraints, push back.
6. If the agent asks for confirmation, respond according to your goal.
7. If the agent completes the task, thank them and end the conversation.
8. Keep each response concise and natural, usually under 80 words.
```

### 8.4 用户 persona（后续深化）

第一版用默认 normal persona。多 persona（impatient / confused / price_sensitive / policy_questioning）与 user-sim 鲁棒性评测列入后续深化方向。

---

## 9. Rollout 流程设计

### 9.1 单条 trajectory 生成流程

```text
1. 从 τ³-bench Airline 取一个 task
2. 初始化 airline environment 与 database state
3. 用 task.user_scenario 初始化 Interaction（API user simulator）
4. 用 policy.md + tool schema 构造 Qwen3-8B agent 的 system prompt（原生 tool 格式）
5. user simulator 生成第一条用户消息
6. Agent 生成自然语言回复或 tool call
7. tool call -> 调用本地 τ³-bench tool（async）
8. 自然语言 -> 交给 user simulator 继续回应（async）
9. 重复直到成功 / 超步数 / 违规 / 超长
10. τ³-bench verifier 计算 reward
11. 保存完整 trajectory
```

### 9.2 伪代码（概念示意；实际由 verl AgentLoop 调度并发）

```python
async def rollout_one_task(task, policy_client, user_interaction, env):
    messages = env.build_initial_messages(task)            # system: policy + tools
    messages.append(await user_interaction.first_message(task.user_scenario))

    for step in range(MAX_AGENT_ACTIONS):
        action = await policy_client.generate(messages)     # 由 SGLang server 并发服务
        messages.append(action.to_message())

        if action.is_tool_call:
            obs = await env.call_tool(action.tool_name, action.arguments)  # async
            messages.append(obs.to_message())
        else:
            messages.append(await user_interaction.respond(messages))      # async, 带缓存

        if env.is_success() or env.has_policy_violation():
            break
        if count_tokens(messages) > MAX_TOTAL_TOKENS:
            break

    reward_info = env.evaluate(messages)                    # 复用 τ³-bench verifier
    return {
        "task_id": task.id, "messages": messages,
        "reward": reward_info.final_reward,
        "reward_components": reward_info.components,
        "policy_version": policy_client.version,
        "stats": collect_stats(messages),
    }
```

verl AgentLoop 会同时调度多条 `rollout_one_task`，`await` 外部 I/O 时自动切换，GPU 生成不空转。

---

## 10. Trajectory 数据格式

建议保存为 JSONL，每行一条 trajectory：

```json
{
  "task_id": "8",
  "domain": "airline",
  "benchmark": "tau3-bench",
  "policy_model": "Qwen3-8B-Instruct-LoRA",
  "policy_version": 120,
  "user_simulator": {"model": "gpt-4o-mini", "temperature": 0.7, "persona": "normal"},
  "messages": [
    {"role": "system", "content": "airline policy + tool instructions"},
    {"role": "user", "content": "I want to change my flight."},
    {"role": "assistant", "content": "Sure, may I have your reservation ID?"},
    {"role": "user", "content": "It is ABC123."},
    {"role": "assistant", "tool_calls": [{"name": "get_reservation_details", "arguments": {"reservation_id": "ABC123"}}]},
    {"role": "tool", "content": "{...}"}
  ],
  "reward": 1.0,
  "reward_components": {"db_reward": 1.0, "communicate_reward": 1.0, "policy_violation": false, "invalid_tool_call": false},
  "stats": {"num_turns": 9, "num_tool_calls": 4, "num_tokens": 5230}
}
```

---

## 11. Reward 设计

### 11.1 主 reward（第一版：标准）

第一版直接复用 τ³-bench verifier，使用最稳的口径：

```python
reward = db_reward * communicate_reward     # 0/1（依任务 reward_basis）
```

```text
db_reward：最终数据库状态是否达到目标状态
communicate_reward：是否向用户传达必要信息（communicate_info / nl_assertions）
```

> 若任务 `reward_basis` 只含 DB，则 `reward = db_reward`。

### 11.2 违规惩罚（第一版从简）

```text
成功：1    失败：0    严重 policy violation：-1
```

`invalid_tool_call` / `missing_confirmation` / `too_many_turns` 等先**只记录为 diagnostic（用于失败分析）**，第一版不计入 reward，避免 reward 设计过早复杂化。把它们纳入 shaped reward 列入后续深化方向。

### 11.3 关于 LLM-as-judge

本项目主 reward 以 verifier 为准（稳定、可复现、无 judge bias）。但需注意：**τ³-bench 的 communicate_info / nl_assertions 检查本身可能由 LLM 判定**——因此 communicate_reward 并非完全无 LLM。两个选项：
- 接受 τ³-bench 原生的 LLM-based communicate 检查（保持与官方一致）；
- 第一版只用 `DB reward` 起步（更确定性），communicate 留到第二版。

二选一在阶段 0 确认（取决于 τ³-bench 当前实现细节）。

### 11.4 稀疏 reward 风险与第一版对策

二值 reward 下，若同组全 0 或全 1（std=0）则该组无训练信号。第一版对策：

```text
group_size (rollout.n) >= 8        # 不用 2/4，降低全 0/全 1 概率
丢弃 std=0 的组（仅用于 logging）
阶段1 base rollout 先确认 base 有足够非零成功率与方差（GO/NO-GO，见 §15）
```

更强的手段（DAPO dynamic sampling、shaped/partial reward）列入后续深化方向。

---

## 12. GRPO 训练设计（第一版：标准 GRPO）

### 12.1 Group 单位

以同一个 `task_id` 为 group，对每个 task 采样 `rollout.n` 条 trajectory。

### 12.2 Advantage

```python
mean_r = mean(group_rewards); std_r = std(group_rewards)
adv_i = (reward_i - mean_r) / (std_r + 1e-6)
```

同组 reward 全相同（std=0）则丢弃该组用于训练。

### 12.3 Loss Mask（多轮）

只对 Agent 的 assistant token 计 loss：

```text
system / user / tool observation：不算 loss
assistant 自然语言 + assistant tool call：算 loss
```

多轮场景下，整条 trajectory 的所有 assistant turn 共享同一个 trajectory-level advantage（标准 GRPO 做法）。turn-level / step-level credit assignment 列入后续深化方向。

### 12.4 KL 正则（LoRA 省显存技巧）

```yaml
kl_coef: 0.01      # 可在 0.005 / 0.01 / 0.02 之间试
```

**参考模型 = 关闭 LoRA adapter 的同一份 base 模型**，无需额外加载一份 ref model，省显存（verl 在 LoRA 下支持这种 ref logprob 计算）。

---

## 13. On-policy rollout 工程要点（替代原 staleness 章节）

> 第一版为 on-policy，无 staleness 控制需求。本章聚焦"并发隐藏延迟"的工程实现。

### 13.1 并发与延迟隐藏

```yaml
rollout:
  mode: async            # server-mode AgentLoop（AsyncLLMServer）
  n: 8                   # 每个 task 的 group size
  concurrency: high      # 同时在飞的 trajectory 数，喂满 GPU 生成
multi_turn:
  enable: true
```

要点：
- tool handler、interaction handler 必须是**真异步**（aiohttp 等），禁止阻塞式 `requests`。
- 并发数要足够大，使 GPU 生成不因等待外部 I/O 而空转。
- 注意外部 API 的 **rate limit / 并发上限**，必要时加本地并发闸与重试退避。

### 13.2 User-sim 回复缓存

```text
缓存 key = task_id + policy_version + user_model + temperature + conversation_hash
作用：降低 API 成本、提高复现性、便于离线 debug 与失败分析
```

### 13.3 权重同步

HybridEngine 在每个训练 step 后、下一轮 rollout 前，将更新后的 **LoRA adapter** 同步给 SGLang server（verl 内置处理）。on-policy，无需 staleness 过滤。

---

## 14. 轨迹长度控制

2×A100 80GB 显存虽足，但长轨迹训练昂贵。第一版用 4K/8K，不追求 16K。

### 14.1 推荐限制

```yaml
max_agent_actions: 12       # MVP 可先 8；注意 airline 任务常需 5~7 次工具调用，别截太短
max_tool_calls: 8           # 同上，过紧会制造"假失败"
max_total_tokens: 8192      # MVP 4096
max_tool_observation_tokens: 512
max_new_tokens_per_turn: 256
```

### 14.2 Early Stop

```text
任务成功 / 严重 policy violation / 用户明确结束 -> 提前结束
超过 max_agent_actions -> 结束并判失败
超过 max_total_tokens -> 截断或丢弃
```

### 14.3 Tool Observation 压缩

工具返回的长 JSON 压缩为结构化摘要后再入上下文：

```json
{"tool": "get_reservation_details",
 "summary": {"reservation_id": "ABC123", "status": "confirmed", "flight": "UA123", "cabin": "economy", "change_allowed": true}}
```

---

## 15. 训练阶段规划

### 阶段 0：环境接入与数据理解（2-3 天）

```text
拉取 sierra-research/tau2-bench(τ³) repo，读取 airline tasks.json / policy.md / db.json
跑通官方 evaluator，确认 reward 计算流程（特别是 communicate 是否走 LLM，见 §11.3）
确认 train/held-out 划分（见下方"方法学"）
将 τ³-bench tool/user-sim/verifier 包成 verl Tool / Interaction / reward
产出：airline task schema 分析、工具与参数列表、reward 笔记、10 条人工检查样例
```

**方法学（重要）**：用 `split_tasks.json` 或自定义划分，明确**训练 task 集**与**held-out 评测 task 集**，所有报告指标只看 held-out，避免"在训练 task 上评测"的伪提升。

### 阶段 1：Base Model Rollout + GO/NO-GO 闸门（3-5 天）

```text
用未训练 Qwen3-8B-Instruct 在 airline 训练子集上 rollout（原生 tool 格式）
接入 verl Interaction（API user sim）
记录完整 trajectory，计算 base pass@1
```

**GO/NO-GO 判据（决定 RL-zero 是否可行）**：

```text
看 base 成功率与逐 task reward 方差：
  在 group_size>=8 下，有"足够多"的 task 不是全 0 / 全 1 -> GO，直接 RL-zero
  绝大多数 task 全 0（base 几乎不会做） -> NO-GO -> 回退：
     先做难度课程（从 easy task 起） / 或极简 SFT warmup / 或先上 shaped reward
```

产出：base pass@1、平均轮数、平均工具调用数、平均 token、失败类型统计、GO/NO-GO 结论。

### 阶段 2：标准 GRPO 训练（RL-zero）—— 第一版主体（1-2 周）

```yaml
algorithm: grpo                 # 最标准 GRPO
model: Qwen3-8B-Instruct + LoRA
adv_estimator: grpo
group_size (rollout.n): 8
rollout.mode: async             # 并发 on-policy rollout（方案 a）
multi_turn.enable: true
max_agent_actions: 12
max_total_tokens: 8192
user_simulator: API
reward: 复用 τ³-bench verifier（DB[*COMMUNICATE]）
kl_coef: 0.01
clip_range: 0.2
ref: 关闭 LoRA adapter 的 base
```

产出：GRPO checkpoint、训练/reward/success-rate/policy-violation 曲线、API 成本统计、吞吐统计。

### 阶段 3：评测与失败分析（3-5 天）

```text
held-out 上评测 pass@1 / pass^k / success rate / DB reward / communicate reward
tool-call accuracy / policy violation rate / invalid tool-call rate
平均轮数 / 工具调用数 / token / 每成功任务 API 成本 / 训练吞吐
自动失败原因标注（见 §18.4）
对比：Base vs Base+GRPO(RL-zero)
```

> SFT warmup、DAPO/GSPO、跨域、user-sim 鲁棒性等 ablation 属于后续深化方向（§16），第一版不实现。

---

## 16. 后续深化方向（写入计划，后续实现）

> 第一版聚焦"标准 GRPO + RL-zero + 并发 on-policy rollout"跑通并出对比结果。以下为深化路线，按价值/可行性排序，后续逐项实现。

1. **算法 ablation：GRPO → DAPO → GSPO**
   - DAPO 的 **dynamic sampling** 自动丢弃全 0/全 1 组、clip-higher、token-level loss，正对本项目稀疏多轮 reward 痛点；
   - GSPO 的 **sequence-level importance ratio** 在长序列下更稳，也为方案 b（off-policy 异步）打基础；
   - 干净对比哪种在稀疏多轮 reward 下最稳、最省样本。

2. **Reward / credit assignment 深化**
   - shaped / partial reward：DB 子目标命中比例 + communicate 子项比例 + tool 合法性，缓解稀疏；
   - turn-level / step-level advantage 替代 trajectory-level，做更细的多轮信用分配；
   - 将 `invalid_tool_call` / `missing_confirmation` 等 diagnostic 纳入 reward。

3. **off-policy 异步（方案 b）**
   - verl `one_step_off_policy`（staleness=1）/ `fully_async_policy`（`staleness_threshold` 可调）；
   - 在更多 GPU 上研究 staleness 对稳定性/样本效率/吞吐的影响，可借鉴 AReaL 的 staleness guardrail（过滤过旧样本、按版本漂移自适应 clipping、向当前策略正则化）。

4. **泛化与鲁棒性**
   - 严格 train/held-out 已在第一版做；进一步做 **airline→retail 跨域迁移**（零样本/共训）；
   - **user-sim OOD 鲁棒性**：用 A 模型当用户训练、B 模型评测，检测是否过拟合模拟器；多 persona。

5. **SFT warmup（作为 NO-GO 回退）**：若阶段 1 判定 base 太弱，用强 API 模型蒸馏 expert trajectory 做 LoRA SFT 兜底，再 GRPO。

6. **能力/长度扩展**：开启 Qwen3 thinking 模式实验；16K 长轨迹支持；扩大 task 规模。

---

## 17. 工程模块划分（verl 接入）

```text
project/
  configs/
    grpo_qwen3_airline.yaml      # verl GRPO + LoRA + async multi-turn 主配置
    rollout.yaml
  env/
    tau3_airline_tool.py         # τ³-bench tools -> verl Tool（async）
    tau3_airline_interaction.py  # API user simulator -> verl Interaction（async, 带缓存）
    tau3_reward.py               # 复用 τ³-bench verifier -> verl reward fn
  data/
    tau3_airline/                # tasks/policy/db + train/held-out split
    trajectories/
  agents/
    qwen3_prompt.py              # system prompt（policy.md + 原生 tool schema）
  evaluation/
    eval_passk.py
    eval_tool_calls.py
    failure_analysis.py
  scripts/
    run_base_rollout.sh
    run_grpo.sh
    run_eval.sh
```

> 注意：rollout buffer / 并发调度 / 权重同步 / GRPO loss 等大量逻辑由 **verl 提供**，自定义代码主要是三个适配器（Tool / Interaction / reward）+ prompt + 评测。

---

## 18. 关键实现细节

### 18.1 Tool Call 格式

使用 Qwen3 原生 function calling（Hermes 风格）+ SGLang tool parser。解析失败记 `invalid_tool_call`，环境返回错误 observation。

### 18.2 Assistant Token Loss Mask

```python
labels = input_ids.copy()
labels[non_assistant_token_positions] = -100   # system/user/tool observation 不计 loss
```

### 18.3 Rollout / User-sim 缓存

```text
缓存 key = task_id + policy_version + user_model + temperature + conversation_hash
作用：降低 API 成本、提高复现性、便于离线训练分析与 debug
```

### 18.4 Failure Analysis

每条失败轨迹标注失败原因：

```text
wrong_tool / wrong_arguments / missing_user_info / policy_violation
failed_to_communicate / too_many_turns / user_simulator_confusion / context_truncation
```

---

## 19. 风险与解决方案

| 风险 | 解决方案 |
|---|---|
| RL-zero reward 太稀疏，起不来 | 阶段1 GO/NO-GO 先验证 base 有非零成功率与方差；group_size>=8；丢弃 std=0 组；NO-GO 则回退难度课程/shaped reward/SFT |
| API 成本过高 | 缓存 user-sim 回复；先小 task 集调试；评测阶段少量在线 rollout |
| 轨迹过长 OOM | max_agent_actions/max_total_tokens 控制；tool observation 摘要；assistant-only loss；gradient checkpointing；4K 起步 |
| 外部 API 延迟拖慢 rollout | 并发 rollout（方案 a）+ async handler 隐藏延迟；注意 rate limit |
| verl LoRA / async 配置踩坑 | 以 verl 当前版本官方文档为准核对配置键名；先在小规模 sanity run 跑通再放量 |
| user simulator 不稳定 | 固定 system prompt 与 temperature 范围；记录完整 user response；复用 τ³-bench 官方 user-sim |

---

## 20. 项目验收标准

### 20.1 最小可行版本（MVP）

```text
能读取 τ³-bench Airline tasks / 跑 API user simulator（verl Interaction）
能让 Qwen3-8B 以原生格式调用工具 / 保存 trajectory / 用 τ³-bench verifier 计算 reward
能并发 rollout 隐藏延迟 / 能跑通标准 GRPO（小规模）
```

### 20.2 第一版完整版本

```text
并发 on-policy multi-turn rollout（AgentLoop）
group_size>=8 的标准 GRPO（RL-zero）
LoRA + adapter-off KL 参考
4K/8K context
严格 train/held-out 评测：pass@1/pass^k
failure analysis
Base vs Base+GRPO 对比
```

### 20.3 理想结果

```text
GRPO(RL-zero) 相比 Base 有可见的 success rate 提升（个位数~十几个点量级）
工具调用格式错误率下降
policy violation rate 不上升或下降
平均 turn 数不显著增加
```

---

## 21. 简历表述建议

### 中文版

基于 τ³-bench Airline 构建多轮工具调用客服 Agent 的 RL 训练系统：使用 verl 在 2×A100 80GB 上对 Qwen3-8B 做 LoRA + 标准 GRPO 训练（RL-zero，无 SFT），用 API 大模型经 verl Interaction 模拟真实多轮用户反馈，复用 τ³-bench 工具、数据库状态与 verifier 计算奖励。通过 AgentLoop 并发 on-policy rollout 隐藏 tool/API 延迟、用 adapter-off 作为 KL 参考省显存、assistant-token loss mask 与 trajectory 缓存。在严格 train/held-out 划分下评测 pass@1/pass^k、任务成功率、工具调用准确率与业务规则违规率，对比 Base 与 GRPO。

### 英文版

Built a multi-turn tool-using airline customer-service agent RL pipeline on τ³-bench. Trained a Qwen3-8B LoRA policy with standard GRPO (RL-zero, no SFT) using verl on 2×A100 80GB GPUs. Used an API LLM as a multi-turn user simulator via verl's Interaction abstraction, and reused τ³-bench tools, database state and verifier rewards. Implemented concurrent on-policy multi-turn rollouts (AgentLoop) to hide tool/API latency, adapter-off KL reference to save memory, assistant-token loss masking, and trajectory caching. Evaluated pass@1/pass^k, task success rate, tool-call accuracy and policy-violation rate on a strict held-out split, comparing Base vs GRPO.

---

## 22. 推荐实验配置

> 以下为概念配置，**具体键名以 verl 当前版本官方文档为准**。

### 22.1 MVP 配置

```yaml
model: {name: Qwen3-8B-Instruct, dtype: bf16, lora_rank: 16, lora_alpha: 32, lora_dropout: 0.05}
benchmark: {name: tau3-bench, domain: airline, num_train_tasks: ~, held_out: ~}
rollout:
  backend: sglang
  mode: async              # 并发 on-policy（方案 a）
  multi_turn: true
  user_simulator: api
  user_model: gpt-4o-mini
  user_temperature: 0.7
  n: 8                     # group size
  max_agent_actions: 8
  max_total_tokens: 4096
rl: {algorithm: grpo, adv_estimator: grpo, kl_coef: 0.01, clip_range: 0.2, ref: adapter_off}
training: {gpus: 2, gpu_type: A100 80GB PCIe, fsdp: true, micro_batch_size: 1, grad_accum: 8, gradient_checkpointing: true}
```

### 22.2 第一版主实验配置

```yaml
model: {name: Qwen3-8B-Instruct, dtype: bf16, lora_rank: 16, lora_alpha: 32, lora_dropout: 0.05}
benchmark: {name: tau3-bench, domain: airline, num_train_tasks: ~, held_out: ~}
rollout:
  backend: sglang
  mode: async
  multi_turn: true
  user_simulator: api
  user_model: gpt-4o-mini
  user_temperature: 0.7
  n: 8
  max_agent_actions: 12
  max_total_tokens: 8192
rl: {algorithm: grpo, adv_estimator: grpo, kl_coef: 0.01, clip_range: 0.2, normalize_advantages: true, ref: adapter_off}
training: {gpus: 2, gpu_type: A100 80GB PCIe, fsdp: true, micro_batch_size: 1, grad_accum: 8-16, gradient_checkpointing: true, flash_attention: true}
```

---

## 23. 面试讲解重点

1. **为什么 τ³-bench 适合 agentic RL**：有环境、工具、数据库状态和 verifier，不是静态问答。
2. **为什么换 verl 而非 slime**：slime 后端 Megatron 不支持 LoRA；verl LoRA 一等公民、原生多轮 agentic rollout（AgentLoop）。这体现框架选型的判断力。
3. **为什么 RL-zero（不做 SFT）**：用 Qwen3 原生 tool 格式后无需 SFT 学格式；阶段1 验证 base 有非零成功率即可 bootstrap；并配 GO/NO-GO 闸门把关。
4. **"异步"到底指什么**：是 rollout 内部并发隐藏 tool/API 延迟（完全 on-policy），不是 off-policy 解耦——讲清这个区别和"没有免费午餐"。
5. **为什么用 GRPO**：同一 task 多条 trajectory，组内 reward 做 relative advantage，无需 critic。
6. **稀疏 reward 怎么处理**：group_size>=8、丢 std=0 组、GO/NO-GO 验证方差；进一步可上 DAPO dynamic sampling / shaped reward。
7. **怎么省显存**：LoRA + adapter-off KL 参考 + assistant-only loss + gradient checkpointing + 4K→8K。
8. **怎么保证结果可信**：严格 train/held-out 划分，看 pass@1/pass^k、DB/communicate reward、tool-call error、policy violation，而非只看 loss。

---

## 24. 最终推荐路线

第一阶段（本计划主体）：

```text
τ³-bench Airline，明确 train/held-out
Qwen3-8B-Instruct + LoRA
verl 标准 GRPO（RL-zero，无 SFT）
API user simulator（verl Interaction）+ 缓存
并发 on-policy rollout（方案 a）隐藏 tool/API 延迟
4K 起步、group_size>=8
阶段1 GO/NO-GO 闸门 -> 阶段2 GRPO -> 阶段3 held-out 评测 + 失败分析
```

后续深化（§16）：DAPO/GSPO 算法对比、shaped/turn-level reward、off-policy 异步与 staleness 研究、跨域迁移、user-sim 鲁棒性、SFT 回退、thinking/16K 扩展。

---

## 25. 总结

这个项目的价值不在于"把 Qwen3-8B 跑起来"，而在于完整复现一个真实 Agentic RL 训练闭环：

```text
τ³-bench task
  -> verl Interaction（API user simulator）
  -> Qwen3-8B policy agent（原生 tool 调用）
  -> τ³-bench tool/database environment
  -> τ³-bench verifier reward
  -> 并发 on-policy multi-turn rollout（AgentLoop）
  -> grouped trajectories
  -> verl 标准 GRPO update（LoRA, adapter-off KL）
```

它能很好地展示：LLM post-training、Agentic RL、工具调用训练、并发 rollout 系统、verifier-based reward、LoRA 显存优化、长轨迹控制、工程化评测与失败分析，以及**框架选型与训练范式（RL-zero）的判断力**。对面试项目而言，这是比普通数学 GRPO 或简单 function-calling SFT 更有区分度的方向。

---

## 26. Stage 0 实测结论（已完成）

环境接入已跑通并验证（详见 `notes/airline_understanding.md` 与 `README.md`）：

- **运行环境**：conda env `tau2verl`（克隆自 `verl`，保护原 env）。tau2 经 `pip install -e` 装入；`openai` 被 litellm 升到 2.37，但 verl/vllm/sglang/litellm 均 import 正常，无致命冲突；本机走 SOCKS 代理，需补 `pip install httpx[socks] socksio`。
- **外部 API**：user-sim / agent / judge 用 **gpt-5**，经第三方 OpenAI 兼容代理 **`https://yunwu.ai/v1`**（`.env` 设 `OPENAI_API_KEY` + `OPENAI_API_BASE`/`OPENAI_BASE_URL`）。gpt-5 是 reasoning 模型，**不要传 `temperature`**（llm_args 留空）。
- **接入策略**：策略B —— 自定义 `AgentLoopBase` 子类 `@register("tau2_airline")`（本版 verl 无 Interaction 抽象，弃用早期"verl Interaction"说法）。reward 复用 tau2 `evaluate_simulation(EvaluationType.ALL)`，逐组件子分来自 `RewardInfo.reward_breakdown`。
- **任务规模**：airline 共 50 任务，split `train=40 / test=10`（disjoint，held-out 用 test；test 按 gpt-5 功能场景标签分层，每场景≥1，本地 `airline_split.json` 维护，不碰 submodule）。14 个工具。
- **重要 reward 结论**：**全部 50 个 airline 任务的 `reward_basis` 都是 `[DB, COMMUNICATE]`，0 个含 NL_ASSERTION / ACTION**。即 airline 的 reward 完全确定性（DB 哈希比对 + 子串匹配），**gpt-5 NL judge 对 airline 永不触发、训练时 reward 计算零 judge 成本**。§11.3 的"communicate 是否走 LLM"悬念就此清除：airline 不需要 judge；gpt-5 仅用于 user simulator（每轮一次调用）。
- **验收**：`pytest` 7/7 通过；理解笔记产出；AgentLoop 注册通过；sanity rollout（task 0，gpt-5）端到端跑通，多轮 + 工具调用，reward=1.0（DB=1.0, COMMUNICATE=1.0）。

下一步进入**阶段1**：用 Qwen3-8B（下载中）在 train 子集 base rollout，做 GO/NO-GO（看 base 成功率与逐 task reward 方差，决定 RL-zero 是否可直接 bootstrap）。
