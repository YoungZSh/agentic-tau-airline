# Rollout 加固:user-sim thinking 修复 + 防御性容错

> 时间:2026-06-06
> 触发:`outputs/qwen3_8b_lora_grpo_tau2_airline_20260606_015313` 在 **step 17 的 rollout 阶段崩溃**,整个训练 job 退出
> 配置:Qwen3-8B(SFT-merged)+ LoRA rank 32 + GDPO;user-sim / NL judge = 本地 `qwen3.6-usersim`(Qwen3.6-35B-A3B-FP8,vllm 0.19.1,:8011)
> 数据来源:`outputs/<run>/logs/train_*.log`、`logs/rollouts/*.jsonl`(每步 64 行 = batch 8 × n 8 = 1024 条/16 步)

本文记录从"一条退化 rollout 崩掉整个训练"出发,逐层定位到 **user-sim 不可靠 + think 泄漏 + 缺少防御**,并落地的一组修复。**核心结论:一条 rollout 的偶发退化(空生成 / user-sim 误判)绝不应该 kill 整个 batch。根因是本地 user-sim 在关闭 thinking 时会随机误判场景、吐裸控制 token;治本是给 user-sim 开 thinking(+ reasoning-parser),治标是给整条链路补上重试 + 异常排除,让个别失败降级为"丢弃这条样本"而不是"训练崩溃"。**

---

## 0. 崩溃现场

`train_*.log` 末尾的 traceback:

```
src/.../rollout/tau2_session.py:107  respond_user
  -> tau2/user/user_simulator.py:235  _generate_next_message -> generate
  -> tau2/utils/llm_utils.py:233      validate_message
AssertionError: Message must have content or tool calls. got AssistantMessage
```

第 16 步成功 dump、第 17 步 rollout 中崩溃。异常顺着 `asyncio.gather` 冒泡到 `ray_trainer.fit()` → **整个训练进程退出**。

---

## 1. 问题一:空 assistant 轮 → 整个 batch 崩溃

**根因链路**:某条 rollout 里 policy 生成了一个 **strip 后为空** 的 assistant 轮(vLLM 返回空生成 / 立即 EOS),既无 tool call、content 也是空串。`build_tau2_assistant("")` 按 `content if content else None` 把它变成 `content=None`、`tool_calls=None` 的 `AssistantMessage` → 交给 user simulator 时 tau2 的 `validate_message` 断言失败。

**关键脆弱点**:64 条并发 rollout 里只要 **1 条** 退化,就 kill 掉整个 step。前 16 步只是没踩到。

> 旁证:reward 路径不受影响 —— `evaluate_simulation` 不调 `validate_message_history`,`set_state` 重放只处理 `is_tool_call()` 的消息。崩溃**只**发生在 user simulator 这一条路径。

**修复**(`agent_loop.py`):user-sim 分支里,可见内容为空且无 tool call → 判为退化轮,用 `AGENT_STOP` 优雅结束轨迹,空 token 仍留在 loss(mask=1),evaluator 照常给低 reward。

---

## 2. 问题二:think 块泄漏进 user-sim 与 evaluator

排查 #1 时发现:Qwen3 thinking 是**开着**的(step 16:812/831 个非空 `<think>` 块),而 hermes tool parser **只剥 `<tool_call>` 标签、不剥 `<think>`**。于是 `record_assistant(content)` 把 **think + 回复整段**存进 tau2 `AssistantMessage.content`:

- user simulator 读到了 agent 的**私有推理**(崩溃日志里 `content: <think>` 即实证);
- COMMUNICATE / NL_ASSERTION judge 也在对"含推理的文本"打分,而非用户可见回复。

**修复**(`conversion.py` + `agent_loop.py`):新增 `strip_think()`,取最后一个 `</think>` 之后的可见回复(未闭合 → 空)。**tool 分支和 user 分支都用剥离后的 `visible`** 喂给 tau2 侧;**token 轨迹保持原样**(think 仍计入 loss)。两条数据通路彻底解耦:

| 数据通路 | 含 think? | 来源 |
|---|---|---|
| policy 的 loss + 上下文(`prompt_ids`/`response_mask`) | ✅ 有 | 原始 `output.token_ids` |
| user-sim 看到的 / judge 评的(`session.messages`) | ❌ 已剥 | `record_assistant(visible)` |

退化轮 guard 也改为基于 `visible`,因此同时覆盖"真空生成"和"think-only 无回复"两种退化。

---

## 3. 问题三:user-sim 重复回话 / 误判(根因)

### 3.1 现象与复现

score=0 的 rollout(row 63,Sophia 任务 task 4)初始 prompt 直接暴露:

```
assistant: Hi! How can I help you today?      ← 固定问候
user:      ###OUT-OF-SCOPE###                  ← user-sim 开场第一句就是裸控制 token
assistant: <think>...</think> Hello! How can I help you today?
user:      Hello! How can I help you today?    ← 开始镜像 agent，陷入死循环直到 max_turns
```

全 1024 条 rollout 量化:

| 现象 | 数量 |
|---|---|
| 第一条 user 消息是裸控制 token(`###OUT-OF-SCOPE###` 等) | **65 / 1024(6.3%)** |
| ≥3 次 "How can I help" 回弹死循环 | 41(其中 22 由首条控制 token 种下) |

### 3.2 这是模型问题,不是 prompt 问题

- **Bug 1(框架)**:`Tau2Session.start()` 对**第一条** user 消息**不检查 `is_stop`**(只有循环内的中途 user 轮才查,首轮漏了),所以首轮控制 token 没被终止、反而喂给 policy,种下死循环。
- **根因(模型)**:task 4 的 `reason_for_call` 信息充足("编故事骗取不该给的赔偿"),且 turn-0 prompt 在 8 个样本间**逐字相同**(指南静态、场景按任务固定、问候是常量、persona=None)。**同一输入,有的样本正常入戏、有的吐 `###OUT-OF-SCOPE###`** → 纯属解码采样方差 / 模型不可靠(temp>0 + vllm MoE 连续批处理即使 temp=0 也不确定),被 **thinking 关闭**(一步直出、无推理草稿)进一步放大。

### 3.3 治本:给 user-sim 开 thinking + reasoning-parser

模型 chat template(`chat_template.jinja:147-152`)在 `enable_thinking=True` 时把 `<think>\n` **预填进 prompt**,模型输出 `...推理... </think>\n\n答案`(输出里只有 `</think>`)。给足预算会正常闭合并产出干净答案。所以 `--reasoning-parser qwen3` 能把 `<think>` 分离到 `reasoning_content`,`content`(tau2 读的用户回复)保持干净。

**A/B 实测(Sophia task 4,各 32 次 turn-0 采样)**:

| 配置 | 正常入戏 | control-token 异常 | 其它 |
|---|---|---|---|
| thinking **OFF**(旧) | 26/32 | **6**(OUT-OF-SCOPE×4 / TRANSFER×1 / STOP×1) | 0 |
| thinking **ON**(新) | 31/32 | **0** ✅ | 1×空回复 |

**结论:开 thinking 把误判型控制 token 从 6/32 压到 0。** 代价:推理啰嗦(一句回复带 ~1–3k 推理 token、偶发自我纠正循环),且引入新的 ~3% **空回复**(病态循环耗尽预算、`</think>` 未闭合 → `content` 空)。

---

## 4. 防御性容错(让个别失败不杀训练)

### 4.1 退化回复重试 guard

`Tau2Session._generate_user()`:user-sim 返回**空 content** 或**首轮裸控制 token** 时重试(默认 3 次,`TAU2_USER_DEGENERATE_RETRIES` 可调),重试间**回滚 user-sim 状态**避免历史污染;耗尽则当作 stop 优雅终止。`start()` 和 `respond_user()` 都走它。

- 空回复(~3%):重试后 ≈0;
- 首轮控制 token(Bug 1):重试救回,救不回则干净终止;
- 中途控制 token:仍按 `is_stop` 正常终止(合法信号,不重试)。

### 4.2 异常排除:个别 rollout 出错 → 丢弃,不崩

把轨迹构建抽成 `agent_loop.py: _rollout_trajectory`,`run()` 用 `try/except` 兜住。**任何环节**(user-sim 重试后仍异常、tool exec、generation)抛异常 →

- **该 rollout 排除**:`response_mask` 全置 0 → policy/KL/entropy loss(都按 response_mask 过滤)零贡献,等价于"不参与梯度更新";`reward=0`;
- **打日志**:`logger.exception("tau2_airline rollout EXCLUDED (task=%s)")`(带 traceback,可 grep);
- **计数**:`extra_fields["excluded"]`(0/1),盯系统性故障;
- **惰性样本兜底**:即使异常发生在首次生成前,也产出"非空 prompt + 1 个 masked eos",避开 `prompt_ids[-0:]` 返回整个 prompt 的坑;
- **never crash the batch**。

顺序:**退化先重试(4.1 + litellm `num_retries`)→ 耗尽才排除(4.2 是 backstop)**。

### 4.3 连接池(对冲 thinking 带来的并发压力)

`.env` 设上 `TAU2_LLM_MAX_CONNECTIONS=256 / TAU2_LLM_MAX_KEEPALIVE=128 / TAU2_USER_TIMEOUT=300`,重新激活 `litellm_setup.py` 的连接池放大(此前 `TAU2_LLM_MAX_CONNECTIONS` 未设 → no-op)。thinking-on 后每次调用持连接更久,64 路并发下易触发历史 step-36 那类 `PoolTimeout`。

---

## 5. user-sim 防御版图(改前 → 改后)

| 层 | 覆盖 | 改前 | 改后 |
|---|---|---|---|
| 传输异常 / 超时 | litellm `num_retries` + `timeout` | 走默认 | 走默认(timeout 收到 300s) |
| 并发 PoolTimeout | 连接池放大 | ⚠️ no-op | ✅ 已激活(4.3) |
| 语义退化(空 / 首轮控制 token) | 重试 | ❌ 无 | ✅ `_generate_user`(4.1) |
| 中途 stop / 失控循环 | `is_stop` + `max_user_turns`/`response_length` | ✅ | ✅ |
| reward 异常 | try/except → reward=0 | ✅ | ✅ |
| **user-sim 调用本身抛异常** | 排除该 rollout | ❌ **崩 batch** | ✅ 排除 + 日志 + 计数(4.2) |

---

## 6. caveat:排除样本对 GRPO baseline 的影响

被排除样本 `response_mask=0` 不回传梯度,但以 `reward=0` 进入 GRPO 同组(n=8)的 baseline(mean/std)。

- **单组偏差**:mean 被拉低 ≈ `mean/8`(用 step-16 的 reward mean≈0.23、std≈0.42 → advantage 平移 ≈ 0.07);std 被抬高 → 衰减该组 advantage。两者方向相反、部分抵消,量级 `O(mean/n)`,可忽略。
- **放大因子**:受影响的是"组"不是"样本",一条坏样本污染整组 → 受影响组占比 ≈ `n × p`(约 8× 单条出错率)。p=1% → ~8% 的组;p=3% → ~21%。每组偏差仍 ~0.07,整体影响小。
- **边界(偏安全)**:整组全挂 → std=0 → advantage=0 → 该组零梯度,不注入错误信号。彻底消除偏差需改 verl 的 advantage 计算(只读 submodule),低 p 下不值得。

**看门狗**:`excluded` 计数正常应 ~0–几个 %;某步飙到 >10% 说明 user-sim/基建系统性故障,应停下查根因(那才是真正会毁训练的情况),而非容忍偏差。

---

## 7. 改动清单

| 文件 | 改动 |
|---|---|
| `src/.../rollout/conversion.py` | 新增 `strip_think()`(§2) |
| `src/.../rollout/agent_loop.py` | think 剥离 + 空轮 guard(§1/2);抽出 `_rollout_trajectory` + 异常排除(§4.2) |
| `src/.../rollout/tau2_session.py` | `_generate_user()` 退化重试 guard(§4.1) |
| `src/.../env/airline_interaction.py` | thinking 解耦成 `TAU2_USER_DISABLE_THINKING`(默认开),按模型名 gate(gpt-5 不发 extra_body)(§3.3) |
| `scripts/setup/serve_usersim.sh` | 加 `--reasoning-parser qwen3`(§3.3) |
| `scripts/setup/verify_local_models.py` | 重写成 parser 的 before/after 探针 |
| `.env` | thinking 解耦(judge OFF / user-sim ON)+ 连接池 vars(§4.3) |
| `tests/{test_conversion,test_agent_loop,test_tau2_session}.py` | `strip_think` / 退化重试 / 排除机制 回归(共 22 passed) |

---

## 8. 运维 / 续跑

```bash
# 1) 重启 user-sim server(加载 --reasoning-parser qwen3)
kill <old_serve_pid>; bash scripts/setup/serve_usersim.sh
# 2) 验证:期望 ALL OK(user-sim 含推理但 content 干净；judge 干净)
conda run -n tau2verl python scripts/setup/verify_local_models.py
# 3) 从最近 checkpoint 续跑(save_freq=10 → 有 step-10 ckpt；step-16 未存)
```

> 待观察:开 thinking 后 user-sim 延迟上升(每轮多吐 1–3k 推理 token),建议续跑后用 smoke 量一下 rollout wall-clock 与 `excluded` 计数。
