# 2 卡引擎选型:共置 + offload vs 分卡 fully_async(含 wake_up OOM 修复)

- **日期**:2026-06-08
- **状态**:决策已定 —— **现在用"共置 + offload"把 AReaL 复现探针跑出结论;分卡 fully_async 作为"配方验证为正后"的吞吐优化项**
- **关联**:
  - [`grpo_flat_curve_and_areal_recipe.md`](grpo_flat_curve_and_areal_recipe.md) — 平线第二根因 + AReaL 配方对照(探针的由来)
  - [`tau3_airline_qwen3_verl_grpo_plan.md`](tau3_airline_qwen3_verl_grpo_plan.md) — 总设计

---

## TL;DR

1. AReaL 复现探针(RL-zero + grpo + 无 KL + clip0.4 + full-batch=40 + lr1e-4,LoRA,2 卡)**连续两次崩在 vLLM `wake_up` 的 CUDA OOM**(run `..._135840`、`..._152750`)。
2. 根因不是抢卡(第二次 GPU 1,2 是满血空卡):**verl HybridEngine `sleep_level=1` 不挪 vLLM 权重 + 我们 `param_offload=False` + full-batch=40 的常驻数据(320 条 rollout,是旧 batch=8 的 5×)→ 训练完 vLLM 醒来要不回它的 40GB**。
3. **修复:开 FSDP offload**(rollout 时把 actor 参数/优化器换出到 CPU,腾给 vLLM)。LoRA 下换出的是冻结 base,每步 ~1-2s,可忽略。已做成 env 开关。
4. 顺带摸清了 verl 的**引擎/入口版图**和**分卡可行性**:
   - 标准 `main_ppo`(我们用的,已 deprecated)和推荐的 `main_ppo_sync` **都被硬断言锁死在共置**;
   - verl 里**唯一**实现"训练/rollout 分卡"的是实验性 `fully_async_policy`;
   - 2 卡做 1+1 **显存上可行**(LoRA),但**只能走重型 fully_async**,且其规模型收益对 2 卡存疑——**真正对我们划算的是"rollout/train 重叠"**(我们 rollout 慢且 API-bound,共置时训练卡大量空转)。
5. **决策顺序**:先用便宜的 offload 探针拿到"配方会不会学"的证据;**会学**再投入做 fully_async 分卡优化,**不学**就先查为什么不学。不在未验证的配方上过早造吞吐系统。

---

## 1. 两次 wake_up OOM 的复盘

| run | 卡 | 现象 | 真因 |
|---|---|---|---|
| `..._135840` | 默认落到 GPU **1,2**(被占) | `wake_up` CUDA OOM,0 step | 抢卡(没传 `GPU_LIST` → wrapper 默认 "1 2",而 1,2 有旧 hold) |
| `..._152750` | GPU **1,2 满血空卡** | **仍** `wake_up` CUDA OOM,0 step | **不是抢卡** → HybridEngine 睡/醒显存冲突 |

第二次坐实了根因。日志关键线索:
- `WARNING: Setting the sleep level to 1 may cause a memory overflow`(verl 自己的警告);
- 崩在 `checkpoint_manager.update_weights → server_handle.wake_up`;
- **好消息**:崩前 base 基线 val 打出来了 —— **`step:0 reward=0.3, db=0.4`**(RL-zero 的 base 有信号,对上 AReaL ~0.2-0.3 起点)。

### 机理:共置 2 卡上有"两个租客"抢显存

RL 训练时同一对 GPU 上有两套东西:
- **vLLM rollout**:权重 + KV(`gpu_mem_util=0.5` → 40GB/卡);
- **FSDP actor**:LoRA(冻结 base 16GB + adapter + 优化器)。

verl HybridEngine 让两者分时,但 **`sleep_level=1` 睡着了 vLLM 权重也不挪**;我们又 `param_offload=False`,FSDP 全程占着卡。**full-batch=40 的常驻 rollout 数据(320 条 vs 旧 64 条)吃光余量** → vLLM `wake_up` 重新分配 KV 时 OOM。旧 batch=8 能过,只是常驻小、余量够。

---

## 2. 修复:FSDP offload(已落地为 env 开关)

**offload 在干嘛**:rollout 阶段把 actor 的参数+优化器从 GPU 搬到 CPU,GPU 基本只剩 vLLM;训练阶段再搬回。两个租客**轮流**用显存,而不是同时挤。

| 时刻 | 不开 offload(旧默认) | 开 offload |
|---|---|---|
| Rollout | vLLM + actor 都常驻 GPU | actor 在 CPU,GPU 给 vLLM |
| **wake_up** | actor 占着 → vLLM 要不回 40GB → **OOM** | actor 在 CPU → vLLM 拿回 → ✅ |

- **代价**:每步多一次 GPU↔CPU 搬运。LoRA 只搬冻结 base(~16GB),~1-2s/步,相对分钟级步时可忽略。
- **不变**:梯度/更新/结果完全一样,纯"时间换显存"。
- 历史:**之前所有 GRPO 跑(含 SFT)都是 offload=False**;这是第一次开。旧的能不开是因为 batch=8 常驻小;**full-batch=40 在 2 卡共置下必须开**。
- AReaL 自己 `enable_offload: false`,是因为它**训练/rollout 在不同卡**(24 卡);2 卡共置没那条件,必须 offload。

### 代码改动(本轮)
- `scripts/train/run_grpo.sh`:`use_kl_loss` / `clip_ratio` / **`param_offload` / `optimizer_offload`** 全做成 env 可控(默认保持旧行为 False/True/0.2,不影响别的跑法)。
- `scripts/train/run_grpo_areal_probe.sh`(新):固化整套 AReaL 配方,**默认开 offload**;一行启动。

---

## 3. verl 引擎/入口版图(本轮摸清)

| 入口 | 状态 | 架构 | 用我们? |
|---|---|---|---|
| **`main_ppo.py`** | **已 deprecated**(文件头注明 v0.8.0 起由 sync 替代) | 共置 HybridEngine(`RayPPOTrainer`) | ✅ 当前在用,**能跑、deprecated≠坏** |
| **`main_ppo_sync.py`** | 官方推荐替代 | **仍是共置** + TransferQueue 零拷贝/ReplayBuffer | ❌ 未用;需装 `TransferQueue==0.1.6`(tau2verl 没装) |
| **`fully_async_policy/fully_async_main.py`** | experimental | **真·分卡**(Rollouter/Trainer 解耦) | ❌ 未用;重型、规模型 |

**硬约束**:`ray_trainer.py:334` —— `assert self.hybrid_engine, "Currently, only support hybrid engine"`。**标准 `main_ppo` 和 `main_ppo_sync` 被硬断言锁死共置**,改 `hybrid_engine=false` 直接起不来。`recipe/one_step_off_policy/` 在本版**不存在**。→ **verl 里想分卡,只有 fully_async 一条路。**

要点:
- `main_ppo_sync` 改的是**数据管线**(零拷贝/ReplayBuffer/每 prompt 不同 n/agent loop 多输出),**不是 GPU 架构**——迁过去**照样得 offload**,不解决 OOM。迁移=独立工程(装 TQ + 换入口 + 验我们的 agent loop/reward/GDPO)。
- `main_ppo_sync` 的 `AgentLoopWorkerTQ(AgentLoopWorker)` 是**子类**,我们的 `Tau2AirlineAgentLoop` 注册大概率能沿用(兼容性好兆头)。

---

## 4. 2 卡分卡(1+1)可行性 —— 纠正 + 真账

**先纠正**:之前说"2 卡不行"讲过头了。**显存上 LoRA 8B 做 1+1 装得下**:
- 训练卡(1 GPU,无 FSDP 分片):冻结 base 16GB + 小 LoRA 优化器 + 激活 ≈ 56-60GB,塞进 80GB;
- rollout 卡(1 GPU,独占):权重 16GB + ~60GB KV,并发反而比共置(0.5 池)更强。

**真正的卡点是软件**:verl 没有"轻量同步 1+1"模式,要分卡只能吞下 fully_async 的整套异步机制。

**真正对我们划算的收益 = rollout/train 重叠**(比泛泛的"规模论"实在):
- 共置每步是**串行** `rollout(分钟级、大量时间等 DeepSeek API)→ train(GPU 算)`;rollout 阶段**训练卡空转**;
- 分卡 + 异步后两者**重叠**:trainer 训第 N 批时,rollouter 在另一张卡产第 N+1 批;每步墙钟 `rollout+train` → `max(rollout,train)`;
- 我们 rollout 又慢又 API-bound、训练卡空转时间长 → **重叠能回收的多**。这是 fully_async 对 2 卡仍可能划算的真实理由。

---

## 5. fully_async 自研可行性评估("希望大吗")

不是从零写,而是**改 verl 现成的 `fully_async_policy`**(Rollouter / Trainer / MessageQueue / NCCL 参数同步 / partial rollout 都有)。

**乐观面:**
- ✅ 它**要求 AgentLoop server 模式** → 我们 `Tau2AirlineAgentLoop` 正好是;
- ✅ user-sim 是**外部 API**,不占 GPU,分卡更干净;
- ✅ **LoRA 让参数同步很小**(只同步 adapter),比全参分卡好做。

**风险/硬骨头:**
1. experimental 代码,基本只在**大规模**验过,**2 卡迷你 1+1 很可能没人跑过** → 踩没踩过的坑;
2. 把自定义 **agent loop + reward + GDPO** 接进异步路径(reward_score/reward_extra_info 走 MessageQueue 是否通);
3. **NCCL 参数同步**跨两进程组在 2 卡上建通信,fiddly;
4. 1 卡训练**无 FSDP 分片**,激活更挤,可能要砍 `ppo_max_token_len_per_gpu`;
5. **staleness 调参**:要 ≥1 步 off-policy 才有重叠收益(AReaL 也 `max_head_offpolicyness=2`,可接受,但偏离纯 on-policy)。

**诚实结论**:能跑通——**中上概率**;真提速——**很可能**(我们这负载重叠收益实在);代价——**是个真项目(以天计,实验性代码上调)**。

---

## 6. 决策:先验证配方,再造优化系统

**当前一个"能学"的训练结果都还没有**(探针一直 OOM,不知道 AReaL 配方在 LoRA 下学不学得动)。在未验证的配方上先造吞吐优化器 = 过早优化。

顺序:
1. **现在(~25-33元)**:共置 + offload 把探针跑出**一个结论** —— 从 base 基线 0.3 往上爬不爬。
2. **若涨** → **才值得**投入做 fully_async 分卡优化(目标明确:给已验证能学的配方提速)。里程碑:先让 verl 自带 fully_async 示例**在 2 卡 1+1 跑通**(验机制)→ 换进 Tau2 agent loop/reward → 调 staleness 拿重叠。
3. **若不涨** → 省下几天,先查"为什么不学"(LR/rank,再不行全参)。

### 探针启动(共置 + offload)
```bash
GPU_LIST="1 2" AGENT_NUM_WORKERS=32 bash scripts/train/run_grpo_areal_probe.sh
```
判读:base 基线已知 = 0.3(db 0.4);盯 `critic/score/mean` 与每 2 步 val 是否从 0.3 往上爬。

---

## 7. 证据位置
- OOM 根因/基线:`outputs/qwen3_8b_lora_areal_probe_20260608_152750/logs/`
- 共置硬约束:`third_party/verl/verl/trainer/ppo/ray_trainer.py:334`
- 入口弃用:`third_party/verl/verl/trainer/main_ppo.py`(`@deprecated`)
- 分卡实现:`third_party/verl/verl/experimental/fully_async_policy/`(`README_zh.md`、`fully_async_main.py`)
- 脚本:`scripts/train/run_grpo.sh`(offload/KL/clip env 开关)、`scripts/train/run_grpo_areal_probe.sh`
