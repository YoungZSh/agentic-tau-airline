# 把 GRPO 训练迁到 verl fully_async(分卡 / 1 训练 + 1 rollout)+ 在单训练卡上打通 AReaL 式 decoupled-PPO

- **日期**:2026-06-09
- **状态**:**机制全部验证通过**(on-policy / AReaL 式异步 / 真 decoupled 三档都在 2 卡 1+1 上 rc=0 跑通);**Phase 3 真长跑(学不学 + 实测加速比)尚未做**
- **关联**:
  - [`verl_2gpu_engine_offload_and_async.md`](verl_2gpu_engine_offload_and_async.md) — 引擎选型 + 当初"分卡只有 fully_async 一条路"的判断(本工作即其落地)
  - [`grpo_flat_curve_and_areal_recipe.md`](grpo_flat_curve_and_areal_recipe.md) — AReaL 配方对照的由来
  - memory:`fully-async-2gpu-validated`、`verl-2gpu-engine-constraints`、`gpu-rehold-immediately-after-run`、`areal-probe-first-learning-result`

---

## TL;DR

1. 共置(`main_ppo` HybridEngine,2 卡分时)单步 ~42min,`gen` 占 65%、训练卡在 rollout 阶段空转。改成 **1 卡 FSDP 训练 + 1 卡 vLLM rollout** 的 **verl `fully_async_policy`**,让 rollout(N+1) 与 train(N) 重叠。
2. **我们 repo 里几乎不写 Python**:新增一个启动脚本 + 一个 ~50 行的 monkeypatch + agent_loop 里 ~15 行真逻辑。重活全在 verl 现成框架里。
3. 三档机制都在 2 卡 1+1 上 **rc=0 验证通过**:
   - **Phase 1** on-policy(grpo,staleness=0):跑通,确认分卡机制(Ray 1+1 放置 / NCCL 参数同步 / MessageQueue / 我们的 agent loop + DeepSeek user-sim)。
   - **Phase 2** AReaL 式异步(gdpo + staleness=0.5 + partial + bypass):跑通,**GDPO 子分穿过 MessageQueue**、`trainer/idle_ratio` 0.76→0.61(重叠收益兑现)。
   - **真 decoupled-PPO**(bypass_mode=False + rollout_is=token):靠一个 monkeypatch 在**单训练卡**上跑通,staleness=0 下 `ppo_kl=0/clipfrac=0`(重算 old_log_prob 正确)。
4. **关键认知**:"decoupled 要 ≥2 训练卡"是 **verl 的实现约束(`save_model_to_cpu` 绑了 FSDP DTensor 分片),不是算法本质**。算法上 decoupled 只需"参数快照 + 一次前向",单卡可做 → 用 monkeypatch 退回普通 `state_dict` 拷贝即可。
5. **预估加速比 ~1.3×(区间 1.2–1.5×)**,不是数量级飞跃:2 卡 1:1 固定卡比、每阶段砍到 1 卡变慢、rollout 是瓶颈。要 2× 以上得加卡(更多 rollout 卡做 2:1)。

---

## 1. 改了哪些文件

| 文件 | 改动 | 性质 |
|---|---|---|
| `scripts/train/run_grpo_fully_async.sh` | **新增 ~245 行** | 分卡 fully_async 启动脚本(`run_grpo.sh` 的兄弟);每个旋钮 env 可控 |
| `src/tau2_airline_verl/rollout/agent_loop.py` | **+35 / −1**(真逻辑 ~15 行) | 唯一的业务逻辑改动:聚合并透出 `min/max_global_steps` 版本戳 |
| `src/tau2_airline_verl/patches/sitecustomize.py` | **新增 ~50 行** | decoupled-on-1GPU monkeypatch(`bypass_mode=False` 时自动激活) |

**没有改 `third_party/`(只读 submodule)**,也没改 `src/` 里除 agent_loop 外的任何东西。

---

## 2. 架构与数据流

```
GPU_train (1 卡)             GPU_rollout (1 卡)
┌─────────────────┐         ┌──────────────────────┐
│ FSDP2 Actor     │         │ vLLM server (tp=1)    │
│ LoRA r32 / 冻结 │◄──NCCL──┤ gpu_mem 0.8           │
│ base, 无 offload│  只同步 │ 跑 Tau2AirlineAgentLoop│
│ fsdp_size=1     │ adapter │ (policy=vLLM,        │
└─────────────────┘         │  tool=tau2 env,      │
                            │  user=DeepSeek API)   │
                            └──────────────────────┘
```

- 入口:`verl.experimental.fully_async_policy.fully_async_main`(`hybrid_engine=False`),两组 Ray worker 各占各卡,**不分时**。
- 四组件:**Rollouter**(连续流式产样本)/ **MessageQueue**(缓冲、带版本戳的 replay buffer)/ **Trainer**(拉 mini-batch 训)/ **ParameterSynchronizer**(checkpoint-engine,`backend=nccl`,LoRA 下只推 adapter)。
- **生产**:每个 prompt 的 n 个 response(一个 GRPO 组)凑齐 → 作为一个带 `global_steps` 版本戳的 sample 进 MQ;只在"队列满"或"超前到 staleness 上限"时暂停(`_should_pause_generation`)。
- **消费**:Trainer 攒够 `ppo_mini_batch × require_batches` 个组 → 一次 PPO 更新 → 每 `trigger_parameter_sync_step` 步 NCCL 推权重。
- **异步**:`staleness_threshold` 允许 rollouter 用旧权重超前生产,与训练重叠;`staleness=0` 近乎同步流水。
- **调度/缓冲/版本戳/staleness 与 AReaL 同构**;差别只在 off-policy 修正强度与中断粒度。

---

## 3. 两个关键的集成修复

### 3.1 `min/max_global_steps` 版本戳(agent_loop.py,~15 行)
fully_async 的 trainer 组 batch 时(`detach_utils.py:153`)对每条轨迹做 `max_global_steps − min_global_steps`(staleness/partial 追踪),缺失则 `None − None` 崩溃。标准 `ToolAgentLoop` 会透出这俩,我们的自定义 loop 没有。
- 修复:在 `_rollout_trajectory` 每轮 `generate()` 后聚合 `min(of mins)/max(of maxes)`,挂到 `AgentLoopOutput.extra_fields`,缺省兜底 0(共置无害)。

### 3.2 decoupled-on-1GPU monkeypatch(sitecustomize.py,~50 行)
- **问题**:`bypass_mode=False`(真 3 策略 decoupled + rollout_is)要给 proximal 锚点做参数快照 `fsdp2_sharded_save_to_cpu`,它**断言 FSDP2 分片的 DTensor 参数**(需 ≥2 训练卡)。单训练卡(`fsdp_size=1`)无 DTensor → `AssertionError: No DTensor-type parameters found`。
- **本质**:这是 verl 实现假设,不是算法要求。decoupled 只需"参数快照 + 一次前向"。
- **修复**:monkeypatch `verl.utils.fsdp_utils.fsdp2_sharded_save/load_to_cpu`,无 DTensor 时退回**只存可训练参数(LoRA adapter;冻结 base 跨版本不变,跳过)的普通 `state_dict` 拷贝**;多卡场景原样委托原实现。
- **注入**:`PYTHONPATH` + `sitecustomize`(Python 在每个 Ray worker 启动时自动 import)。**必须延迟**(meta_path 钩子,在 verl 自然 import `fsdp_utils` 时才打补丁)——**不能在 startup eager `import torch`/verl**,否则 Ray 还没给 worker 设 `CUDA_VISIBLE_DEVICES`,会触发 `No CUDA GPUs are available`。
- **激活**:`run_grpo_fully_async.sh` 在 `BYPASS_MODE=False` 时自动设 `PYTHONPATH`+`VERL_DECOUPLED_1GPU_PATCH=1`;否则零影响。

---

## 4. 验证结论(全部 rc=0 / 0 traceback)

| 档位 | 配置 | 结果 |
|---|---|---|
| Phase 1 on-policy | grpo,staleness=0,trigger=1,partial=False,bypass | 跑通;首次 NCCL 同步 ~40s;step:0/1/2;`aborted_ratio=0` |
| Phase 2 AReaL 式异步 | **gdpo** + staleness=0.5 + partial=True + bypass | 跑通;**GDPO 优势算出(db/comm/db_comm 穿过 MQ)**;`trainer/idle_ratio` 0.76→**0.61** |
| 真 decoupled(单卡) | **bypass_mode=False** + rollout_is=token + staleness=0 | 跑通;**`ppo_kl=0.0`/`pg_clipfrac=0.0`** sanity(重算 old_log_prob≈rollout,目标正确);`grad_norm` 健康 |

- decoupled patch 另有**离线单测**:只存可训练参数 + 恢复正确 + 非 DTensor 路径 OK。
- 期间两次 OOM 已查实为**外部用户的 `bench_latent.py`(~26GB)反复抢占物理 GPU 1**(trainer 53GB + 26GB > 80GB),非 patch 问题;干净卡上 53GB trainer 放得下,一次过。

---

## 5. 旋钮速查(`run_grpo_fully_async.sh`,均 env 覆盖)

```bash
# 资源(2 卡 1+1)
GPU_LIST="1 2"  TRAIN_GPUS=1  ROLLOUT_GPUS=1  ROLLOUT_TP=1  ROLLOUT_GPU_MEM_UTIL=0.8

# 算法(默认对齐已验证的 areal_probe:base Qwen3-8B, clip0.4, no-KL, lr1e-4)
ADV_ESTIMATOR=grpo            # grpo(默认,单标量)/ gdpo(子分已验证可穿 MQ)
USE_KL_LOSS=False  CLIP_RATIO=0.4  ACTOR_LR=1e-4

# 异步(默认 on-policy;打开即 AReaL 式)
STALENESS_THRESHOLD=0         # >0 才有重叠提速;verl 建议 <1
TRIGGER_SYNC_STEP=1           # >1 = 流式重叠(Mode 2)
PARTIAL_ROLLOUT=False

# off-policy 修正
BYPASS_MODE=True              # True=2 策略(单卡默认);False=真 decoupled(自动激活 monkeypatch)
ROLLOUT_IS=null               # token/sequence,仅 bypass_mode=False 生效
ROLLOUT_RS=null               # seq_mean_k3 等;拒绝采样,bypass 下也能用、单卡安全

# 显存(训练卡独占默认不 offload;被抢/吃紧时再开)
PARAM_OFFLOAD=False  OPTIMIZER_OFFLOAD=False  PPO_MAX_TOKEN_LEN_PER_GPU=20480
```

两套典型启动:
```bash
# 稳妥(bypass,2 策略)
bash scripts/train/run_grpo_fully_async.sh
# 真 decoupled(对齐 AReaL,可放心推高 staleness)
BYPASS_MODE=False ROLLOUT_IS=token STALENESS_THRESHOLD=0.5 TRIGGER_SYNC_STEP=4 \
  bash scripts/train/run_grpo_fully_async.sh
```

---

## 6. off-policy 修正:三种手段(对我们 2 卡的可用性)

| 手段 | 机制 | 单卡可用 | 备注 |
|---|---|---|---|
| **bypass(2 策略)** | `old_log_prob = rollout logprob`,比率 π_θ/π_rollout | ✅ | 默认;低 staleness 下接近 decoupled |
| **Rejection Sampling(rollout_rs)** | 把太 off-policy 的样本**整条 mask** | ✅ | 糙(丢数据)、稳;`bypass_mode=True` 即可 |
| **真 decoupled + IS(rollout_is)** | 重算 π_prox 锚点,按 IS 权重重加权(AReaL 做法) | ✅(靠本次 monkeypatch) | 原本要 ≥2 训练卡;现单卡可跑 |

AReaL 的算法灵魂就是 decoupled(3 策略);bypass 是其简化前身。本工作让三者在 2 卡上都可用。

---

## 7. 加速比预估(~1.3×,区间 1.2–1.5×)

- 共置基线:step 2610s = `gen` 1702s(65%,API/长尾主导)+ train ~907s,**串行**。
- 分卡:step ≈ `max(gen_1卡, train_1卡)` + (1−重叠率)·min(...)。`gen` API-bound → 1 卡上 ≈ 不变;`train_1卡` ≈ 1.6–1.8× × 907 ≈ 1500s;重叠率由 staleness 决定。
- **上不去 AReaL 的 2–3× 的原因**:① 1:1 卡比固定、rollout 是瓶颈(idle 0.61)、训练卡空转无法回收;② 每阶段砍到 1 卡变慢;③ partial 只到单 decode 级。
- **真值待 Phase 3 实测**(`timing_s/step` vs 2610s、`trainer/idle_ratio`)。

---

## 8. 运维要点(本次踩的坑)

- **跑完立刻占卡**:`run_grpo_then_hold.sh` / `gpu_hold.py`(伪装成 `VLLM::EngineCore`)。本次所有 smoke 都用 `( timeout 1800 <run>; setsid <hold> & )` 包好——退出/崩/超时都自动重占。机器共享、空槽秒被抢(本次被 `bench_latent.py` 多次抢 GPU 1)。
- **zsh 不分词**:`for p in $pids` 会把多 PID 当一个 token;杀进程用 `... | while read -r p` 逐行处理。
- **sitecustomize 延迟注入**:别在 startup eager import torch/verl(Ray 设 device 前碰 CUDA → "No CUDA GPUs")。

---

## 9. 尚未做 / 下一步

- **Phase 3 真长跑**(需 2 张无人抢的卡):判读 ① val reward/db 是否仍从共置基线 0.5/0.5 爬到 ~0.6/0.7(off-policy 没搅坏学习);② 每等效步墙钟 vs 共置 42min + `trainer/idle_ratio`。
- bypass 与 decoupled 都已就绪,可分别跑做对照。
- staleness 梯度:从 0.3–0.5 起,盯 `ppo_kl/pg_clipfrac/entropy/response_length` 失稳信号,稳再往 1 爬(我们比 AReaL 更保守,因单卡无法做更强的修正预算)。

---

## 10. 证据 / 关键代码位置

- 启动脚本:`scripts/train/run_grpo_fully_async.sh`
- 版本戳修复:`src/tau2_airline_verl/rollout/agent_loop.py`(`_rollout_trajectory` 聚合 + `run()` 透出 `min/max_global_steps`)
- decoupled patch:`src/tau2_airline_verl/patches/sitecustomize.py`
- verl 侧:`verl/experimental/fully_async_policy/{fully_async_main,fully_async_rollouter,fully_async_trainer,detach_utils}.py`、`verl/experimental/separation/{ray_trainer,engine_workers}.py`、`verl/utils/fsdp_utils.py`(`fsdp2_sharded_save/load_to_cpu`)、`verl/trainer/config/algorithm.py`(`RolloutCorrectionConfig`)
- 验证日志:`outputs/fa_smoke_smoke2/`(Phase 1)、`outputs/fa_bsmoke_bsmoke2/`(Phase 2)、`outputs/fa_dsmoke_dsmoke5/`(decoupled)
