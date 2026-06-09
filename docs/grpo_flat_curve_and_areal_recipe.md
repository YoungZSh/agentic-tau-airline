# GRPO 平线第二根因 + AReaL 官方 airline 配方对照

- **日期**:2026-06-08
- **状态**:诊断已完成,改法待验证(下一步:先单独关 KL 做对照实验)
- **关联**:
  - [`verl_2gpu_engine_offload_and_async.md`](verl_2gpu_engine_offload_and_async.md) — 探针落地时的引擎选型:共置+offload 修 wake_up OOM、分卡 fully_async 评估、执行顺序
  - [`usersim_rootcause_deepseek_fix.md`](usersim_rootcause_deepseek_fix.md) — 第一根因(user-sim 污染信号)及 DeepSeek V4 修复
  - [`grpo_zero_reward_diagnosis.md`](grpo_zero_reward_diagnosis.md) — 最初的零 reward / informativeness 诊断
  - [`gdpo_decoupled_advantage.md`](gdpo_decoupled_advantage.md) — 我们的 GDPO 维度解耦方案
  - [`tau3_airline_qwen3_verl_grpo_plan.md`](tau3_airline_qwen3_verl_grpo_plan.md) — 总设计

---

## TL;DR

1. 换上 DeepSeek V4 user-sim(第一根因修复)后,**reward 信号干净了,但 GRPO 曲线照样平**。
   逐任务对比证明:**策略 80 步基本冻在 SFT 起点没动**(37/40 任务纹丝不动,净 Δ=-0.007)。
2. 这暴露了**第二根因:策略根本没更新**。三道"刹车"把它按死了:
   **① KL 锚回 SFT(`kl_loss_coef=0.01`)、② PPO clip 0.2 太窄、③ LoRA r32 + lr 1e-5 步子太小**。
3. 对照 **AReaL 官方 `examples/tau2/config_8b_airline.yaml`**(和我们同款 Qwen3-8B / airline / n=8 /
   官方 `evaluate_simulation` reward):他们在**每一个影响"策略能不能动"的旋钮上都更激进**——
   **全参训练、KL 完全关、clip 0.4、reward shaping、batch 30、从原厂模型 RL-zero**。
4. 下一步最高 ROI 的一刀:**先把 KL 关掉**做对照,看曲线动不动。

---

## 1. 被分析的 run:`20260607_191951`

- 路径:`outputs/qwen3_8b_lora_grpo_tau2_airline_20260607_191951/`
- 起点:SFT cold-start 的 merged 权重(`qwen3_8b_full_sft_tau2_airline_20260605_152720/hf_merged`)
- **本 run 已修复第一根因**:user-sim 从本地 Qwen3.6-35B-A3B(FP8/MoE,角色翻转 78%)换成
  **DeepSeek V4 flash**(角色翻转降到 ~1%,db 信号恢复到 gpt-5 的 ~81%)。详见 user-sim 文档。
- 关键超参(`scripts/train/run_grpo.sh` 默认):`adv_estimator=gdpo`、`lora_rank=32/alpha=64`、
  `actor_lr=1e-5`、`kl_loss_coef=0.01`(`use_kl_loss=True`,`kl_loss_type=low_var_kl`)、
  `train_batch_size=8`、`ppo_mini_batch_size=8`(1 次更新/步,纯 on-policy)、`rollout_n=8`、
  `clip_ratio`=verl 默认 0.2、`total_epochs=20`(≈100 步)。
- 实际跑到 **~step 90**(计划 100)后停止。

---

## 2. 平线证据(信号已干净,曲线仍不动)

### 2.1 验证曲线(8 个验证点,step 10–80,无趋势)

| step | val reward | val db | val comm |
|---|---|---|---|
| 10 | 0.6 | 0.6 | 0.9 |
| 20 | 0.7 | 0.7 | 0.8 |
| 30 | 0.5 | 0.5 | 0.8 |
| 40 | 0.6 | 0.6 | 1.0 |
| 50 | 0.8 | 0.8 | 0.9 |
| 60 | 0.7 | 0.7 | 0.8 |
| 70 | 0.5 | 0.5 | 0.8 |
| 80 | 0.5 | 0.5 | 0.7 |

- **在 0.6 上下随机跳,没有上升趋势。**
- **`val reward` 完全等于 `val db`** → reward 由 db 主导;comm 一直在 0.7–1.0 高位(近饱和)。
  所以"该涨的"全在 db 维度,而 db 维度是平的。

### 2.2 逐任务训练 reward:早期(step 1–20)vs 晚期(step 65–85)

| 变化 | 任务数 | 例子 |
|---|---|---|
| ↑ 升(Δ>0.15) | **1** | task 41: 0.50→0.84 |
| ↓ 降(Δ<-0.15) | 2 | task 2: 1.00→0.84;task 25: 0.56→0.34 |
| · 基本不变 | **37** | — |
| **平均 Δ** | **-0.007** | 约等于零,还略降 |

- **37/40 任务纹丝不动,净变化 -0.007。** step 80 ≈ step 1 ≈ SFT。
- 永久死任务(全程 db=0)依旧死:`14, 18, 23, 29, 32, 33, 39, 42, 44`
  → DeepSeek 下 SFT 也做不出,**GRPO 没法从"组内全 0"里 bootstrap**(零方差=零梯度)。
- 头部任务(`1,4,19,26,27,28,30,43,46,47,48` 等 ~12 个)已 ~1.0 → 无上升空间。
- 真正可学的中间档(`7=0.09, 8=0.25, 12=0.16, 22=0.19, 24≈0, 35=0.25`)**有方差、本该能学,但没动**。

### 2.3 策略更新内部量(最后几步)

| 量 | 值 | 含义 |
|---|---|---|
| `actor/grad_norm` | **~0.02** | 梯度极小 → 几乎没在更新 |
| `actor/pg_loss` | ~1e-8 | on-policy 下组归一 advantage 均值≈0,符合预期 |
| `actor/ppo_kl` | **0.0** | 1 次 on-policy 更新,策略在被测 token 上没动 |
| `actor/kl_loss` | ~5e-4(×coef 0.01) | KL 项在把策略拉回 ref |
| `actor/entropy` | ~0.5–0.57 | 中等,没坍缩也没爆 |
| `critic/advantages` | mean≈0, max~2.1, min~-2.7 | advantage 已被 std 归一到单位方差量级 |

> ⚠️ 排查中曾被一个**显示截断假象**误导:把 `pg_loss=-2.6e-08` 截成 `-2.6`,一度误判为"loss 剧烈震荡/不稳定"。
> 修正后真实 `pg_loss≈1e-8`,**无不稳定**——是"策略冻住"而非"训练发散"。记此教训:打印浮点别按字符截断。

---

## 3. 第二根因诊断:策略被三道刹车按死

信号干净了还是平,说明问题不在 reward,而在**优化没把干净信号转成策略更新**。三道刹车,按影响排序:

1. **⭐ KL 锚回 SFT**:`use_kl_loss=True` + `kl_loss_coef=0.01`。SFT cold-start 后,KL 的参考策略
   **就是 SFT 自己**(LoRA 关 adapter = SFT-merged base)。等于"先 SFT 定住,再不许偏离" → 冻。
2. **⭐ PPO clip 太窄**:verl 默认 `clip_ratio=0.2`,每步策略移动幅度被夹半。
3. **⭐ LoRA 步子太小**:r32 + lr 1e-5。LoRA 通常需要比全参大一个量级的 LR(~1e-4)才动得起来;
   `grad_norm 0.02` 就是症状。

辅助因素:`train_batch_size=8`(组数少、噪声大)、`total_epochs=20`(只 ~100 步,偏短)。

> 注:`norm_adv_by_std_in_grpo=True`(组内 std 归一)已把 advantage 归一到单位方差,所以
> **reward 的绝对尺度不是我们的杠杆**(见 §5.4)。我们的杠杆是 KL / clip / LoRA 步长。

---

## 4. 上一个 run 的试错时间线(信号链)

| 阶段 | 现象 | 结论 |
|---|---|---|
| 本地 Qwen3.6-35B-A3B user-sim | GRPO 曲线平,db reward 被砸 ~90% | **第一根因**:user-sim 角色翻转/镜像循环污染信号 |
| A/B 量化(`outputs/usersim_ab_20260607`) | 本地 vs gpt-5:db 差 ~90% | 坐实 user-sim 是污染源 |
| 换 DeepSeek V4 flash | 角色翻转 78%→~1%,db 恢复 gpt-5 的 ~81%,~1/100 成本 | 第一根因**已修复** |
| 重训 `20260607_191951`(干净信号) | **曲线仍平**(§2) | 暴露**第二根因**:策略冻住,GRPO 没在学 |
| 删除 qwen `enable_thinking=False` 死代码 | — | 对 DeepSeek 无效,清理掉 |

---

## 5. AReaL 官方 airline 配方对照(同款 Qwen3-8B)

来源:`github.com/areal-project/AReaL` → `examples/tau2/`,通过 gh API 读取。
我们的直接同款是 **`config_8b_airline.yaml`**(Qwen3-8B、domain=airline、n_samples=8、
官方 `evaluate_simulation` reward)。**逐项已到代码坐实,不靠记忆。**

### 5.1 四个关键事实(已验证)

1. **没有 SFT,从原厂 Qwen3-8B 直接 RL-zero**
   `config_8b_airline.yaml:55-57`:`actor.path: Qwen/Qwen3-8B`、`init_from_scratch: false`
   (=加载预训练权重,不是 SFT ckpt;`ref` 也是同一个 Qwen3-8B)。`train.py` 只跑 `PPOTrainer`,
   无任务级 SFT。底座是 Qwen 官方 post-train 过的 instruct/thinking 模型(非 `-Base`)。
   **→ 反衬我们"SFT 饱和 + KL 锚回 SFT"的平线诊断:他们证明 airline 上不做 SFT 直接 RL 能涨。**

2. **训练数据 = tau2 官方 split 的 task_id**(轨迹在线 rollout 生成)
   `train.py:36-47`:`registry.get_task_splits_loader(domain)()` → `splits["train"/"test"]` → task_id。
   数据行只有 `{task_id, split}`,和我们一样多轮轨迹在线生成。**用官方 split,不是我们自定义的 40/10。**
   两个细节值得抄:`train.py:50-54` 任务<128 复制补齐;`train.py:62-63` `group_filter`
   `rewards.mean() <= 0.95` 才保留 → **已学会(组均值>0.95)的组动态丢弃**(informativeness 过滤)。

3. **user-sim = Qwen2.5-72B dense / bf16 / temp 0**(不是 fp8、不是 MoE)
   `config:127-131`:`user_llm: openai/self-hosted-Qwen2.5-72B`、`{temperature:0.0, max_completion_tokens:512}`。
   `README:65-74` 部署:`--model-path Qwen/Qwen2.5-72B --tool-call-parser qwen25
   --chat-template qwen3_nonthinking.jinja --dp-size 2 --tp-size 4`(8 卡,无量化 → bf16)。
   **正好是我们踩坑那个 user-sim 的反面**(我们:35B-**A3B** MoE / **FP8** / 带 thinking)。
   印证:user-sim 要够强够稳(dense 72B / temp0)才不污染信号。

4. **reward shaping 公式**(见 §5.4)。

### 5.2 逐项对比

| 旋钮 | AReaL 8B airline | 我们 | 对"策略不动"的影响 |
|---|---|---|---|
| **训练参数** | **全参**(archon:d8) | LoRA r32 @ lr1e-5 | ⭐根本:全参移动幅度远大于 r32 |
| **KL** | `kl_ctl: 0.0`(关) | `kl_loss_coef=0.01`(开,锚 SFT) | ⭐最大:KL 把策略拽回 SFT |
| **PPO clip** | `eps_clip: 0.4` | 默认 **0.2** | ⭐每步更新被夹半 |
| **LR** | 1.7e-5(全参,constant) | 1e-5(LoRA) | LoRA 应 ~1e-4,偏小一个量级 |
| **batch(任务/步)** | 30 | **8** | 组数少 → 噪声大、平 |
| **reward shaping** | `(r-0.5)×10`,clip±20 | 无 | §5.4:被 std 归一抵消,非主因 |
| **adv norm** | `mean/std: batch` | `norm_adv_by_std_in_grpo`(组内 std) | — |
| **起点** | base Qwen3(无 SFT) | SFT-merged | SFT 饱和 + KL 锚 = 双重"别动" |
| **步数** | 500 / 200 epoch | ~100 / 20 epoch | 偏短 |
| **informativeness 过滤** | `group_filter` 丢已学会的组 | 无 | 我们浪费已饱和任务的算力 |
| user-sim | Qwen2.5-72B dense bf16 | DeepSeek V4 flash | 都 OK(信号已洗净) |
| agent thinking | `enable_thinking: True` | think-preserving | 一致 |
| 其他 | `use_decoupled_loss`、`recompute_logprob`、`rejection_sampling(ratio>5 丢)`、`invalid_format_penalty=0.1`、`max_head_offpolicyness=2`(异步) | 无 | 次要 |

### 5.3 一句话

我们的 **GDPO 维度解耦方向没错**(用按维度归一解决组内零方差),但被三道刹车按死:
**KL 锚回 SFT、clip 0.2 砍更新、LoRA r32+lr1e-5 步子太小**。
AReaL 是用"**全参 + KL 关 + clip 0.4 + 大 batch + RL-zero**"硬推上去的。

### 5.4 reward shaping 确切公式(并说明为何对我们不是主杠杆)

`areal/trainer/ppo/actor.py:167-171`:
```python
reward_score = (reward_score + self.reward_bias) * self.reward_scaling   # bias=-0.5, scaling=10
reward_score = torch.clip(reward_score, max=self.reward_clip, min=-self.reward_clip)  # clip=20
```
即 `r' = clip((r - 0.5) × 10, ±20)`,把 `r∈[0,1]` 映成 `[-5,+5]`:
- **bias=-0.5(recenter)**:全正 [0,1] → 有符号;失败拿到**真负**信号而非"小正"。
- **scaling=×10(rescale)**:抬高 reward 相对 KL/entropy/value 的量级。
- clip±20 在此不触发(范围才 ±5),只是保险丝。

**caveat(诚实):** airline config 同时开 `adv_norm: {mean/std: batch}`。一个**全局仿射变换
(加常数+乘常数)在"减均值/除标准差"下基本被抵消** → 在这套配置里 reward shaping 杠杆不大,
真正起作用是在 `adv_norm=null` 或 KL 开着时。我们用 `norm_adv_by_std_in_grpo=True`,尺度同样被除掉,
**所以 reward shaping 不是我们的主杠杆**——主杠杆仍是 KL / clip / LoRA 步长。

---

## 6. 改法(分两档)

### A. 2 卡现实可行版(保留 LoRA,改 4 个旋钮)
1. **KL 关掉**:`KL_LOSS_COEF=0` 或 `actor.use_kl_loss=False` —— 最高优先,先解开 SFT 锚。
2. **clip 放宽**:`+actor_rollout_ref.actor.clip_ratio=0.4`。
3. **LoRA 步子加大**:`ACTOR_LR=5e-5`(甚至 1e-4)、`LORA_RANK=64 LORA_ALPHA=128`。
4. **batch / 步数加大**:`TRAIN_BATCH_SIZE=16`(看显存),`TOTAL_EPOCHS` 拉到能跑 ~300 步。

### B. 对齐 AReaL 正解(需算力)
全参 GRPO / 从 base 起。我们已有全参 SFT 的 FSDP 配置,但 2×A100 同时塞全参 8B + vllm rollout 非常紧
(AReaL 用 24 卡)——这正是当初选 LoRA 的原因。作为后续在更多卡上的目标。

### 建议的下一步
**先做实验①:只把 KL 关掉,其余不动**,做最干净的对照。若 KL 锚是主因,关掉后曲线应立刻开始动。
据此再决定要不要叠加 clip/LR/rank/batch。

---

## 7. 关键文件 / 证据位置

- 本 run:`outputs/qwen3_8b_lora_grpo_tau2_airline_20260607_191951/logs/train_20260607_191951.log`
- 训练脚本:`scripts/train/run_grpo.sh`(权威入口,每个旋钮 env-var 可覆盖)
- AReaL 参照(已读):`examples/tau2/{config_8b_airline.yaml, train.py, agent.py, utils.py, README.md}`、
  `areal/trainer/ppo/actor.py:167-171`、`areal/api/cli_args.py:1469-1514`
