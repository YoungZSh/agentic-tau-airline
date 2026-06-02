# τ³-bench Airline × verl GRPO 起始训练配置

> 本文档汇总一次关于 Airline 数据集与 RL 训练细节讨论得出的**起始配置建议值**及其依据，
> 供落实到 `configs/grpo_qwen3_airline.yaml` / `configs/rollout.yaml` 时参考。
> 当前这些 config 仍是 stage-0 占位（placeholder），下表给出开训前需要补齐/确认的关键项。
>
> 原则：**所有"轮数/温度"类的值都是起点，不是终点**——真正决定何时停、训多深的是 held-out test 曲线，不是预设遍数。

---

## 0. 一句话定位

数据是 τ³-bench Airline（train 30 / test 20，不重叠），无官方示范轨迹，纯 outcome reward
（`reward = DB × COMMUNICATE`），因此走 **RL-zero（无 SFT）+ GRPO + LoRA**。下表为该设定下的起始配置。

---

## 1. 起始配置值

| 参数 | 建议起点 | 理由 |
|---|---|---|
| `data.train_batch_size` | **30**（full-batch） | 数据仅 30 条，full-batch 下 `1 step = 1 epoch = 全部数据`，梯度方差最小、最稳；数据大才需 mini-batch |
| `actor_rollout_ref.rollout.n`（GRPO group size） | **8**（可试 16） | reward 是 0/1 稀疏，组太小易出现一组全 0/全 1 → advantage=0 无梯度 |
| `actor_rollout_ref.rollout.temperature`（policy 采样） | **0.8 ~ 1.0** | RL 探索需要多样性，这是"该开高温度"的地方（**不是** UserSimulator） |
| `actor_rollout_ref.actor.ppo_epochs` | **1**（最多 2） | GRPO 标准；同批数据多步更新会 off-policy 漂移、易崩 |
| `actor_rollout_ref.actor.kl_loss_coef` | **0.01**（现状保留） | 把策略拉住别偏离 base 太远，抗过拟合 / reward-hacking |
| LoRA `rank / alpha` | **16 / 32**（现状保留） | 低秩限制容量，天然减缓过拟合，允许多过几遍数据 |
| `trainer.total_epochs`（= step 上限，full-batch 下） | **100 ~ 150** | 配合 early stop；实务常在 50~100 step 内整体饱和 |
| test 评估间隔 | **每 ~10 step** | 监控泛化，及时发现过拟合 |

### UserSimulator / NL judge

| 选项 | 说明 |
|---|---|
| 现状：**gpt-5** | 更强，但贵；且 gpt-5 是 reasoning 模型，**不接受 temperature 参数**（`usersim.yaml` 注释已说明），只能用内置采样 |
| 备选：**gpt-4.1 @ temperature=0.0** | 对齐 tau2 官方默认、可复现、能命中 UserSim 缓存；要可控/省钱时优先 |

> 注意区分两个温度轴：**policy rollout 温度该高（探索）**；**UserSimulator 温度该稳（忠实演剧本、可复现）**。
> tau2 官方默认 agent/user/judge 温度统一为 `0.0`（`third_party/tau2-bench/src/tau2/config.py:19/20/25`）。

---

## 2. 评测口径（对齐 tau2 官方，用于 report）

| 参数 | 值 | 出处 |
|---|---|---|
| `num_trials` | **4**（算 **pass^4**） | tau2 论文 / leaderboard（`config.py` 默认 1，论文用 4） |
| `max_steps` | **200** | `config.py:4` |
| `max_errors` | **10** | `config.py:5` |
| `seed` | 固定（如 300） | `config.py:6` |

用官方协议跑 held-out test，才能和 tau2 leaderboard 横向对比。

---

## 3. 训练循环术语（verl，避免混淆）

- **1 step** = 处理 `train_batch_size` 个 prompt（rollout + reward + 更新一次）；**≠ 一次梯度更新**
  （内部还按 `ppo_mini_batch_size` 切 mini-batch 多次 `optimizer.step`）。
- **1 epoch** = 全部数据过一遍 = `⌈N / train_batch_size⌉` 个 step
  （`verl/trainer/ppo/ray_trainer.py:438`：`total_training_steps = len(dataloader) * total_epochs`）。
- **full-batch（train_batch_size=30）下**：`1 step = 1 epoch = 全部 30 条各过一遍`。
- `ppo_epochs`（同批数据反复更新几遍）与"数据过几个 epoch"是**两根不同的轴**。

---

## 4. 训练多深 / 何时停（核心：不靠遍数，靠信号）

GRPO 的 advantage 会**自动退火**：某任务的 n 条 rollout 一旦全对/全错 → advantage=0 → 该任务梯度归零。
因此"每条 train 几次"没有统一值——简单题几遍即饱和、难题可过上百遍仍在学。

**停止判据（三选一触发即考虑停）：**
1. **held-out test pass^k**：train 涨但 test 见顶/回落 → 过拟合，停。（硬指标）
2. **非饱和任务占比**：仍处于 `0 < group_reward < 1` 的任务比例掉到很低 → 边际收益趋零。
3. **KL 散度**持续走高 → 偏离 base 太远，reward-hacking 前兆。

**抗过拟合兜底**：LoRA（低秩限容量）+ KL 正则（`kl_loss_coef=0.01`）。

---

## 5. 可选：扩任务池以延长可训深度

从 `db.json`（500 用户 / 2000 预订）派生**任务 / persona 变体**，扩大任务池：
任务越多越不易整体饱和，每条的"重复压力"更小，能更稳地多训而不死记。
这比拉高 UserSimulator 温度更安全（后者会污染 reward、破坏缓存）。
