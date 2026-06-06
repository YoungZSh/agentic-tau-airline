# 解耦优势:DB / COMMUNICATE 各自组内归一化(GDPO)

> 适用场景:τ²-airline 多轮 GRPO,reward = `DB × COMMUNICATE`(全 50 任务的 `reward_basis` 都是这两项)
> 入口:`scripts/train/run_grpo.sh` 的 `ADV_ESTIMATOR` 开关(默认 **gdpo**;`ADV_ESTIMATOR=grpo` 回退)
> 验证:2026-06-05 —— 离线证明「乘积全 0 但 COMM 有方差」的组,GRPO 零梯度、GDPO 非零梯度

本文记录为什么把奖励的乘积拆成 **DB、COMMUNICATE、DB×COMMUNICATE 三个分量、各自在 GRPO
组内单独归一化再相加**,以及如何用 verl 0.9 原生的 `gdpo` advantage estimator 实现它——
**不改只读 submodule**。

---

## 1. 问题:乘积奖励让一半的组「零方差 = 零梯度」

τ²-airline 的官方标量奖励是各 `reward_basis` 分量的**乘积**:本域全部 50 个任务的
basis 都是 `{DB, COMMUNICATE}`,所以

```
reward = DB × COMMUNICATE
```

- `DB ∈ {0,1}`:轨迹结束时 FlightDB 的 hash 是否与标准答案一致(二值)。
- `COMMUNICATE ∈ [0,1]`:该报给用户的关键信息(金额、航班号……)报全了没有。

GRPO 的优势是**组内**的 `(r − mean_g)/std_g`。一旦一个组(同一任务的 `n` 条 rollout)的
乘积**全为 0**,组内方差为 0 → 优势恒 0 → **该组对梯度无贡献**。而乘积特别容易全 0:只要
DB 这一项还没学会(始终为 0),无论 COMMUNICATE 报得多好,乘积都是 0。

`docs/grpo_zero_reward_diagnosis.md` 里那批 COMMUNICATE-locked 任务(7/14/18/23,「必须把
1628、327、1786…这些数字报给用户」)正是如此:SFT 冷启动后实测 **COMMUNICATE=1.0、DB=0.0**,
乘积仍是 0。模型其实在 COMMUNICATE 维度上**有进步、有区分度**,但乘积把这个信号抹平了。

---

## 2. 方案:逐维解耦归一化再相加(GDPO)

不要先乘起来再归一化,而是把每个维度**当作独立奖励,先各自在组内归一化,再加权求和**:

```
Step 1  逐维组内归一化(就是对每一维各跑一次 GRPO):
        A_DB    = (DB    − μ_g(DB))    / (σ_g(DB)    + ε)
        A_COMM  = (COMM  − μ_g(COMM))  / (σ_g(COMM)  + ε)
        A_PROD  = (PROD  − μ_g(PROD))  / (σ_g(PROD)  + ε)     # PROD = DB × COMM = 官方标量

Step 2  加权求和:   A_sum = w_DB·A_DB + w_COMM·A_COMM + w_PROD·A_PROD     # 默认 w = (1,1,1)

Step 3  batch 级 whiten:  A_final = whiten(A_sum, response_mask)
```

为什么这样能救上面那种组:即使 PROD 全 0(`A_PROD=0`)、DB 全 0(`A_DB=0`),只要
COMMUNICATE 在组内有 0/1 的区分,`A_COMM` 就非零 → `A_sum` 非零 → **有梯度**。

**为什么保留冗余的第三维 `DB×COMM`?** 前两维是「分别把 DB、COMM 学好」的稠密信号;第三维
把优化目标**重新对齐到真正的任务成功**(两者同时为真),避免模型只去刷某一维而牺牲整体。
组内归一化天然自动平衡量纲,无需手调 λ/α/β。这一点也是为什么不在标量层做 `product + 小额
partial credit` 那种 shaping——那需要手调权重,且会改变奖励口径。

> **它救不了什么:** 如果一个组在**所有**维度上都零方差(DB、COMM、PROD 全相同,例如某任务
> 每条 rollout 都彻底失败),GDPO 仍是零梯度——它不会无中生有。它只在「**某一维有方差、但
> 乘积没有**」时补出信号。所以它与 SFT 冷启动**互补**:SFT 先把 COMMUNICATE 那一维从「全 0」
> 顶到「有 0/1 区分」,GDPO 再把这个区分变成梯度。

---

## 3. 用法

### 3.1 通过 run_grpo.sh(默认已开)

```bash
bash scripts/train/run_grpo.sh                 # 默认 ADV_ESTIMATOR=gdpo
ADV_ESTIMATOR=grpo bash scripts/train/run_grpo.sh   # 回退到单标量 GRPO(三个 key 自动被忽略)
```

权重默认等权,可调:

```bash
GDPO_REWARD_WEIGHTS='[1.0,1.0,1.0]' bash scripts/train/run_grpo.sh
GDPO_REWARD_KEYS='[db,comm,db_comm]'                # 维度名(须与 AgentLoop 发出的 key 一致)
```

### 3.2 最终落到的 verl 参数

```
algorithm.adv_estimator=gdpo
algorithm.gdpo_reward_keys=[db,comm,db_comm]
algorithm.gdpo_reward_weights=[1.0,1.0,1.0]
```

---

## 4. 接线:子分量怎么从 AgentLoop 流到 advantage 阶段

verl 0.9 原生支持 GDPO,所以**只接两根线、不动 submodule**:

1. **`Tau2AirlineAgentLoop`(`rollout/agent_loop.py`)发出三个子分量。** 把它们放进
   `AgentLoopOutput.extra_fields["reward_extra_info"]`:

   ```python
   extra_fields={
       "reward_extra_info": {"db": db_score, "comm": comm_score, "db_comm": reward_score},
       ...
   }
   ```

   verl 会在 batch 组装时把 `reward_extra_info` 的每个 key **flatten 进 `non_tensor_batch`**
   (`third_party/verl/.../agent_loop.py:978-982`)。注意:verl **只取第一条样本的 key 集合**,
   所以三个 key 必须在每条轨迹上都出现——我们在算 reward 前把 `db/comm` 默认成 `0.0`,即使
   reward 计算抛异常也不会漏 key。`db_comm` 直接复用官方标量 `reward_score`(= DB×COMM)。

2. **`run_grpo.sh` 切到 `gdpo` 并给出三个 key。** GDPO 的 `compute_gdpo_outcome_advantage`
   (`core_algos.py:361`)从 `non_tensor_batch[key]` 取每一维,用 `batch["prompts"]` /
   `batch["attention_mask"]` / `uid` 把标量放回最后一个 response token,逐维调
   `compute_grpo_outcome_advantage` 再加权求和。

**reward 标量(`rm_scores`)不变**,仍是官方乘积——GDPO 只在 advantage 层用子分量,token-level
reward 被它忽略(只在没给 key 时回退)。因此:

> **eval 完全不受影响。** held-out 评估走 tau2 自己的 `run_domain` + 官方 `evaluate_simulation`
> 乘积口径,base 与 trained 的数字仍可直接比较。改的是「怎么算梯度」,不是「怎么打分」。

---

## 5. 离线验证(实测)

构造一个最能说明问题的组:**DB 恒为 0、COMMUNICATE 在组内 0/1 变动、乘积全 0**,分别用
GRPO(对乘积)和 GDPO(对三维)算优势:

| estimator | 每条 rollout 的优势(last token) | max\|adv\| | 结论 |
|----|----|----|----|
| GRPO on `DB×COMM` | `[0, 0, 0, 0]` | 0.0 | **零梯度,卡死** |
| GDPO `(db\|comm\|prod)` | `[−0.96, +0.96, −0.96, +0.96]` | 0.96 | **非零,且符号正确** |

GDPO 给出的优势里,报了数字(COMM=1)的 rollout 拿正优势、没报的拿负优势——正是我们想要的
学习信号。此时 `A_DB=A_PROD=0`(无方差),全部信号来自 `A_COMM`,whiten 后落到 ±0.96。

复现(单卡、无需 optimizer/GPU,直接调 verl 的 kernel):

```python
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage, compute_gdpo_outcome_advantage
# db=[0,0,0,0], comm=[0,1,0,1], db_comm=db*comm=[0,0,0,0]; 同一 uid 组
# GRPO(token_level_rewards=db_comm) -> 全 0; GDPO(keys=[db,comm,db_comm]) -> ±0.96
```

---

## 6. 注意事项 / 坑

- **三个 key 必须每条轨迹都有**:verl 用第一条样本的 key 集合当全 batch 的标准
  (`agent_loop.py:980`)。reward 异常时 `db/comm` 已默认 0.0,不会漏 key。
- **冗余维 `db_comm`** 与前两维不独立(是它们的乘积),这是**有意**的——强调「同时成功」。
  它在某些组可能零方差(全成功/全失败),那一维贡献 0,无害。
- **全维度零方差的组仍无梯度**:GDPO 不无中生有(见 §2 末)。这正是为什么要先做 SFT 冷启动。
- **回退安全**:`ADV_ESTIMATOR=grpo` 时三个 key 仍被 AgentLoop 发出,但 GRPO 分支不读它们,
  纯无害开销。
- **算法出处**:GDPO = Group reward-Decoupled Normalization Policy Optimization,verl 代码内
  引用 arxiv 2601.05242。本仓库用的是 verl 0.9 内置实现,未自研 estimator。
