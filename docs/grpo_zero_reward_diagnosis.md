# GRPO 训练「全 0 任务 / 曲线不动」诊断

> 诊断时间:2026-06-03 ～ 06-04(基于 step 1–35 的 rollout JSONL)
> 训练配置:Qwen3-8B + LoRA + 标准 GRPO(RL-zero,无 SFT),train=40 / test=10 split
> 数据来源:`outputs/<run>/logs/rollouts/*.jsonl`(每步 64 行 = batch 8 × n 8)

本文总结在排查「训练曲线为什么是平的」时发现的一组问题。**核心结论:rollout 流水线本身是健康的,曲线不动的主因不是「模型学不动」,而是 40 个 train 任务里有 14 个(35%)在每个 group 内 8 次采样全 0 → 组内零方差 → GRPO 零 advantage → 零梯度,白烧 35% 的 rollout 算力。而这 14 个全 0 任务里,真正属于「模型能力问题」的只有一半,另一半是环境配置和 reward 口径在信号到达模型之前就把它掐断了。**

---

## 0. 先确认:rollout 机制本身是正常的 ✅

随机抽样的轨迹(如 task 7)完全健康,排除了流水线 bug:

- **格式干净**:`<think>` / `<tool_call>` / `<tool_response>` / user 轮规整交替,Qwen3 hermes 工具格式正确。
- **工具真实命中 DB**:`get_reservation_details("XEHM4B")` 返回了真实乘客 `daiki_muller_1116` 的完整预订。
- **user simulator 正常**(gpt-5 应答)、**轨迹正常终止**、**reward 正常赋值**。
- 数量对(每步 64 行),**几乎没有截断**(轨迹末尾被切的只有 4/256)。
- 组内**有方差**的任务(中等难度)确实在涨:task 4 (.71→.92)、13 (.38→.75)、27 (.62→.79)、11、25、45、28 … —— GRPO 在这些任务上正常工作。

> 一个易被采样噪声误导的点:**per-step 均值(0.03–0.36 间震荡、无趋势)不能用来判断有没有在学**。每步只抽 8/40 任务、任务难度差异极大,均值主要被「这一步抽到哪些任务」主导。要看真实学习信号,必须**追踪同一任务跨 epoch 的 reward**。

---

## 1. 现象:14 个任务全程死锁在 reward = 0.00

按 task_id 追踪同一任务跨 step 的 reward,剥离采样噪声后,以下 **14 个任务(占 train 集 35%)从 step 1 到 35 全程 0.00**:

```
7, 12, 14, 15, 18, 23, 24, 29, 32, 35, 37, 39, 42, 44
```

它们组内零方差 → 零梯度,既学不到东西,又占满 rollout 算力。**这不是随机分布,而是系统性地集中在 split 最难的一端**,且可精确归为三种**不同层次**的病因。

---

## 2. 三种病因(最重要的结论)

| 病因 | 层次 | 任务 | 本质 | 处置方向 |
|---|---|---|---|---|
| ① turn-cap 数学不可解 | **环境配置层** | 39, 42, 44 | 环境根本没给够步数做完 | 改 config / 删换任务 |
| ② COMMUNICATE 维度锁死 | **reward 设计层** | 7, 14, 18, 23 | 模型做对了、reward 却判 0 | 改 reward 口径 |
| ③ 纯策略不会做 | **模型能力层** | 12, 15, 24, 29, 32, 35, 37(+7) | 真·模型不会(GRPO 本职) | SFT 冷启动 / 继续 RL |

**关键洞察:①②③ 会制造一模一样的「组内全 0 → 零梯度」表象**,让训练曲线看起来「模型在这些任务上学不动」。但只有 ③ 是真·模型不会;① 是环境没给机会,② 是做对了不给分。而 ③ 又恰恰被 ①② 饿死——它本可靠「偶尔撞对一次」产生梯度慢慢学会,但 ① 让它撞不到、② 让它撞到了也不算分。

```
①turn-cap     → 环境配置层 → 改 config / 删任务
②COMMUNICATE  → reward设计层 → 改 reward 口径
③策略不会做   → 模型能力层 → GRPO / SFT 本职工作
```

---

## 病因①:turn-cap 数学不可解(环境配置层)

**现象**:39/42/44 全 0,且大部分轨迹根本没执行到写动作就被切断。

**因果链(均经读代码确认)**:

1. **policy 硬约束**:airline `policy.md` 明文 *"You should only make one tool call at a time"*。
2. **verl 实现约束**:`max_parallel_calls=1`(verl 默认,本项目未覆盖)。模型即使一轮里塞多个 `<tool_call>`(task 39 实测单轮塞过 22 个),`fcalls[:1]`(`agent_loop.py:147`)**只执行第一个,其余全丢**。
3. 两者叠加 → **每个动作(哪怕只读 get/search)都独占一个 assistant turn**,只读动作无法并行省轮次。
4. `max_assistant_turns`(诊断时为 12)→ 最多 12 个动作。但:

   | task | gold 动作数 | 乐观所需轮次 | 说明 |
   |---|---|---|---|
   | **44** | **19**(16 读 + 3 写) | ~21 | 枚举 6 预订 + 10 次航班搜索 + 3 次改签;光读数据就吃满 16 轮,写动作排在第 17/18/19 步,永远轮不到 |
   | **39** | 11 | ~13 | 取用户详情 + 逐查 7 预订 + 取消 3 个 |
   | **42** | 10 | 恰好 12,零余量 | 枚举 7 预订 + 推断该取消哪 2 个 |

5. **DB reward 是全库 hash 严格匹配**(`agent_db_hash == predicted_agent_db_hash`,二值 0/1)。写动作一个没执行 → DB 状态 = 初始态 ≠ gold 终态 → **DB 分恒 0** → 总 reward 恒 0。

**证据**:
- task 42:96% 的轨迹顶到第 12 轮,尾部停在第 12 个 `get_reservation_details` 后、`assistant` header 刚生成就被 break——**0 个 cancel 执行**。
- task 44:56 条轨迹里只有 1 条执行过哪怕 1 次写动作。

**与病因③的根本区别**:换一个完美模型、一步不浪费,12 轮也只能走到 gold 的第 12 步,而要改的 DB 在第 17 步以后——「**12 个格子装不下 19 个必须串行的动作**」。这是配置缺陷,跟模型能力无关。

---

## 病因②:COMMUNICATE 维度锁死(reward 设计层,最隐蔽)

**现象**:task 7 有 **9/56 条轨迹已经把 DB 操作做到逐字正确**,但 reward 仍是 0。

**因果链**:

1. **reward 是各分量的乘积**(不是加权和)。task 7 = `DB × COMMUNICATE`。
2. **COMMUNICATE 是子串匹配**:对 `communicate_info` 里**每个**字符串,必须有某条 assistant 文本(去逗号 + lower 后)包含它;**全部命中**才得 1 分,否则 0。
3. task 7 要求模型主动报出「用户其他即将到来航班的总价 = 1628」。这个数字不是工具的直接返回值,要靠模型**主动枚举全部预订、逐个把 `price` 字段相加、再用自然语言说出来**(「算 + 说」)。
4. 模型从不做这一步,典型原因:(a) 没把「报总价」当成必须完成的动作;(b) 甚至**误判工具拿不到**(把 `get_user_details` 返回的预订号当成「没有明细」),据此 `transfer`;(c) 注意力全在 DB 操作上。
5. **乘积结构的致命性**:只要 COMMUNICATE=0,DB 做到 100% 也被乘成 0。**它把「DB 已学对」的好轨迹和「完全做错」的烂轨迹判成同一个 0 分**,组内零方差、零梯度——**惩罚了正确的探索**。

**波及面(已验证,修正了之前的高估)**:

写脚本对全部 40 个 train 任务复刻 COMMUNICATE 子串匹配规则,自检通过(task 7 应全 miss),结果:

- 40 个 train 任务**全部** `reward_basis = [DB, COMMUNICATE]`;
- 但**只有 5 个任务的 `communicate_info` 非空**(要求报具体数字),其余 35 个 COMMUNICATE 维度恒满足、不构成锁;
- 真正被锁死(命中率 0%)的是 **4 个**:

  | task | 要求说出的数字 | 命中率 | |
  |---|---|---|---|
  | **7** | `1628`(其他航班总价) | **0%** | 🔒 锁死 |
  | **14** | `327; 1000; 1786`(余额明细) | **0%** | 🔒 锁死 |
  | **18** | `23553`(总额) | **0%** | 🔒 锁死 |
  | **23** | `327; 1000; 1286`(余额明细) | **0%** | 🔒 锁死 |
  | 11(对照) | `5244` | **54%** | ✅ informative,可学 |

> **修正**:诊断早期曾「强烈怀疑 12/44 也是 COMMUNICATE 锁死」——**已证伪**,它们 `communicate_info` 其实是空的,属于纯 DB/turn 问题。**task 11 是关键对照**:同样要报数字,但模型有时会报(54%),所以能学——证明「报数字」本身可学,7/14/18/23 是从没触发过那一步。

**证据**:task 7 的 56 条里,9 条的 `update_reservation_flights(...cabin=business...credit_card_2408938)` + 两次 cancel 与 gold **参数逐字相同**,但 **0/56 说出过 "1628"** → 全 0。

---

## 病因③:纯策略不会做(模型能力 / 探索层,GRPO 本该解决的)

**现象**:task 7 中 43/56 走 `transfer_to_human_agents` 逃逸(首次 transfer 在第 1–13 轮高度分散)。涉及 **12, 15, 24, 29, 32, 35, 37**(task 7 同时中 ②③)。

两个具体策略缺陷:

**(a) `transfer_to_human_agents` 逃逸 = 局部最优陷阱**
- transfer 是一个「任何时候调用都能干净结束 episode、且看起来合理」的安全终止动作。
- RL-zero 冷启动时模型还没发现正确的多步解法,在「不确定 / 政策受限」局面下,transfer 是它能找到的风险最低出路 → 被反复采样自我强化 → 学会用 transfer 代替「把决定权交还用户」。
- task 7 里只有 28/56 会先把「basic economy 不能取消」告诉用户,其余直接 transfer,**连让 user simulator 提出「那先升舱」这个关键分支的机会都不给**。

**(b) 政策暗坑没掌握**
- tau2 核心难点:**「basic economy 不能直接改签 / 取消」,正确补救是「先升舱到 business/economy 再操作」或「取消重订」**。
- 模型**理解了前半句**(think 里能正确推出「basic economy 不满足取消白名单」),但**没学会后半句的补救链**,把「不能直接做」当成终局结论 → transfer。

**为什么 GRPO 很难自己爬出来**:正确解是一条很长的特定动作链(升舱→确认→取消→取消→报数字),base 模型随机采样几乎撞不全。task 7 的 9/56 已算「接近撞到」,但同时中了病因②,这点宝贵的接近又被锁死成 0,换不成梯度。

> 这一类(12/15/24/29/32/35/37,gold 动作数 < 12、turn 够得着)**是 GRPO / SFT 冷启动真正该优化的**,但前提是先有一条非 0 轨迹给出梯度。

---

## 3. 关键代码 / 配置定位

| 机制 | 位置 | 说明 |
|---|---|---|
| 一轮只执行 N 个 tool call | `agent_loop.py:147` `fcalls[: self.max_parallel_calls]` | `max_parallel_calls` 默认 1(`agent_loop.py:76`),来自 `rollout.multi_turn`,是 **verl** 配置项,非 tau2 |
| response token 长度闸 | `agent_loop.py:142` `if len(response_mask) >= self.response_length` | 比 turn 闸更早触发;`response_length = max_response_length` |
| tau2 多 tool-call 执行 | `tau2_session.py` `execute_tools` 的 `for tool_call in ...` 循环 | tau2 执行层**支持**顺序执行多个(DB 同步就地 mutate),不用改 submodule |
| reward 拆分 | `env/reward.py` `compute_reward` | 包装 `evaluate_simulation(...ALL)`;最终 reward = `reward_basis` 各分量乘积;子分(DB/COMMUNICATE/ACTION/NL/ENV)被暴露但**训练时只把标量写进 jsonl 的 `score`,子分没存 → 要拿子分得重放** |

---

## 4. 已做的改动 与 待决策

### 已改
- **`scripts/train/run_grpo.sh`:`max_assistant_turns` 默认 12 → 24**(`MAX_ASSISTANT_TURNS`)。`max_user_turns` 维持 12(长任务 user 轮只有 4–6 个,不是瓶颈)。

### ⚠️ 重要:光改 turn=24 还不够生效
`max_response_length` 仍是 **12288**、`ppo_max_token_len_per_gpu` 仍是 **20480**,**都没跟着抬**。实测 task 44 跑满 12 轮时已 ≈ 36700 字符 ≈ 11k token,逼近 12288;跑到 24 轮需 ≈ 22k token,**远超 12288**。结果:长任务会在第 ~13–14 轮撞 `response_length` 闸(`agent_loop.py:142`)被切断,**到不了第 24 轮**——只是从「被 turn 切」换成「被 length 切」,DB 照样没改完 → 照样 0。

要让 turn=24 **真正生效**,需同步抬(注释里写了 `ppo_max_token_len_per_gpu` 必须 ≥ prompt+response):

| 参数 | 现值 | 建议 | 后果 |
|---|---|---|---|
| `max_response_length` | 12288 | ~22000 | 让 24 轮装得下 |
| `ppo_max_token_len_per_gpu` | 20480 | ~28672(=6144+22000+余量) | 否则打包序列放不下 |
| | | | **显存压力明显增大,2×A100 80G 有 OOM 风险**(必要时降 `gpu_memory_utilization`,现 0.5) |

### 关于 `max_parallel_calls`(更便宜但是 hack)
- 把它调大,允许一轮顺序执行多个只读 tool call → task 44 的 16 个读塌缩成 2–3 轮,19 动作能压进预算,**不拉长 token 序列、不增显存**。代码路径现在就跑得通(只在 `run_grpo.sh` 加 `actor_rollout_ref.rollout.multi_turn.max_parallel_calls=N`),不用改 tau2 / agent loop。
- **但它是 hack**:(1) 违反 policy「一次一个 tool call」;(2) 与官方 eval 路径(tau2 `run_domain` runner,单 tool-call)不一致,造成 train/inference gap、打乱 base vs trained 对比口径;(3) 对这 4/3 个 `reward_basis` 只看 DB+COMMUNICATE 的任务无害,但对带 ACTION/NL_ASSERTION 的任务可能被判违规。

### 处置优先级建议
1. **3 个 turn 不可解任务(39/42/44):优先从 train split 删除/更换** > 抬全局 turn。只占 3/40,删了不心疼;抬 turn 要让另外 37 个任务陪绑、长任务空转更久、还冒 OOM。注:全 50 任务 `annotations` 均为 null(无官方难度标注),删任务只能靠这种推算。
2. 若坚持训练这类超长任务:优先试 `max_parallel_calls`(便宜)而非 `max_assistant_turns`(贵 + 连锁 OOM)。
3. **真正该花力气的是病因②③** —— 抬 turn 最多动 3/14 个全 0 任务,且其中 44 解锁后可能仍 0。

---

## 5. 一个待评估的方向:SFT 冷启动

病因③(7 个任务)是「模型真不会」,RL-zero 在这种长链硬任务上冷启动困难(撞不出第一条非 0 轨迹就没梯度)。一个候选:用现成的 tau2 airline SFT 数据先冷启动。

- 候选数据集:`inclusionAI/AReaL-tau2-data`(HF)。`tau2_sft_train.jsonl` 874MB / 33531 条,其中 **airline 12842 条**;三 domain 混在一起,需按 domain 过滤。已下载到本地 `local_datasets/`(经 `--noproxy` 绕过本机代理,见下)。
- **尚未确认能否直接用**:还需核对它的 tool-call 格式、以及它的 policy 与本项目 `policy.md` 是否一致(初步抽取出现可疑差异:AReaL policy 只抽到 438 字符 vs 我们的 7676,未确认是真用精简 policy 还是抽取截断)。
- **这是一个方向性决定**:项目当前定位写死为 **RL-zero、明确「无 SFT」**(CLAUDE.md / plan)。引入 SFT 冷启动不是坏事(诊断确实显示 RL-zero 冷启动困难),但要有意识地做这个 trade-off。

> 环境提示:本机所有请求默认走代理 `127.0.0.1:17897`,`hf-mirror.com` 不在 `NO_PROXY` 白名单 → 会被代理拦截超时。下 HF 镜像须 `--noproxy`/`NO_PROXY` 直连。

---

## 6. 最高优先级待办

1. ~~验证病因②波及面(拆 DB/COMMUNICATE 子分)~~ ✅ 已完成:确认仅 4 个(7/14/18/23)。
2. **决定 reward 口径**:病因② 是 reward 把可学信号抹平了。是否在训练期按 `reward_basis` 分量分别给信号(而非只用乘积标量),让「DB 做对」也能拿到部分 reward / 产生组内方差?——这会直接改变 7/14/18/23 能否学动。
3. **决定 39/42/44 去留**:删/换,还是抬 `max_response_length`+`ppo_max_token`(赌显存)真正放开 turn=24。
4. **决定是否引入 SFT 冷启动**:先核对 AReaL 数据格式/policy 兼容性,再定是否打破 RL-zero 定位。
