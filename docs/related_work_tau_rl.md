# 相关工作:τ-bench / 多轮工具调用 Agent 的 RL 微调

> 用途:收录与本项目(τ²-bench Airline × verl × LoRA GRPO,RL-zero)同类的论文与开源代码,
> 便于对照 reward 设计、训练动态、用户模拟器接法等关键决策。
> 维护:发现新工作时按下面的分区追加,并在末尾"对本项目的启示"里更新一句话结论。

本项目定位回顾(对照基准):
- **base**:Qwen3-8B-Instruct,**RL-zero(无 SFT)**,LoRA r32/α64,2×A100。
- **reward**:τ² 官方 `evaluate_simulation`(DB hash + NL assertion,分量乘积),**非自定义规则**。
- **user simulator**:外部 API 模型(gpt-5)在线模拟用户,**接进 rollout 回路**。
- **rollout**:verl AgentLoop,`mode=async`(并发 rollout 隐延迟,**完全 on-policy**,无 staleness)。

---

## A. 论文(直接做 τ-bench / 多轮 agentic RL,优先看)

### A1. Multi-Turn RL for Tool-Calling Agents with Iterative Reward Calibration (IRC)
- 链接:https://arxiv.org/pdf/2604.02869
- 一句话:自称**首个公开的 τ-bench airline RL 训练结果**。MT-GRPO + GTPO,Qwen3.5-4B 63.8→66.7、
  Qwen3-30B-A3B 58.0→69.5;训练后的 4B 超过 GPT-4.1(49.4)/GPT-4o(42.8)。
- **核心教训(与我们最相关)**:朴素设计的 **per-turn dense reward 会让性能掉最多 14 个点**——
  reward 的"区分度"和 advantage 方向不一致时会反噬。⇒ 支持我们用官方 outcome 口径、不手搓中间整形奖励。

### A2. MUA-RL:Multi-turn User-interacting Agent Reinforcement Learning
- 链接:https://arxiv.org/abs/2508.18669 ｜ OpenReview:https://openreview.net/forum?id=BtU29vylui
- 一句话:**首次把 LLM 用户模拟器放进 RL 训练回路**(与我们 gpt-5 user-sim 同构)。
  MUA-RL-32B:TAU2 Retail 67.3 / Airline 45.4 / Telecom 28.3,非思考设定下匹敌 DeepSeek-V3、Qwen3-235B。
- **核心立场**:**只用最终任务完成度做 reward,不奖励中间步骤/格式**,理由是中间整形易被 hack。
  ⇒ 与我们的 reward 哲学一致,可作为"outcome-only 是对的"的主要外部背书。

### A3. EigenData:From Self-Evolving Synthetic Data to Verifiable-Reward RL
- 链接:https://arxiv.org/abs/2601.22607
- 一句话:层级式多智能体框架 **EigenData**,把**自进化合成数据**(工具落地的对话 + 每实例
  可执行 checker)与**可验证奖励 RL** 串起来,闭环 self-evolving 提升生成可靠性;RL 用
  GRPO-style **trajectory-level group-relative advantage + dynamic filtering**。
  **τ²-bench Airline 73.0% / Telecom 98.3% pass@1**(目前 airline 上很高的公开数字)。
- **核心看点(与我们最相关)**:
  - **每实例可执行 checker 当可验证奖励**——比我们用官方 `evaluate_simulation` 更细粒度,
    但同属 outcome/verifiable 口径、不手搓 per-turn 整形奖励(与 A1/A2 立场一致)。
  - **专门处理 user-simulator 噪声**:对 user model 做微调以压低模拟用户带来的 noisy signal——
    直接对应我们 gpt-5 在线 user-sim 的可靠性/噪声问题。
  - **SFT cold-start + GRPO** 的 pipeline,与我们现在的 **SFT→GDPO** 路线同构,可对照数据合成接法。

### A4. 其他 reward / credit 设计变体
- RC-GRPO(Reward-Conditioned GRPO for multi-turn tool calling):https://arxiv.org/pdf/2602.03025
- Reinforcing Multi-Turn Reasoning via Turn-Level Reward Design:https://arxiv.org/pdf/2505.11821
- CM2:RL with Checklist Rewards for Multi-Turn & Multi-Step Agentic Tool Use:https://arxiv.org/pdf/2602.12268

### A5. 综述 / 实操指南
- A Practitioner's Guide to Multi-turn Agentic Reinforcement Learning:https://arxiv.org/html/2510.01132v1
- SkyRL-Agent:Efficient RL Training for Multi-turn LLM Agent:https://arxiv.org/abs/2511.16108

---

## B. 开源代码库(可直接看 / 借鉴)

### B1. OpenPipe / ART(Agent Reinforcement Trainer)
- 链接:https://github.com/OpenPipe/ART
- 多步 agent 的 GRPO 训练器,封装友好、不强绑 verl,支持 Qwen3 等;有 τ-bench 案例。
  解读:https://machinelearningatscale.substack.com/p/openpipe-rl-for-multi-turn-agents
- 看点:它的 AgentLoop / 轨迹构建抽象,可与我们的 `Tau2AirlineAgentLoop` 对照。

### B2. Danau5tin / terminal-bench-rl
- 链接:https://github.com/Danau5tin/terminal-bench-rl
- 长程 GRPO(terminal/coding 任务),scales 到 32×H100。非 τ-bench,但工程结构高度同构,适合抄架构。

### B3. SkyRL-Agent(UC Berkeley Sky Computing Lab)
- 项目页:https://sky.cs.berkeley.edu/project/skyrl/ ｜ 论文:https://arxiv.org/abs/2511.16108
- 多轮长程 agent RL 框架,异步 pipeline dispatcher(~1.55× 加速),可对接 verl/Tinker。
  ⇒ 我们未来若要做**真异步 rollout(one_step_off / fully_async)**时的主要参考。

### B4. GeneralReasoning / tau2bench(OpenReward)
- 链接:https://openreward.ai/GeneralReasoning/tau2bench
- 把 tau2bench 包成现成的 RL environment。

### B5. amazon-agi / tau2-bench-verified
- 链接:https://github.com/amazon-agi/tau2-bench-verified
- 修正版 τ²:修了原版 task 定义 / 期望动作 / 评测标准与 policy/DB 不对齐的 bug。
- **强烈建议做评测可信度时换用此数据**,规避原版的标注/泄漏问题。

### B6. 通用底座框架
- verl(本项目所用):https://github.com/verl-project/verl
- 其他:rLLM、verl-tool、RAGEN(agentic RL 通用框架)。

### B7. 已分析过的对照项目
- qiqihezh / agentic-grpo-longhorizon:https://github.com/qiqihezh/agentic-grpo-longhorizon
- 同域(τ² airline × verl × GRPO),但走 **SFT→GRPO + 手搓 PRM-Lite/LATA 整形 reward**。
  其 "+37%" 基于自定义整形奖励且自承认有 16/40 训练任务泄漏,数字需打折看。
  对照价值:训练动态(小批 batch=4、跑 300–500 步)值得借鉴;reward 设计是反面教材(见 A1/A2)。

### B8. 基准本体
- τ-bench:https://github.com/sierra-research/tau-bench
- τ²-bench:https://github.com/sierra-research/tau2-bench

---

## 对本项目的启示(持续更新)

1. **reward 哲学站得住**:A1/A2 都支持"用最终 outcome 口径、不手搓 per-turn 整形奖励"——
   这正是我们用 τ² 官方 `evaluate_simulation` 的依据;对照项目的 PRM-Lite 是反例。
2. **user-sim 接进回路是主流且正确**(A2/MUA-RL),我们的 gpt-5 在线 user-sim 与之同构。
3. **训练动态是当前主要短板**:同类工作普遍跑数百步;我们当前全量 batch=40 只有 ~20 步。
   已定方向:**调小 `TRAIN_BATCH_SIZE`(40→8,免费加更新数)+ 抬 `TOTAL_EPOCHS` 把总步数顶到 ~150–300**。
4. **评测可信度**:考虑切到 B5(tau2-bench-verified)以规避标注/泄漏问题。
5. **异步深化**:若日后要 off-policy 真异步,优先参考 B3(SkyRL-Agent)与 verl 的 one_step_off/fully_async recipe。
6. **user-sim 噪声是真问题**:A3(EigenData)专门微调 user model 来压噪声,且其 SFT→GRPO + 可验证
   checker 路线与我们的 SFT→GDPO 同构、airline 数字(73.0%)很高——是当前最值得贴身对照的工作。
