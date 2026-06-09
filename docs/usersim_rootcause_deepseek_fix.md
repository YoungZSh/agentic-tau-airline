# GRPO 曲线持平排查:user-sim 是元凶,DeepSeek V4 救回信号

> 排查时间:2026-06-07
> 起点:`outputs/qwen3_8b_lora_grpo_tau2_airline_20260606_015313` 训练 80 步、reward 持平
> 结论落地:`.env` 切 `openai/deepseek-v4-flash`,从 SFT ckpt 重启(`outputs/..._20260607_191951`)
> 数据/产物:`outputs/usersim_ab_20260607/`(4 个 user-sim 的逐 sim 结果 + `ab_eval.py`)、
> 完整对话 `third_party/tau2-bench/data/simulations/20260607_14*`

**核心结论(一句话):GRPO 曲线持平不是模型学不动,而是 06-06 把 user-sim 从 gpt-5 换成本地
`Qwen3.6-35B-A3B` 后,这个 3B-active 的小模型守不住"顾客"角色、78% 的对话翻转成客服并陷入
逐字复读死循环,把 reward 信号整体压掉了约 90%。换成 DeepSeek V4(flash)后,角色翻转降到 1%、
信号恢复到 gpt-5 的 ~81%,成本可忽略(整训 ~46 元)。**

> **后续(2026-06-08):** 修复 user-sim 后从 SFT 重训(`..._20260607_191951`),**曲线仍平** ——
> 这暴露了**第二根因:策略冻在 SFT 没更新**(KL 锚 / clip 0.2 / LoRA 步子太小)。
> 含 AReaL 官方 airline 配方对照,见 [`grpo_flat_curve_and_areal_recipe.md`](grpo_flat_curve_and_areal_recipe.md)。

---

## 0. TL;DR — 因果链

```
本地 A3B user-sim 守不住"你是顾客"角色(被 agent 开场白带成客服 persona)
  → 78% 对话角色翻转、53% 逐字复读死循环 → 86% 顶到 max_steps 才结束
  → 顾客从不给 user_id / 不推进任务 → agent 做不了 DB 写 → db ≈ 0
  → reward 信号被压掉 ~90% → informative group 变少 + 天花板被砸低
  → GRPO 曲线持平、grad_norm ~0.02
```

修复 = 换一个守得住角色的 user-sim(DeepSeek V4 flash),其余训练栈不动。

---

## 1. 现象:训练 reward 是一条平线

最新 run(step 11→80,约 23h)的训练 reward(`critic/score/mean`):

| 区间 | 均值 | 斜率 |
|---|---|---|
| 前半(step 11–45) | 0.482 | — |
| 后半(step 46–80) | 0.481 | **≈ -0.00003/步**,70 步累计 -0.002 |

验证 reward@1(10 任务单采样)在 0.5–0.6 噪声区间徘徊;`grad_norm` 恒在 ~0.02、`pg_loss ~1e-9`。
entropy / response_length 健康稳定 → **不是训练崩了,是"不动"**。

---

## 2. 第一层排查:梯度饥饿(group informativeness)

GRPO 梯度只来自**组内 reward 有方差**的 group。统计 80 步全部 rollout(每步 64 条,按 `gts` 分组):

- `train_batch_size=8` → 每步仅 **8 个 group**;
- score-key informative 占比 **均值 50.9%** → 平均每步只有 **~4/8 组**产生梯度;
- **9/40 任务(22.5%)近乎永久全败**(mean_db < 0.05):`14,18,23,24,29,33,39,42,44`;**0 个**永久全过;
- 死任务不是被 12288 截断:读 task 14 轨迹,模型**干净地输出"转人工"**(行为错,非长度问题);
- `comm` 双峰饱和(77% == 1.0),GDPO 等权 `[db,comm,db_comm]` 会在 db 无望的死任务上抽 comm 梯度,稀释信号。

当时的(错误)结论倾向是"剔死任务 / 调 batch·LR·KL"。**但下一步推翻了这个前提。**

---

## 3. 转折:SFT 救活的任务,在当前 run 里又死了

`docs/grpo_zero_reward_diagnosis.md §5.1` 记录:SFT 冷启动后,原 14 个全 0 任务里 **7 个复活**
(`12,15,18,24,29,39,42`)。但当前 run 测到的死任务是 `14,18,23,24,29,33,39,42,44`,两者一对:

- **SFT 救活、现在又死的:`18,24,29,39,42`(5 个回归)**;
- **以前不在死名单、现在新死的:`33`**。

判据:这些任务的 db **从 step 1 就是 0.00(出生即死 DOA)**,不是"先活后衰" → **排除"GRPO 把 SFT 练废"**,
只能是 rollout 条件变了。

---

## 4. 定位:user-sim 被换了

| 事实 | 值 |
|---|---|
| GRPO 起点 | ✅ SFT 合并权重(`.../qwen3_8b_full_sft_tau2_airline_20260605_152720/hf_merged`) |
| SFT 复活那次评估(`sft_rollout_eval.log`,06-05 18:22) | user-sim = **gpt-5** |
| 当前 GRPO run(06-06 14:16 起) | user-sim = **本地 `qwen3.6-usersim`**(Qwen3.6-35B-A3B,:8011) |
| `EXCLUDED` 计数 | 仅 2 次 → 本地 user-sim **不是崩溃**,是对话动力学变了 |
| judge 影响 | 几乎无关:train 任务 `reward_basis=[DB,COMMUNICATE]`,db 是确定性 hash、comm 是子串匹配,都不依赖 NL judge |

**唯一变量 = user-sim。** db 与 judge 无关,所以 db=0 只能是本地 user-sim 让对话走不到正确 DB 终态。

---

## 5. 受控 A/B:量化 user-sim 的影响

设置:**纯 SFT 模型(无 LoRA)· tau2 `run_domain` 路径 · 8 任务 × 8 trial · 只换 user-sim**。
任务集 = 回归 `18,24,29,39,42` + 负控 `44` + 正控 `4,13`。

| task | 角色 | gpt-5 db | 本地 A3B db |
|---|---|---|---|
| 4 | 正控 | **1.00** | 0.12 |
| 13 | 正控 | 0.75 | 0.25 |
| 18 | 回归 | 0.62 | **0.00** |
| 24 | 回归 | 0.38 | **0.00** |
| 29 | 回归 | 0.12 | **0.00** |
| 39 | 回归 | 0.25 | **0.00** |
| 42 | 回归 | 0.88 | **0.00** |
| 44 | 负控 | 0.00 | 0.00 |
| **平均** | | **0.50** | **0.05** |

**本地 user-sim 只保留约 9–11% 的 reward 信号。** 5/5 回归任务清零;连 gpt-5 下满分(1.00)的 task 4
都被打到 0.12 → 不是只杀难任务,是**全面扣血**;负控 44 两边都 0(对照干净)。

---

## 6. 机制:角色翻转 + 镜像死循环

读 task 4 完整对话(gpt-5 成功 vs 本地失败):

- **gpt-5**:`"My name is Sophia Silva, and my user ID is sophia_silva_7557..."` → 提供信息、推进任务 → reward 1.0、`user_stop`。
- **本地 A3B**:第 2 轮就翻转成客服 —— `"I'd be happy to help you... could you please provide your user ID?"` →
  真 agent 困惑 → user-sim `"I apologize for the confusion!... provide your user ID?"` **逐字复读到 max_steps** → reward 0。

全量统计(各 64 条对话):

| 指标 | gpt-5 | 本地 A3B |
|---|---|---|
| 干净结束 `user_stop` | 84% | 14% |
| `max_steps` 卡死 | 14% | **86%** |
| user 说 agent 话术(角色翻转) | 4% | **78%** |
| user 逐字复读 | 0% | **53%** |

根因:A3B(仅激活 ~3B)守不住角色,被 agent 开场白"How can I help you"带成客服 persona。
现有加固(`rollout_hardening_usersim.md §3`)只拦**首轮**的空/控制 token,这个翻转发生在**第 2 轮后且是流畅句子**,
所有 guard 都不触发;加"中途镜像检测"也救不回 reward(顾客仍没给信息)。**只能换更强模型或回 gpt-5。**

---

## 7. 解法:DeepSeek V4 当 user-sim

同一 A/B 工装,加测 `deepseek-v4-pro` / `deepseek-v4-flash`(API,thinking 默认开,content 与 reasoning_content
API 层分离、无 `<think>` 泄漏):

| user-sim | 平均 db | 角色翻转 | max_steps 卡死 | 干净结束 |
|---|---|---|---|---|
| gpt-5(上限) | 0.50 | 4% | 14% | 84% |
| 本地 A3B(现状) | 0.05 | 78% | 85% | 14% |
| **deepseek-v4-pro** | **0.44(88%)** | 9% | 7% | 92% |
| **deepseek-v4-flash** | **0.41(81%)** | **1%** | 10% | 87% |

**推荐 flash**:翻转率最低(1%,比 gpt-5 还好)、最便宜;pro 聚合 db 略高但贵 3 倍、翻转率反而 9%(N=8 噪声)。
5 个回归任务救活 4 个(只剩 29,它在 gpt-5 下也才 0.12),正控完全追平 gpt-5。

---

## 8. 成本(flash)

A/B 实测每对话 ~14K token,训练规模 ~6,500 对话(100 步 × 64),按 flash 0.5 元/百万 token:

| 训练量 | 估算 |
|---|---|
| 跑满 20 epoch | **≈ 46 元(~$6.4)** |
| 10 epoch | ≈ 23 元 |
| 5 epoch | ≈ 11 元 |

偏高估计(实际更低):输入 67% 缓存命中、A/B 任务偏难对话偏长。**结论:整训几十块,可忽略;flash vs pro 按质量选不按成本选。**

---

## 9. thinking 控制的发现(实测)

| 参数(发给 deepseek-v4-flash) | reasoning_tokens | 效果 |
|---|---|---|
| 无参数(默认) | 75 | thinking **开** |
| Qwen 式 `chat_template_kwargs.enable_thinking=False` | 75 | **被忽略,无效** |
| 顶层 `enable_thinking=False` | 75 | **被忽略,无效** |
| `thinking={"type":"disabled"}` | None | ✅ **唯一能关的方式** |
| `reasoning_effort` | — | 取值 `low/medium/high/max/xhigh`(无 none/minimal) |

**`enable_thinking` 是 Qwen/vLLM 概念,DeepSeek 不认、静默忽略。** 故代码里"qwen 才发 extra_body"的门控对 DeepSeek 双重失效。

---

## 10. 落地改动

**`.env`(唯一功能性改动)**:`OPENAI_BASE_URL=https://api.deepseek.com`、`OPENAI_API_KEY`=DeepSeek key、
`TAU2_USER_MODEL`/`TAU2_NL_JUDGE_MODEL=openai/deepseek-v4-flash`;删掉已死的 `TAU2_DISABLE_THINKING`/`TAU2_USER_DISABLE_THINKING`。

**代码清理**(qwen 专属 enable_thinking 已无用):
- `env/airline_interaction.py`:删 `if "qwen" in model` 的 `chat_template_kwargs.enable_thinking` 注入块,`_user_llm_args_from_env` 去掉无用 `model` 参数;
- `env/reward.py`:`set_nl_judge_model` 删 `TAU2_DISABLE_THINKING → enable_thinking=False` 注入。
- 验证:ruff 通过、`py_compile` 通过、全量离线测试 22 passed。

**重启**(从 SFT 干净起步,弃掉 80 步坏 LoRA):
```bash
# 停旧 run(-9 避免包装脚本 EXIT trap 把 GPU 重新 hold 住)
pkill -9 -f run_grpo_then_hold.sh; pkill -9 -f "verl.trainer.main_ppo"
conda run -n tau2verl ray stop --force
# 新 run(GPU_LIST 自动设 CUDA_VISIBLE_DEVICES;不带 RESUME_FROM = fresh from SFT)
GPU_LIST="3 4" bash scripts/train/run_grpo_then_hold.sh
```

---

## 11. 重启后健康检查(`outputs/..._20260607_191951`)

前 3 步:

| step | db mean | db std | comm mean | 单步耗时 |
|---|---|---|---|---|
| 1 | 0.55 | 0.50 | 0.69 | 669s |
| 2 | 0.59 | 0.49 | 0.84 | 612s |
| 3 | 0.42 | 0.49 | 0.73 | 587s |

**db std ~0.49 → 组内方差大 → informative group 多 → 有梯度可学**(信号活过来了);`aborted_ratio=0`、`EXCLUDED=0`。

时长:**~10 分钟/步 × 100 步 + 验证 ≈ 18 小时**(约 6/8 下午跑完 20 epoch;可按 val 曲线见顶早停)。
瓶颈是 rollout(~350s/步,DeepSeek thinking + 网络往返)。

**日志里那条刷屏 ERROR `This model isn't mapped yet. model=deepseek-v4-flash` 是 litellm 算钱时查不到 V4 价格,
纯 cost-tracking 噪声,LLM 调用本身成功,与训练无关。** 可选:`litellm.register_model({...})` 注册价格以消除刷屏并拿到真实 $ 成本。

---

## 12. 经验教训

1. **per-step reward 均值不能用来判断学没学**:每步只抽 8/40 任务、难度差异大,要追同一任务跨 epoch 的 reward,或看 group informativeness。
2. **评测环境(user-sim/judge)是 reward 信号的一部分**:换 user-sim 等于换了 reward 分布。迁移前必须做受控 A/B 对比,而不是直接上训练。
3. **"死任务"先分清病因**:环境配置(turn cap)/ reward 口径(乘积锁死)/ 模型能力 / **评测环境**——表象都是"组内全 0",处置完全不同。
4. **小 active-param 的 MoE 不适合做 user-sim**:角色扮演是长程指令遵循,3B-active 守不住角色;DeepSeek V4 / gpt-5 这类强模型才行。
5. **`tau2` 默认把完整 simulation 落盘**到 `third_party/tau2-bench/data/simulations/<timestamp>_.../results.json`,排查对话动力学时直接读它,不用重跑。
