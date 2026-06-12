# fully-async trainer OOM 排查实录:逐 micro-batch 的计算图滞留泄漏

> 适用场景:disaggregated fully-async(1 train + 2 rollout GPU),Qwen3-8B + LoRA,
> 多轮 tool-calling,16k token 动态 micro-batch
> 时间:2026-06-11,一天内 6 次失败 run → 根因 → 修复 → 上游 issue/PR
> 结论:**不是显存不够,是 verl unified engine 每个训练 micro-batch 泄漏 ~0.27 GiB**
> 修复:`src/tau2_airline_verl/patches/sitecustomize.py` 的 `VERL_TRAIN_METRICS_DETACH=1`
> (run_grpo_fully_async.sh 默认开);上游 [issue #6698](https://github.com/verl-project/verl/issues/6698)
> / [PR #6699](https://github.com/verl-project/verl/pull/6699) **已于 2026-06-12 合并**(dbbf0853)

---

## 1. 症状

`run_grpo_fully_async.sh` 的每一次 run 都死在**第一次 `update_actor` 的 backward**:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 5.04 GiB.
GPU 0 ... 78.85 GiB memory in use. Of the allocated memory 61.49 GiB is
allocated by PyTorch, and 10.07 GiB is reserved by PyTorch but unallocated.
```

rollout、step 0 验证、param_sync 全部正常;唯独 trainer 卡(80GB A800)在攒满
40 prompt × n=8 = 320 条轨迹、进入第一次更新后必爆。

## 2. 排查时间线:三层"对症但不治本"的修复

每一层都修掉了 OOM 报文里指认的问题,**每一层都真实生效,但峰值每次都涨回 ~70 GiB**,
只是死点越来越深——这是事后看最重要的线索,当时被当作"还差一点"连续误读了三次:

| run | 修复 | 验证生效的证据 | 死点 | 失败分配 | PyTorch 已占 |
|---|---|---|---|---|---|
| 15:28 | expandable_segments(碎片) | reserved-unallocated 10.07→0.3 GiB | lm_head logits 物化 | 5.15 GiB | 66.5 GiB |
| 16:10 | + fused linear-CE(USE_FUSED_KERNELS) | 死点穿过 lm_head 进入 MLP gc-recompute | MLP 重算 | 434 MiB | 71.3 GiB |
| 17:03 | + model_dtype=bf16(fp32 master→bf16) | After-FSDP 30.84→15.58 GB,param_sync 44s→23s | fused-CE backward | 1.16 GiB | 70.4 GiB |

三层修复各自的固定收益(碎片 ~10 GiB、logits 物化 ~5-10 GiB、fp32 master ~15 GiB)
全部被"某个随 micro-batch 增长的东西"吃掉。静态读代码到此已无法推进。

> 实现注意:expandable_segments **不能**像 run_sft.sh 那样全局 export
> `PYTORCH_CUDA_ALLOC_CONF`——ray worker 继承 env,vLLM sleep-mode 的
> `CuMemAllocator.__init__`(cumem.py:150)硬 assert 拒绝它,rollout server 会启动即崩。
> 所以走 sitecustomize 在 `TrainingWorker.__init__`(只在 trainer 进程实例化)运行时调
> verl 自带的 `set_expandable_segments(True)`。fused kernels 与 bf16 master 同理由
> 脚本旋钮 `USE_FUSED_KERNELS` / `MODEL_DTYPE` 接入。

## 3. 转折:每 micro-batch 显存打印 + 内存快照

放弃猜测,上实证(都在 sitecustomize 的 `TRAIN_MEM_SNAPSHOT=1` 诊断 patch 里):

1. **每个 micro-batch 前打印** `torch.cuda.memory_allocated()`;
2. trainer 进程开 `torch.cuda.memory._record_memory_history(max_entries=1.5M)`;
3. OOM 时(以及 `VERL_MEM_SNAPSHOT_AT_MB=N` 指定的第 N 个 micro-batch)dump 快照。

显存曲线立刻把问题钉死(run 19:05,一次 update ≈ 400 个 micro-batch):

```
#1–#250   恒定 15.6 GiB   ← old_log_prob + prox 的 no_grad 重算前向,不漏
#250–#400 每个 +0.27 GiB  ← 一进入带 backward 的训练段就线性增长
#400      64 GiB → OOM
```

**无梯度不漏、有梯度每步漏固定量 = 计算图被滞留**,与"显存不够"彻底区分开。

快照分析(`scripts/debug/analyze_mem_snapshot.py`)的几个实战要点:

- 存活 block 的 `frames` 大多为空:**backward 线程(C++)的分配没有 Python 栈**,
  纯按栈聚合会失败;
- 解法一:按 `device_traces` 的 alloc 事件给存活 block 配栈,用**前后最近的带栈事件
  夹逼**定位无栈分配;
- 解法二:200k 默认事件窗口只够 ~1 个 micro-batch,泄漏块的 alloc 事件早被挤出环形
  缓冲——所以要么加大窗口,要么**在泄漏刚开始后不久(第 N 个 micro-batch)主动 dump**,
  让泄漏块的分配事件还在窗口内;
- 按 block 尺寸做整数分解(除以 hidden/intermediate/vocab × 2B/4B)可以在无栈时猜身份,
  但容易误导(本案中 36×368 MiB 至今没分解出整维度),夹逼定位才是决定性的。

最终定位:每个训练 micro-batch 永久滞留**一对同尺寸 [total_nnz, 4096] bf16 张量
(~2×139 MiB @16k token)**:

1. **embedding 输出**(C++ 栈:`at::native::embedding`,Python 邻位:`F.embedding`);
2. 它在 backward 末尾(`_fused_linear_for_ppo_bwd` 之后)产生的**梯度缓冲**。

## 4. 根因机制链

```
PEFT enable_input_require_grads()            # LoRA + gc 必需:让梯度能流进 adapter
  → embedding 输出 requires_grad
  → 非重入 gradient-checkpoint 的 frame 把它作为重算入口保存
       (这类引用不随 loss.backward() 释放,只随【图的销毁】释放)

verl FSDPEngineWithLMHead.forward_step:
  output = {"model_output": model_output,    # ← log_probs/entropy 没 detach,挂着 grad_fn!
            "loss": loss.detach().item(),    # ← loss 倒是 detach 了
            "metrics": metrics}              # ← ppo_loss 还把活的 pg_loss/ppo_kl 塞了进来
  → forward_backward_batch 把它 append 进 output_lst,
    直到整个 mini-batch(~400 个 micro-batch)跑完才统一后处理
  → 每个训练 micro-batch 的图被锚住 → checkpoint frame 的 embedding 输出 + 梯度缓冲
    全部释放不掉 → 0.27 GiB/micro-batch × ~150 ≈ 40 GiB/update
```

**为什么 colocated(run_grpo.sh / areal_probe)没爆**:同一份引擎代码(main_ppo 也走
`engine_workers`),但 DP=2 把泄漏摊到两张卡、bypass 模式少一遍 prox 重算、response cap
更短——~20 GiB/卡的泄漏刚好藏进 80 GB。**这个 bug 在 colocated 上一直潜伏**。

**为什么社区没踩到**:full-param 训练没有 `enable_input_require_grads`,checkpoint frame
不锚 embedding 输出,每 micro-batch 滞留的只有 KB 级小张量;LoRA + 16k 长序列 + 单卡
trainer + 一次 update 几百个 micro-batch,把它放大成必现 OOM。

## 5. 修复与验证

`sitecustomize.py` 的 `VERL_TRAIN_METRICS_DETACH=1`(脚本默认开)的有效成分只有一个:

1. **唯一的锚**:包 `FSDPEngineWithLMHead.forward_step`,返回前 detach `model_output`
   里所有带 grad_fn 的张量;
2. ~~metrics detach~~:后来写回归测试时发现 `Metric.append` 对张量**本来就会
   `detach().item()`**,metrics 从不锚图——这也解释了为什么当时"先 detach metrics"
   的对照 run 毫无效果(无图可锚),上游 PR 里对应的 commit 已 revert。

这些输出只在 backward 之后用于指标聚合,detach 零行为影响。验证(run 22:15):

| 训练段 micro-batch | #250 | #300 | #350 | #400 |
|---|---|---|---|---|
| 修复前 | 24.8 GiB | 37.9 GiB | ~50 GiB | 64 GiB → OOM |
| 修复后 | **16.2** | **16.2** | **16.2** | **16.2 GiB** ✅ |

900+ micro-batch 持续平坦,零 OOM,端到端训练正常推进。step:1 指标全面健康:
pg_loss 0.0217、grad_norm 0.015、训练/rollout log-prob Pearson 0.993、KL 0.002、
trainer 峰值 31.5/79 GB——同时证明 fused kernels + LoRA 的数值无恙。

### 5.1 回归测试原理(`tests/workers/test_engine_forward_step_detach_on_cpu.py`)

不测显存数字,测因果:**"持有 forward_step 的输出"不得再意味着"持有整张计算图"**。
做法是把生产泄漏链的每个环节换成最小等价物,用 `weakref` 直接观测"该死的张量死没死":

| 生产环节 | 测试等价物 |
|---|---|
| 冻结 8B base + LoRA | 冻结 `Embedding` + 可训练 `Linear` |
| PEFT `enable_input_require_grads()` | `x = embed(ids); x.requires_grad_(True)` |
| gradient-checkpoint 的 decoder 层 | `checkpoint(proj, x, use_reentrant=False)`(frame 保存 x 作重算入口) |
| forward_step 返回带 grad_fn 的 log_probs | stub `prepare_model_outputs` 返回 `hidden.sum(-1)` |
| output_lst 攒 400 个 micro-batch | backward + del loss + gc 后**故意继续持有 output 字典** |
| 泄漏物(embedding 输出) | `weakref.ref(x)` 探针 |

断言两条:① `output["model_output"]` 里不许有 grad_fn(契约);② 只剩 output 在手时
weakref 必须失效(行为)。引用链 `log_probs.grad_fn → 图 → checkpoint unpack hook →
frame → x` 完全由那一个 detach 决定生死:未修复必红、修复后必绿(本地双向验证过)。
另有健全性检查 `proj.weight.grad is not None`,确保 detach 没误伤梯度流。
纯 CPU、~4 秒,挂在 verl 的 cpu_unit_tests 里(`_on_cpu.py` 命名约定)。

## 6. 上游贡献(已合并 ✅)

- issue:[verl-project/verl#6698](https://github.com/verl-project/verl/issues/6698)
  (完整根因 + profiling 证据 + 修复前后对比)
- PR:[verl-project/verl#6699](https://github.com/verl-project/verl/pull/6699)
  ——**2026-06-12 由 maintainer wuxibin89 合并进 main**,merge commit `dbbf0853`。
  最终 diff:`transformer_impl.py` forward_step detach(一处)+ 回归测试
  `tests/workers/test_engine_forward_step_detach_on_cpu.py`。

### 6.1 PR 全过程(首次给 verl 投稿的流程记录)

1. **提交**:不从本地 pin 的旧版推送——抓取 main@41a5244 的原文件、在其上打补丁、
   走 GitHub contents API 提交到 fork 分支(本机 git push 不通,API 通)。
   插曲:第一次用错了 gh 登录的账号,关闭重发(#6696/#6697 → #6698/#6699),
   教训 = 对外发布前先 `gh api user` 核对身份。
2. **流程关卡**:签 CLA(cla-assistant,必须本人点)→ 外部 PR 的 CI 每次 push 都要
   maintainer 批("approve and run",在飞书群 ci-request 请求,Tianle Zhong 触发)。
3. **第一轮 CI**:2 失败,均与 diff 无关(cpu_unit_tests 在 main 同 commit 同错挂;
   VL e2e 是 runner OOM-killer)——逐条 triage 发在 PR 评论里,显著降低 reviewer 成本。
4. **maintainer 要求加测试** → 写回归测试(§5.1);**写测试时自查出 losses.py 那个
   commit 是死代码**(`Metric.append` 本就 detach)→ 主动 revert,diff 收敛到最小。
   自己抓出来比被 reviewer 抓出来好得多。
5. **第二轮 CI**:pre-commit 挂(ruff format 对新测试文件的 lambda 格式)→ 按 CI 日志
   给出的期望 diff 修复;其余 7 个 e2e/硬件失败用**跨轮对照**定性
   (同样的引擎代码上一轮全绿 → runner 抖动),triage 评论附证据请求重跑。
6. **合并轮**:pre-commit 绿,47 过 6 败(全是已 triage 的 flake),maintainer 直接合并。
   注:我们的回归测试在 CI 里其实**从未真正执行**——cpu_unit_tests 用 `pytest -x`,
   main 上那个坏的 distillation 测试按字母序排前面、首败即停;本地红/绿双向验证是
   它正确性的实际依据。

### 6.2 合并后的本地待办

- 升级 `third_party/verl` submodule 跨过 `dbbf0853` 后,可移除本地 sitecustomize 里的
  `VERL_TRAIN_METRICS_DETACH` patch(其余 patch 各有独立用途,逐个评估)。
- 在那之前本地 patch 继续生效,与上游修复语义相同,无冲突(patch 包的是同一个函数,
  对已修复的代码做 detach 是幂等的)。

## 7. 沉淀的工具与旋钮

| 东西 | 在哪 | 用途 |
|---|---|---|
| `TRAIN_EXPANDABLE_SEGMENTS`(默认 True) | run_grpo_fully_async.sh → sitecustomize | trainer-only expandable segments(全局 env 会崩 sleep-mode vLLM) |
| `USE_FUSED_KERNELS` | 同上 | fused linear-CE,消 lm_head logits 物化(见 docs/sft_fused_kernels.md) |
| `MODEL_DTYPE`(默认 bf16) | 同上 | master 参数 bf16,LoRA 下 fp32 master 纯浪费 ~15 GiB,还把 param_sync 加速一倍 |
| `TRAIN_METRICS_DETACH`(默认 True) | 同上 | **本案根因修复** |
| `TRAIN_MEM_SNAPSHOT=1` + `VERL_MEM_SNAPSHOT_AT_MB=N` | 同上 | 每 micro-batch 显存打印 + OOM/第 N 个 micro-batch 快照 dump |
| `scripts/debug/analyze_mem_snapshot.py` | scripts/debug/ | 快照按分配栈聚合;无栈(backward 线程)分配用事件夹逼定位 |

## 8. 经验教训

1. **"修一点好一点但峰值回弹"= 泄漏,不是容量问题。** 三次"差一点点"(40 MiB!)的
   OOM 报文把排查带偏了三轮;早一步画 per-micro-batch 显存曲线,一眼就能分清
   流量(泄漏)与存量(容量)。
2. **no_grad 段平坦 + 训练段线性增长**是计算图滞留的指纹,优先怀疑"谁拿着带
   grad_fn 的张量不放",而不是怀疑算子本身的峰值。
3. backward 线程的分配没有 Python 栈,`_record_memory_history` 要配合事件时间线
   夹逼;主动在泄漏早期 dump 比等 OOM dump 信息量大得多(环形缓冲)。
4. detach 检查要顺着**整条引用链**做:loss detach 了不代表 metrics 干净,metrics
   干净了不代表 model_output 干净——本案两处锚,修一处不够。
5. 对只读 submodule,sitecustomize 懒加载 meta-path 钩子(见 `src/tau2_airline_verl/patches/`)
   是干净的 patch 载体:按 env var 逐 patch 守门、不碰源码、fail-safe。
