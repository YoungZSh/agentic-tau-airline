# SFT 显存优化:Fused Linear Cross-Entropy(`use_fused_kernels`)

> 适用场景:full-param Qwen3-8B 多轮 SFT(纯 ZeRO-3,A100 80GB,序列长达 32k)
> 入口:`scripts/train/run_sft.sh` 的 `USE_FUSED_KERNELS` 开关(默认 **True**)
> 验证:2026-06-05 —— 数值等价(实测 ~1e-6)+ 端到端跑通、不再 OOM

本文记录为什么 32k 长序列 SFT 会在 backward 撞 OOM,以及用 verl 的 **fused linear
cross-entropy**(`model.use_fused_kernels`)如何把它消掉。末尾附相关的显存旋钮全景
(gradient checkpointing / tiled MLP / 分配器碎片)。

---

## 1. 问题:32k SFT 在第一个 backward 就 OOM

### 1.1 显存账(full-param 8B,纯 ZeRO-3)

full-param 8B 的**静态状态**与序列长度无关,是固定的:

| 项 | 大小 |
|----|----|
| params(bf16) | 16 GB |
| grad(bf16) | 16 GB |
| AdamW 优化器状态(fp32 master + m + v) | 96 GB |
| **合计静态** | **128 GB** |

ZeRO-3 把这 128 GB 沿 DP 分片(是 sharding,不是 offload):

- 3 卡 → ~42.7 GB/卡,留 ~37 GB 给 activation/logits
- 2 卡 → 64 GB/卡,只剩 ~16 GB —— 放不下 32k 的峰值,必 OOM

### 1.2 炸点是 lm_head 的 logits,gradient checkpointing 救不了

LM 训练算 loss 的最后几步:

```
hidden  [T, hidden]
  └─ lm_head:  [T, hidden] × [hidden, vocab] ─→ logits [T, vocab]   ← 炸点
       └─ log_softmax + gather(target) ─→ log_prob [T]
            └─ loss = -mean(log_prob over mask)
```

`logits [T, vocab]` 这一步物化的张量,在 Qwen3-8B(`vocab=151936`)、`T≈32768` 时:

```
32768 × 151936 × 2 bytes(bf16) ≈ 9.96 GB
```

这正是当初 OOM 报文里的那笔分配(`Tried to allocate 9.41 GiB`):

```
GPU 2 has a total capacity of 79.25 GiB of which 8.45 GiB is free.
this process has 70.79 GiB memory in use.   ← 训练进程自己用了 70.79GB,再要 9.41GB 就爆
```

关键:**gradient checkpointing 压不到这块**。gc 只在 transformer 各层边界存 activation、反向
按层重算,压的是「层 activation」;而 logits 在 lm_head、最后一层、gc 边界之外,必然完整物化。
词表越大越狠,Qwen3 的 15 万词表把它放大到了 ~10 GB。

---

## 2. 原理:fused linear cross-entropy 怎么省显存

`use_fused_kernels=True` 把「lm_head 投影 + log_softmax + gather log_prob」**融合成一个算子,
并沿 token 维度分块(chunk)计算**:

```
普通:  一次性算出整个 [32768, 151936] logits → 再求 log_prob          (峰值 ~10GB)
fused: 切成小块,每块只算 [chunk, 151936] logits → 立刻得到该块 log_prob
       → 丢弃这块 logits → 下一块,从不在显存里同时持有完整 logits
```

峰值显存从 `O(T × vocab)` 降到 `O(chunk × vocab) + O(T × hidden)`。因为 `hidden(4096) ≪
vocab(151936)`,那 ~10 GB 的 logits 峰值基本消失。**反向也分块重算**,只对 hidden 和 lm_head
权重累积梯度。

> 注意:它「fused」的只是最后 logits→log_prob 这一步,不是整个模型——但 OOM 恰好就卡在这一步。

---

## 3. 用法

### 3.1 通过 run_sft.sh(默认已开)

`run_sft.sh` 默认 `USE_FUSED_KERNELS=True`,直接跑即可:

```bash
CUDA_VISIBLE_DEVICES=<3张干净卡> NGPUS_PER_NODE=3 bash scripts/train/run_sft.sh
```

要关掉对照(二分定位问题时):

```bash
USE_FUSED_KERNELS=False bash scripts/train/run_sft.sh
```

### 3.2 直接传 Hydra 覆盖

脚本最终落到的就是这个 verl 参数:

```
model.use_fused_kernels=True
```

### 3.3 后端选择(`fused_kernel_options.impl_backend`)

| backend | 实现 | 特点 |
|----|----|----|
| `torch`(默认) | `FusedLinearForPPO`(纯 PyTorch 分块) | 无额外依赖,省显存,速度略降 |
| `triton` | `linear_cross_entropy`(Triton kernel) | 更快更省,需要可用的 triton |

默认 `torch` 后端足够;本仓库未切到 triton。

---

## 4. verl 里的接线情况

FSDP engine 在构建模型时读 `model_config.use_fused_kernels` 并传给 `apply_monkey_patch`,
**原生接线、确定生效**:

```
third_party/verl/verl/workers/engine/fsdp/transformer_impl.py:291
    use_fused_kernels = self.model_config.use_fused_kernels
    apply_monkey_patch(..., use_fused_kernels=use_fused_kernels, ...)
```

Qwen3 的 fused forward 实现在 `verl/models/transformers/qwen3_5.py` 与 `dense_common.py`
(`FusedLinearForPPO` / `linear_cross_entropy`)。

> ⚠️ 对比:同一处 `apply_monkey_patch` 调用**没有**传 `use_tiled_mlp`,所以
> `model.tiled_mlp.enabled` 在 FSDP 路径下是死开关(见 §6.2)。fused kernels 没这个问题。

---

## 5. 数值等价性(实测)

fused 与标准实现**数学等价**,只是分块/顺序不同;差异仅来自浮点。eager 对比
(`FusedLinearForPPO` vs 标准 `matmul → cross_entropy`,fp32,Qwen3 同款词表):

| T(tokens) | hidden | vocab | max\|Δlogprob\| | max\|Δentropy\| | max\|Δgrad\| |
|----|----|----|----|----|----|
| 1937 | 3584 | 152064 | 3.81e-06 | 5.72e-06 | 2.38e-07 |
| 2169 | 896 | **151936**(Qwen3) | 1.91e-06 | 9.54e-07 | 1.06e-05 |
| 8192 | 4096 | 102400 | 3.81e-06 | 0.00e+00 | 2.98e-07 |

全部在 **~1e-6** 量级,远小于 verl 自带测试的 1e-4 阈值。forward 的 logprob/entropy、backward
的梯度都对得上 → **开它不影响训练数值**。

复现核心逻辑(单卡即可,无需 optimizer/长序列):

```python
from verl.utils.experimental.torch_functional import FusedLinearForPPO
# 标准: logits = hidden @ weight.T; lp_ref = -cross_entropy(logits, labels)
# fused: lp_fused, ent = FusedLinearForPPO()(hidden, weight, labels, temperature=1.0)
# 对比:  torch.allclose(lp_ref, lp_fused, atol=1e-4, rtol=1e-4)
```

---

## 6. 注意事项 / 坑

### 6.1 fused kernels + Ulysses SP>1 有过数值 bug(issue #6068)

fused forward 曾在 SP 切片**之后**才 `torch.roll(input_ids)`,导致每个 SP rank 的 shard 边界
位置预测错 label —— **每 rank 每 micro-batch 偏 ~1 个位置,表现为 SP>1 下训练质量缓慢退化**。

- 我们 pin 的 verl 已含修复(`tests/special_distributed/test_fused_kernels_ulysses_sp.py`)。
- 当前 SFT `sp_size=1`(纯 ZeRO-3),没有 shard 边界,**不触发**。
- **若将来重开 SP + fused,务必确认用的是修复后的版本(传 `shift_labels`)。**

### 6.2 verl 自带的 `test_linear_cross_entropy.py` 在本机跑不起来

不是数值问题,是它内部 `compile(dynamic=True)` 撞上 torch 2.8 + 当前 triton 的 dynamo bug
(`SymNodeVariable has no attribute 'value'`)。那个 compile 是测试特意加的、非训练必经路径,
所以本文 §5 改用 eager 直接测底层 kernel。

---

## 7. 全景:各显存旋钮压不同部位

32k 长序列 full-param SFT 的显存优化是组合拳,每个旋钮压不同部位:

| 旋钮 | 压哪块 | 默认 | 在 run_sft.sh 的开关 |
|----|----|----|----|
| 纯 ZeRO-3(`sp_size=1`) | 静态 128GB 沿 DP 分片 | on | `SP_SIZE=1` |
| gradient checkpointing | transformer 层 activation | on | `ENABLE_GRADIENT_CHECKPOINTING` |
| **fused linear CE** | **lm_head logits(~10GB)** | **on** | **`USE_FUSED_KERNELS`** |
| tiled MLP | MLP activation | **off** | `TILED_MLP`(见下) |
| `expandable_segments` | 分配器碎片 | on | `PYTORCH_CUDA_ALLOC_CONF` |

**结论:fused kernels 是这组里最对症 OOM 的一个**——它消掉的 ~10 GB logits 正是 backward 撞墙
的那笔分配。加上之后,3 卡纯 ZeRO-3 + 32k 已能跑通、不再 OOM。

### 关于 tiled MLP(默认关,opt-in)

tiled MLP 把 MLP forward/backward 沿 token 分块,进一步压 MLP activation,数值也验证等价
(out diff ~2.4e-4,grad ~2e-3,在 verl 的 1e-2 阈值内)。但 **verl 的 FSDP engine 没把
`model.tiled_mlp` 接进 `apply_monkey_patch`**(只有 megatron/veomni 接了),所以它在本路径下是
死开关。我们用 custom_cls(`src/tau2_airline_verl/sft/keepthink_dataset.py`)在模型 forward 前
显式调 `apply_tiled_mlp_monkey_patch` 作为 workaround,由 `TILED_MLP` 环境变量守门。

**默认关**:因为 verl 自己没在这条路上验证它,保守起见只走 fused kernels(原生接线);需要时
`TILED_MLP=True bash scripts/train/run_sft.sh` 开启。
