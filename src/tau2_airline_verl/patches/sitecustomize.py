# Auto-applied monkeypatches (via PYTHONPATH) for the disaggregated fully-async runs.
# Each patch is individually gated by an env var (set in run_grpo_fully_async.sh), so having
# this dir on PYTHONPATH is inert by itself.
#
# Injection: Python auto-imports `sitecustomize` from any sys.path entry at interpreter startup.
# CRITICAL: we must NOT import torch/verl at startup — in a Ray worker that runs BEFORE Ray
# assigns CUDA_VISIBLE_DEVICES, and eagerly touching torch/verl there breaks CUDA init
# ("No CUDA GPUs are available"). So we only register meta-path hooks here and apply each patch
# lazily, the moment verl itself imports the target module (well after Ray device setup).
# All hooks are wrapped so they can never break interpreter startup (failure => no patch =>
# the run later fails exactly as it would unpatched, i.e. fail-safe).
import os


def _install_lazy_patch(target, apply_fn, tag):
    """One-shot meta-path hook: when `target` finishes importing, run apply_fn(module)."""
    import importlib.abc
    import importlib.util
    import sys

    class _PatchOnImport(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target_mod=None):
            if name != target:
                return None
            # Avoid recursion while we resolve the real spec for the same module.
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(name)
            finally:
                if self not in sys.meta_path:
                    sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return None
            _orig_exec = spec.loader.exec_module

            def exec_module(module, _orig_exec=_orig_exec):
                _orig_exec(module)
                try:
                    apply_fn(module)
                except Exception as _e:
                    print(f"[{tag}] NOT applied ({type(_e).__name__}: {_e})")
                sys.meta_path.remove(self)  # one-shot: stop intercepting after applied

            spec.loader.exec_module = exec_module
            return spec

    try:
        sys.meta_path.insert(0, _PatchOnImport())
    except Exception as _e:
        print(f"[{tag}] hook NOT installed ({type(_e).__name__}: {_e})")


# ---------------------------------------------------------------------------------------------
# Patch 1: decoupled PPO on a SINGLE training GPU (VERL_DECOUPLED_1GPU_PATCH=1).
#
# Why: verl's decoupled path (algorithm.rollout_correction.bypass_mode=False) snapshots the
# proximal policy π_prox via verl.utils.fsdp_utils.fsdp2_sharded_save_to_cpu, which all-gathers
# FSDP2-sharded DTensor params and asserts at least one DTensor exists. With one training GPU
# (fsdp_size=1) params are unsharded plain tensors → it raises
#   "No DTensor-type parameters found in the model. FSDP2 sharding may not be enabled."
# Decoupled PPO needs NO sharding — the anchor is just a param copy + a forward pass — so we
# fall back to a plain CPU copy of the TRAINABLE params (= LoRA adapter; the frozen base is
# identical across policy versions, so it needs no save/restore). Multi-GPU (DTensor present)
# delegates to the original implementation unchanged, so this is purely additive.
# ---------------------------------------------------------------------------------------------
if os.environ.get("VERL_DECOUPLED_1GPU_PATCH") == "1":

    def _apply_decoupled_patch(mod):
        # Runs at the time verl.utils.fsdp_utils finishes importing — Ray has already set the
        # worker's device, so importing torch/DTensor here is safe.
        import torch
        from torch.distributed.tensor import DTensor

        _orig_save = mod.fsdp2_sharded_save_to_cpu
        _orig_load = mod.fsdp2_sharded_load_from_cpu

        def _save_to_cpu_1gpu(model):
            if any(isinstance(p, DTensor) for p in model.parameters()):
                return _orig_save(model)  # sharded (>=2 train GPUs): unchanged
            cpu_state = {
                name: (param.detach().to("cpu", copy=True), None)
                for name, param in model.named_parameters()
                if param.requires_grad
            }
            return cpu_state, None  # global_spec=None: sentinel for the plain path

        def _load_from_cpu_1gpu(model, cpu_sharded_state, target_spec):
            if target_spec is not None:
                return _orig_load(model, cpu_sharded_state, target_spec)  # sharded: unchanged
            with torch.no_grad():
                params = dict(model.named_parameters())
                for name, (cpu_tensor, _spec) in cpu_sharded_state.items():
                    p = params.get(name)
                    if p is not None:
                        p.copy_(cpu_tensor.to(p.device))

        mod.fsdp2_sharded_save_to_cpu = _save_to_cpu_1gpu
        mod.fsdp2_sharded_load_from_cpu = _load_from_cpu_1gpu
        print(
            "[decoupled_1gpu_patch] applied: fsdp2 proximal-anchor save/load falls back to a "
            "plain trainable-param state_dict when params are unsharded (single training GPU)"
        )

    _install_lazy_patch("verl.utils.fsdp_utils", _apply_decoupled_patch, "decoupled_1gpu_patch")


# ---------------------------------------------------------------------------------------------
# Patch 2: expandable segments on the TRAINING worker only (VERL_TRAIN_EXPANDABLE_SEGMENTS=1).
#
# Why: all three fa_areal OOMs (2026-06-11) died in the actor update's backward with a multi-GiB
# allocation failing while ~10 GiB sat reserved-but-unallocated — allocator fragmentation.
# expandable_segments reclaims that. But we canNOT just export PYTORCH_CUDA_ALLOC_CONF globally
# like run_sft.sh does: the rollout vLLM servers run with enable_sleep_mode=True, and vLLM's
# CuMemAllocator __init__ hard-asserts "expandable_segments:True" is NOT in that env var
# (vllm/device_allocator/cumem.py — memory pools are incompatible with expandable segments).
# Ray workers inherit the launch shell's env, so a global export would crash every vLLM server
# at startup. Instead we flip the allocator AT RUNTIME (verl's own set_expandable_segments →
# torch.cuda.memory._set_allocator_settings) inside TrainingWorker.__init__, which only ever
# runs in the trainer process. verl itself runs trainers with expandable segments on in the
# colocated "naive" sync path (engine_workers.sync_rollout_weights), and our nccl
# checkpoint-engine path never reaches those toggles — this patch fills exactly that gap.
# NCCL + expandable segments is proven on this box (run_sft.sh: 3-GPU ZeRO-3, global export).
# ---------------------------------------------------------------------------------------------
if os.environ.get("VERL_TRAIN_EXPANDABLE_SEGMENTS") == "1":

    def _apply_expandable_patch(mod):
        _orig_init = mod.TrainingWorker.__init__

        def _init_with_expandable(self, *args, **kwargs):
            from verl.utils.device import set_expandable_segments

            set_expandable_segments(True)
            print("[train_expandable_segments] enabled on this TrainingWorker process")
            _orig_init(self, *args, **kwargs)

        mod.TrainingWorker.__init__ = _init_with_expandable
        print("[train_expandable_segments] applied: TrainingWorker.__init__ wrapped")

    _install_lazy_patch(
        "verl.workers.engine_workers", _apply_expandable_patch, "train_expandable_segments"
    )


# ---------------------------------------------------------------------------------------------
# Patch 3: allocator-history recording + OOM snapshot dump (VERL_TRAIN_MEM_SNAPSHOT=1).
#
# Why: the 2026-06-11 OOM hunt fixed three real consumers (fragmentation, logits
# materialization, fp32 master params ~15GB) yet the actor-update backward peak returned to
# ~70GiB every time — something in the update path we can't see statically. This patch records
# allocator history in the trainer process and, when train_mini_batch dies with CUDA OOM, dumps
# a torch memory snapshot (every live block + its python stack) to VERL_MEM_SNAPSHOT_DIR before
# re-raising. Analyze with torch.cuda._memory_viz or a stack-aggregation script.
# Diagnostic only — off unless explicitly enabled; recording overhead is CPU-side and small.
# ---------------------------------------------------------------------------------------------
if os.environ.get("VERL_TRAIN_MEM_SNAPSHOT") == "1":

    def _apply_memsnap_patch(mod):
        import torch

        _orig_init = mod.TrainingWorker.__init__

        def _init_with_recording(self, *args, **kwargs):
            # 1.5M entries: the 200k default window only covered ~1 micro-batch — the
            # accumulated 62GiB had no alloc events left to attribute stacks to.
            torch.cuda.memory._record_memory_history(max_entries=1500000)
            print("[train_mem_snapshot] allocator history recording ON (trainer process)")
            _orig_init(self, *args, **kwargs)

        mod.TrainingWorker.__init__ = _init_with_recording

        _orig_tmb = mod.TrainingWorker.train_mini_batch

        def _train_mini_batch_with_snapshot(self, *args, **kwargs):
            try:
                return _orig_tmb(self, *args, **kwargs)
            except torch.OutOfMemoryError:
                out = os.path.join(
                    os.environ.get("VERL_MEM_SNAPSHOT_DIR", "/tmp"),
                    f"oom_snapshot_{os.getpid()}.pickle",
                )
                try:
                    torch.cuda.memory._dump_snapshot(out)
                    print(f"[train_mem_snapshot] OOM memory snapshot dumped to {out}")
                except Exception as _e:
                    print(f"[train_mem_snapshot] snapshot dump FAILED ({type(_e).__name__}: {_e})")
                raise

        mod.TrainingWorker.train_mini_batch = _train_mini_batch_with_snapshot
        print("[train_mem_snapshot] applied: history recording + OOM snapshot dump")

    _install_lazy_patch("verl.workers.engine_workers", _apply_memsnap_patch, "train_mem_snapshot")

    def _apply_microbatch_memlog_patch(mod):
        import torch

        _orig_fs = mod.FSDPEngineWithLMHead.forward_step
        _counter = {"n": 0}
        _snap_at = int(os.environ.get("VERL_MEM_SNAPSHOT_AT_MB", "0"))

        def _forward_step_with_memlog(self, *args, **kwargs):
            _counter["n"] += 1
            a = torch.cuda.memory_allocated() / (1 << 30)
            r = torch.cuda.memory_reserved() / (1 << 30)
            print(f"[train_mem_snapshot] micro-batch #{_counter['n']}: allocated={a:.2f}GiB reserved={r:.2f}GiB")
            # Early snapshot at a fixed micro-batch count: leaked blocks from the few
            # preceding micro-batches still have their alloc events inside the history
            # window (the OOM-time snapshot is hundreds of micro-batches too late for that).
            if _snap_at and _counter["n"] == _snap_at:
                out = os.path.join(
                    os.environ.get("VERL_MEM_SNAPSHOT_DIR", "/tmp"),
                    f"mb{_snap_at}_snapshot_{os.getpid()}.pickle",
                )
                try:
                    torch.cuda.memory._dump_snapshot(out)
                    print(f"[train_mem_snapshot] micro-batch #{_snap_at} snapshot dumped to {out}")
                except Exception as _e:
                    print(f"[train_mem_snapshot] mb-snapshot dump FAILED ({type(_e).__name__}: {_e})")
            return _orig_fs(self, *args, **kwargs)

        mod.FSDPEngineWithLMHead.forward_step = _forward_step_with_memlog
        print("[train_mem_snapshot] applied: per-micro-batch memory logging")

    _install_lazy_patch(
        "verl.workers.engine.fsdp.transformer_impl", _apply_microbatch_memlog_patch, "train_mem_snapshot"
    )


# ---------------------------------------------------------------------------------------------
# Patch 4: detach metric tensors in the actor loss (VERL_TRAIN_METRICS_DETACH=1).
#
# Why: verl's ppo_loss stores LIVE loss tensors in the metrics dict —
#   metrics["actor/pg_loss"] = Metric(value=pg_loss)   (verl/workers/utils/losses.py:119)
# and pg_metrics (ppo_kl etc.) likewise derive from log_prob with grad_fn attached. These
# Metric objects ride forward_step's meta_info into output_lst, which lives until the END of
# the whole mini-batch's micro-batch loop — anchoring part of every micro-batch's autograd
# graph. Memory profiling (run 20260611_190521) shows ~0.22 GiB leaking per TRAINING
# micro-batch (~2 x [T,4096] backward-thread tensors), flat during no-grad forwards — the
# signature of graph retention, not of compute. Metrics are reporting-only (aggregated after
# the loop, never backprop'd), so detaching is behavior-neutral.
# ---------------------------------------------------------------------------------------------
if os.environ.get("VERL_TRAIN_METRICS_DETACH") == "1":

    def _apply_metrics_detach_patch(mod):
        import torch

        def _detach_metrics(metrics):
            for v in metrics.values():
                inner = getattr(v, "value", None)
                if torch.is_tensor(inner) and inner.grad_fn is not None:
                    v.value = inner.detach()

        for fn_name in ("ppo_loss", "sft_loss", "value_loss"):
            _orig = getattr(mod, fn_name, None)
            if _orig is None:
                continue

            def _wrapped(*args, _orig=_orig, **kwargs):
                loss, metrics = _orig(*args, **kwargs)
                _detach_metrics(metrics)
                return loss, metrics

            _wrapped.__name__ = f"_detached_{fn_name}"
            setattr(mod, fn_name, _wrapped)
        print("[train_metrics_detach] applied: loss-fn metric tensors detached from autograd graph")

    _install_lazy_patch("verl.workers.utils.losses", _apply_metrics_detach_patch, "train_metrics_detach")

    # The REAL anchor (profiled via mb270 snapshot, run 20260611_212557): forward_step returns
    # model_output whose log_probs/entropy still carry grad_fn; output_lst holds them until the
    # whole mini-batch (~400 micro-batches) finishes. The retained graph pins, per TRAINING
    # micro-batch, the checkpoint-frame's saved embedding output + its backward-tail grad
    # (2 x ~139MiB = the measured 0.27GiB/micro-batch leak; embedding output requires grad via
    # PEFT enable_input_require_grads). loss is returned separately and backward() runs in the
    # caller BEFORE these outputs are consumed for metrics only -> detaching is behavior-neutral.
    def _apply_output_detach_patch(mod):
        import torch

        _orig_fs = mod.FSDPEngineWithLMHead.forward_step

        def _forward_step_detached(self, *args, **kwargs):
            loss, output = _orig_fs(self, *args, **kwargs)
            mo = output.get("model_output") if isinstance(output, dict) else None
            if isinstance(mo, dict):
                for k, v in mo.items():
                    if torch.is_tensor(v) and v.grad_fn is not None:
                        mo[k] = v.detach()
            return loss, output

        mod.FSDPEngineWithLMHead.forward_step = _forward_step_detached
        print("[train_metrics_detach] applied: forward_step model_output detached from autograd graph")

    _install_lazy_patch(
        "verl.workers.engine.fsdp.transformer_impl", _apply_output_detach_patch, "train_metrics_detach"
    )
