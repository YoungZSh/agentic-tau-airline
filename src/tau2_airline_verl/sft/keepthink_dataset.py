"""verl `MultiTurnSFTDataset` subclass for tau2-airline SFT cold-start.

Two deviations from the stock dataset, both required to keep the SFT token
layout identical to RL rollout (whose assistant tokens are 100% `<think>`-bearing,
mask=1):

1. **Keep `<think>` on every assistant turn.** `MultiTurnSFTDataset` renders each
   message through the tokenizer's chat template. Qwen3's template strips
   `<think>` from every assistant turn that is *not* the current generation turn
   (gated by `loop.index0 > ns.last_query_index`), so reasoning would never enter
   the SFT loss. The template's inner branch already renders think whenever
   `reasoning_content` is present — it is only the outer positional gate that
   suppresses history turns. We flip that single gate to always-true, so every
   assistant turn carrying `reasoning_content` keeps its `<think>` block. One-line
   patch, reversible, fails loud if the upstream template changes.

2. **Inject tool schemas per row.** The airline tool schemas are identical for
   every row; writing them into parquet makes Arrow union-pollute the differing
   per-tool `parameters.properties` (verified: `book_reservation` comes back with
   foreign null-valued params). So we inject the original Python schemas here.

Used via verl SFT config: `data.custom_cls.path=<this file>`,
`data.custom_cls.name=KeepThinkMultiTurnSFTDataset`.
"""

from __future__ import annotations

from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset


def _maybe_apply_tiled_mlp() -> None:
    """Enable verl's TiledMLP on the FSDP SFT path (opt-in via the TILED_MLP env).

    verl's FSDP engine calls `apply_monkey_patch` WITHOUT `use_tiled_mlp`
    (third_party/verl/verl/workers/engine/fsdp/transformer_impl.py:292), so
    `model.tiled_mlp.enabled` is a dead knob on this path — only the megatron/veomni
    backends wire it. We can't edit the read-only submodule, so we apply verl's
    class-level (forward-only) TiledMLP patch ourselves. This module is the SFT
    `custom_cls`, imported in every trainer rank before the model's first forward,
    so patching `Qwen3MLP.forward` here takes effect even on the already-built
    module. No-op unless TILED_MLP is truthy. Numerically equivalent to the standard
    MLP (verified: output diff ~2e-4, grad ~2e-3, within verl's own 1e-2 threshold).
    """
    import os

    if os.environ.get("TILED_MLP", "").strip().lower() not in ("1", "true", "yes", "on"):
        return
    from verl.models.transformers.tiled_mlp import apply_tiled_mlp_monkey_patch

    apply_tiled_mlp_monkey_patch(num_shards=int(os.environ.get("TILED_MLP_SHARDS", "4")), model_type="qwen3")


_maybe_apply_tiled_mlp()


# The outer positional gate in Qwen3's chat template that suppresses <think> on
# history assistant turns. Flipping it to always-true lets the existing inner
# branch (`loop.last or (not loop.last and reasoning_content)`) render <think>
# for any turn that carries reasoning_content.
_THINK_GATE = "{%- if loop.index0 > ns.last_query_index %}"
_THINK_GATE_OPEN = "{%- if true %}"


def keep_think_chat_template(chat_template: str) -> str:
    """Patch a Qwen3 chat template so every assistant turn keeps its <think>.

    Idempotent: calling it on an already-patched template is a no-op (the SFT
    trainer shares one tokenizer across the train and val datasets, so this runs
    more than once).
    """
    if _THINK_GATE_OPEN in chat_template and _THINK_GATE not in chat_template:
        return chat_template  # already patched
    if _THINK_GATE not in chat_template:
        raise ValueError(
            "Qwen3 chat_template does not contain the expected think-gating anchor "
            f"{_THINK_GATE!r}; the keep-think patch must be reviewed against the "
            "current template before SFT can preserve reasoning."
        )
    return chat_template.replace(_THINK_GATE, _THINK_GATE_OPEN)


class KeepThinkMultiTurnSFTDataset(MultiTurnSFTDataset):
    """MultiTurnSFTDataset that preserves <think> per turn and injects airline tools."""

    def __init__(self, parquet_files, tokenizer, config, processor=None, max_samples=-1):
        tokenizer.chat_template = keep_think_chat_template(tokenizer.chat_template)
        super().__init__(
            parquet_files,
            tokenizer,
            config,
            processor=processor,
            max_samples=max_samples,
        )
        # Inject the airline tool schemas (identical for every row) rather than
        # reading a `tools` column, which Arrow would union-pollute.
        from tau2_airline_verl.env.airline_tool import airline_tool_schemas

        tools = airline_tool_schemas()
        self.tools = [tools for _ in range(len(self.messages))]


__all__ = ["KeepThinkMultiTurnSFTDataset", "keep_think_chat_template"]
