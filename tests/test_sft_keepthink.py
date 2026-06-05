"""Verify the keep-think SFT dataset preserves <think> on every assistant turn.

Unlike the other tests in this dir, this one needs the real Qwen3 tokenizer and
the built SFT parquet, so it self-skips when either is absent (e.g. on CI):
    QWEN3_8B_PATH=... python -m pytest tests/test_sft_keepthink.py -q
Build the parquet first with:
    python -m tau2_airline_verl.data.build_sft_parquet
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from omegaconf import OmegaConf

QWEN3_8B_PATH = os.environ.get("QWEN3_8B_PATH")
VAL_PARQUET = Path("data/tau2_airline_sft/val.parquet")

pytestmark = pytest.mark.skipif(
    not QWEN3_8B_PATH or not Path(QWEN3_8B_PATH).exists() or not VAL_PARQUET.exists(),
    reason="needs QWEN3_8B_PATH tokenizer and a built data/tau2_airline_sft/val.parquet",
)


def _dataset_config():
    return OmegaConf.create(
        {
            "pad_mode": "no_padding",
            "truncation": "right",
            "max_length": 32768,
            "messages_key": "messages",
            "tools_key": "tools",
            "enable_thinking_key": "enable_thinking",
            "enable_thinking_default": None,
            "apply_chat_template_kwargs": {},
            "ignore_input_ids_mismatch": True,
            "shuffle": False,
        }
    )


def _load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(QWEN3_8B_PATH, trust_remote_code=True)


def test_keepthink_preserves_think_on_every_turn():
    from tau2_airline_verl.sft.keepthink_dataset import KeepThinkMultiTurnSFTDataset

    tok = _load_tokenizer()
    ds = KeepThinkMultiTurnSFTDataset(
        parquet_files=str(VAL_PARQUET), tokenizer=tok, config=_dataset_config()
    )
    item = ds[0]
    input_ids = item["input_ids"].tolist()
    loss_mask = item["loss_mask"].tolist()

    full_text = tok.decode(input_ids)
    n_assistant = int(ds.dataframe.iloc[0]["extra_info"]["num_assistant_turns"])

    # Every assistant turn carries reasoning in this data, so <think> should
    # appear roughly once per assistant turn (allow the final turn to lack it).
    think_count = full_text.count("<think>")
    assert think_count >= n_assistant - 1, (
        f"expected ~{n_assistant} <think> blocks, got {think_count}"
    )

    # <think> must land inside the trained (loss_mask==1) region, not just context.
    trained_text = tok.decode([t for t, m in zip(input_ids, loss_mask) if m == 1])
    assert "<think>" in trained_text and "</think>" in trained_text
    assert "<tool_call>" in trained_text, "tool calls should be trained too"


def test_user_and_tool_tokens_are_masked_out():
    from tau2_airline_verl.sft.keepthink_dataset import KeepThinkMultiTurnSFTDataset

    tok = _load_tokenizer()
    ds = KeepThinkMultiTurnSFTDataset(
        parquet_files=str(VAL_PARQUET), tokenizer=tok, config=_dataset_config()
    )
    item = ds[0]
    input_ids = item["input_ids"].tolist()
    loss_mask = item["loss_mask"].tolist()

    masked_text = tok.decode([t for t, m in zip(input_ids, loss_mask) if m == 0])
    # tool observations are rendered as <tool_response>; user/system are mask 0.
    assert "<tool_response>" in masked_text
    # at least some tokens are trained and some are not
    assert 0 < sum(loss_mask) < len(loss_mask)


def test_keepthink_keeps_more_think_than_stock():
    """The whole point: stock dataset strips history-turn <think>; we don't."""
    from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset

    from tau2_airline_verl.sft.keepthink_dataset import KeepThinkMultiTurnSFTDataset

    # independent tokenizers — KeepThink mutates tokenizer.chat_template in place
    stock = MultiTurnSFTDataset(
        parquet_files=str(VAL_PARQUET), tokenizer=_load_tokenizer(), config=_dataset_config()
    )
    keep = KeepThinkMultiTurnSFTDataset(
        parquet_files=str(VAL_PARQUET), tokenizer=_load_tokenizer(), config=_dataset_config()
    )
    stock_think = stock.tokenizer.decode(stock[0]["input_ids"].tolist()).count("<think>")
    keep_think = keep.tokenizer.decode(keep[0]["input_ids"].tolist()).count("<think>")
    assert keep_think > stock_think, f"keep={keep_think} stock={stock_think}"
