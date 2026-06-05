"""Build a verl multi-turn SFT parquet from the AReaL airline SFT subset.

Source: ``local_datasets/tau2_sft_train.jsonl`` (the AReaL
``inclusionAI/AReaL-tau2-data`` SFT split). We use only the **airline** subset
(``source_dialog_id`` prefixed ``airline``): 12,842 per-turn rows spanning 999
dialogs.

Each dialog's rows share a strict message *prefix* (verified: ``len(messages)``
is monotonic in ``turn_index``), so the max-``turn_index`` row's
``messages + [answer]`` reconstructs the full conversation. We emit **one row per
dialog** (999) rather than one per turn (12,842) to avoid prefix-overlap, which
would re-train earlier turns many times.

Alignment with RL rollout (so this works as a cold start for GRPO):
- the leading system message is replaced with tau2 airline ``policy.md``
  (``build_system_prompt()``) — the RL/eval system prompt — not the dataset's
  ``<instructions>`` variant;
- ``reasoning`` (history turns) / ``thinking`` (the answer turn) become a
  ``reasoning_content`` field; the keep-think chat template
  (``KeepThinkMultiTurnSFTDataset``) renders it as ``<think>...</think>``;
- ``tool_calls`` are converted to OpenAI form
  ``{"type": "function", "function": {"name", "arguments": <json-str>}}`` —
  ``arguments`` is kept as a JSON *string* so Arrow does not union-pollute the
  nested dict when writing parquet.

Tools are NOT written here (Arrow union-pollutes the differing per-tool
``parameters.properties``); ``KeepThinkMultiTurnSFTDataset`` injects them per row.

Usage:
    python -m tau2_airline_verl.data.build_sft_parquet --out_dir data/tau2_airline_sft
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict

import datasets

from tau2_airline_verl.agents.qwen3_prompt import build_system_prompt

DEFAULT_SRC = "local_datasets/tau2_sft_train.jsonl"
DEFAULT_OUT = "data/tau2_airline_sft"
VAL_FRACTION = 0.05
SEED = 0
# Conversations run very long (median ~15k tok); full-param 8B on 2×A100 can't fit
# the extreme tail. Drop dialogs whose rendered length exceeds this (~83 of 999).
# Set to 0 to disable. The SFT dataset re-caps at max_length with truncation as a
# second guard.
DEFAULT_MAX_TOKENS = 32768


def _convert_assistant(msg: dict) -> dict:
    """tau2/AReaL assistant turn -> verl/OpenAI assistant message.

    Reasoning (``reasoning`` on history turns, ``thinking`` on the answer turn)
    becomes ``reasoning_content`` (bare text; the chat template adds the tags).
    ``tool_calls`` become OpenAI function calls with ``arguments`` json-encoded.
    """
    out: dict = {"role": "assistant", "content": msg.get("content") or ""}
    think = msg.get("reasoning") or msg.get("thinking")
    if think and str(think).strip():
        out["reasoning_content"] = think
    tool_calls = msg.get("tool_calls")
    if tool_calls:
        out["tool_calls"] = [
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc.get("arguments", {}), ensure_ascii=False),
                },
            }
            for i, tc in enumerate(tool_calls)
        ]
    return out


def _convert_message(msg: dict, policy: str) -> dict:
    role = msg.get("role")
    if role == "system":
        return {"role": "system", "content": policy}
    if role == "user":
        return {"role": "user", "content": msg.get("content") or ""}
    if role == "tool":
        # Qwen3 renders tool turns as <tool_response>; only `content` is read.
        return {"role": "tool", "content": msg.get("content") or ""}
    if role == "assistant":
        return _convert_assistant(msg)
    raise ValueError(f"unexpected message role: {role!r}")


def _reconstruct_dialogs(src: str) -> dict[str, list[dict]]:
    """Group airline rows by source_dialog_id; rebuild each full conversation."""
    by_dialog: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    with open(src) as f:
        for line in f:
            obj = json.loads(line)
            sid = str(obj.get("metadata", {}).get("source_dialog_id", ""))
            if not sid.startswith("airline"):
                continue
            ti = obj.get("metadata", {}).get("turn_index")
            by_dialog[sid].append((ti if ti is not None else -1, obj))

    dialogs: dict[str, list[dict]] = {}
    for sid, rows in by_dialog.items():
        _, max_row = max(rows, key=lambda x: x[0])
        dialogs[sid] = list(max_row["messages"]) + [max_row["answer"]]
    return dialogs


def _to_row(sid: str, raw_messages: list[dict], policy: str) -> dict:
    conv = [_convert_message(m, policy) for m in raw_messages]
    if not conv or conv[0]["role"] != "system":
        conv = [{"role": "system", "content": policy}] + conv
    n_assistant = sum(1 for m in conv if m["role"] == "assistant")
    return {
        "messages": conv,
        "extra_info": {"source_dialog_id": sid, "num_assistant_turns": n_assistant},
    }


def _filter_overlong(rows: list[dict], max_tokens: int) -> list[dict]:
    """Drop dialogs whose rendered length exceeds ``max_tokens``.

    Length is measured with the *keep-think* chat template + airline tool schemas
    — the exact rendering used at training time — so the threshold matches the
    SFT ``max_length``. If the tokenizer is unavailable (no ``QWEN3_8B_PATH`` /
    transformers), filtering is skipped with a warning so the build stays offline-able.
    """
    try:
        from transformers import AutoTokenizer

        from tau2_airline_verl.env.airline_tool import airline_tool_schemas
        from tau2_airline_verl.sft.keepthink_dataset import keep_think_chat_template

        path = os.environ.get("QWEN3_8B_PATH")
        if not path:
            print("[build_sft] QWEN3_8B_PATH unset; skipping token filter")
            return rows
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        tok.chat_template = keep_think_chat_template(tok.chat_template)
        tools = airline_tool_schemas()
    except Exception as exc:  # noqa: BLE001 — degrade gracefully to no-filter
        print(f"[build_sft] tokenizer unavailable ({exc}); skipping token filter")
        return rows

    kept, dropped = [], 0
    for row in rows:
        n_tok = len(
            tok.apply_chat_template(
                row["messages"], tools=tools, add_generation_prompt=False, tokenize=True
            )
        )
        if n_tok <= max_tokens:
            kept.append(row)
        else:
            dropped += 1
    print(f"[build_sft] token filter (<= {max_tokens}): kept {len(kept)}, dropped {dropped}")
    return kept


def build(
    out_dir: str = DEFAULT_OUT, src: str = DEFAULT_SRC, max_tokens: int = DEFAULT_MAX_TOKENS
) -> None:
    policy = build_system_prompt()  # tau2 airline policy.md (same for every row)
    dialogs = _reconstruct_dialogs(src)
    rows = [_to_row(sid, msgs, policy) for sid, msgs in dialogs.items()]

    if max_tokens and max_tokens > 0:
        rows = _filter_overlong(rows, max_tokens)

    rng = random.Random(SEED)
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * VAL_FRACTION))
    splits = {"val": rows[:n_val], "train": rows[n_val:]}

    os.makedirs(out_dir, exist_ok=True)
    for split, data in splits.items():
        path = os.path.join(out_dir, f"{split}.parquet")
        datasets.Dataset.from_list(data).to_parquet(path)
        print(f"{split}: {len(data)} dialogs -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=DEFAULT_OUT)
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument(
        "--max_tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="drop dialogs longer than this many tokens (0 disables)",
    )
    args = ap.parse_args()
    build(args.out_dir, args.src, args.max_tokens)
