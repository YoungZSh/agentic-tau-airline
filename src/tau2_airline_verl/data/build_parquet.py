"""Build verl GRPO parquet datasets from tau2 airline tasks (plan §B).

Schema mirrors verl's own agent-loop preprocessor
(`examples/data_preprocess/gsm8k_tool_agent_loop.py`). Each row carries ONLY the
system prompt (airline policy.md); the multi-turn conversation — tau2's fixed
greeting, the user requests (gpt-5 simulator), tool calls and observations — is
generated at rollout time inside `Tau2AirlineAgentLoop.run()`. The task identity
flows through `extra_info.task_id`; the top-level `agent_name` column routes the
row to our registered `tau2_airline` agent loop (agent_loop.py:516-518).

`reward_model` is a harmless placeholder — the real reward is produced by the
agent loop (reward_score -> rm_scores[-1]) via tau2's official evaluator, so no
rule-based reward manager runs.

Usage:
    python -m tau2_airline_verl.data.build_parquet --out_dir data/tau3_airline
"""

from __future__ import annotations

import argparse
import os

import datasets

from tau2_airline_verl.agents.qwen3_prompt import build_system_prompt
from tau2_airline_verl.data.splits import load_eval_tasks, load_train_tasks

DATA_SOURCE = "tau2_airline"
AGENT_NAME = "tau2_airline"


def _build_rows(tasks, split: str, policy: str) -> list[dict]:
    rows = []
    for idx, task in enumerate(tasks):
        rows.append(
            {
                "data_source": DATA_SOURCE,
                "agent_name": AGENT_NAME,
                "prompt": [{"role": "system", "content": policy}],
                "ability": "airline",
                "reward_model": {"style": "rule", "ground_truth": str(task.id)},
                # No tools_kwargs/interaction_kwargs: tau2 tools are driven inside
                # the agent loop, not via verl's tool registry. RLHFDataset defaults
                # the missing keys to {}. (Empty dicts can't be written to parquet.)
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "task_id": str(task.id),
                },
            }
        )
    return rows


def build(out_dir: str) -> None:
    policy = build_system_prompt()  # airline policy.md (same for every task)
    os.makedirs(out_dir, exist_ok=True)
    splits = {"train": load_train_tasks(), "test": load_eval_tasks()}
    for split, tasks in splits.items():
        rows = _build_rows(tasks, split, policy)
        path = os.path.join(out_dir, f"{split}.parquet")
        datasets.Dataset.from_list(rows).to_parquet(path)
        print(f"{split}: {len(rows)} tasks -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/tau3_airline")
    build(ap.parse_args().out_dir)
