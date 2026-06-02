"""Aggregate a trajectory JSONL into a metrics summary, and compare Base vs GRPO
(plan §15 stage 3 / §20.3).

Trajectory JSONL (plan §10), one object per line:
    {"task_id", "reward", "reward_components": {"DB","COMMUNICATE"},
     "termination_reason", "stats": {"num_turns","num_tool_calls",
     "num_invalid_tool_calls","num_tokens"}}

Usage:
    python -m tau2_airline_verl.evaluation.report --base base.jsonl --grpo grpo.jsonl
    python -m tau2_airline_verl.evaluation.report --traj run.jsonl   # single run
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

from tau2_airline_verl.evaluation.eval_passk import pass_at_1, pass_hat_k
from tau2_airline_verl.evaluation.eval_tool_calls import tool_call_stats
from tau2_airline_verl.evaluation.failure_analysis import tag_failure


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(trajectories: list[dict], k: int = 8) -> dict:
    rewards = [float(t.get("reward", 0.0)) for t in trajectories]
    by_task: dict[str, list[float]] = defaultdict(list)
    for t in trajectories:
        by_task[str(t.get("task_id"))].append(float(t.get("reward", 0.0)))

    failure_tags: Counter = Counter()
    for t in trajectories:
        for tag in tag_failure(t):
            failure_tags[tag] += 1

    return {
        "num_trajectories": len(trajectories),
        "pass@1": pass_at_1(rewards),
        f"pass^{k}": pass_hat_k(by_task, k),
        "tool_calls": tool_call_stats(trajectories),
        "failure_tags": dict(failure_tags.most_common()),
    }


def compare(base_path: str, grpo_path: str, k: int = 8) -> dict:
    base = summarize(load_jsonl(base_path), k)
    grpo = summarize(load_jsonl(grpo_path), k)
    return {
        "base": base,
        "grpo": grpo,
        "delta": {
            "pass@1": grpo["pass@1"] - base["pass@1"],
            f"pass^{k}": grpo[f"pass^{k}"] - base[f"pass^{k}"],
            "invalid_tool_call_rate": grpo["tool_calls"]["invalid_tool_call_rate"]
            - base["tool_calls"]["invalid_tool_call_rate"],
            "avg_turns": grpo["tool_calls"]["avg_turns"] - base["tool_calls"]["avg_turns"],
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", help="single trajectory JSONL to summarize")
    ap.add_argument("--base", help="base trajectory JSONL")
    ap.add_argument("--grpo", help="grpo trajectory JSONL")
    ap.add_argument("-k", type=int, default=8)
    args = ap.parse_args()

    if args.base and args.grpo:
        print(json.dumps(compare(args.base, args.grpo, args.k), indent=2))
    elif args.traj:
        print(json.dumps(summarize(load_jsonl(args.traj), args.k), indent=2))
    else:
        ap.error("pass --traj, or both --base and --grpo")


if __name__ == "__main__":
    main()
