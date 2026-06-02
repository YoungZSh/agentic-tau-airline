"""Base rollout (stage 1 GO/NO-GO) + held-out eval (stage 3) via tau2's official
runner, with the agent LLM on a local vLLM OpenAI-compatible server.

Why tau2's own runner (not the verl AgentLoop): the GO/NO-GO gate just needs
base-model pass@1 and per-task reward variance, and tau2's `run_domain` gives
exactly that with the official reward path (`evaluate_simulation`) — no verl
training stack required. The reward口径 is identical to training, so base vs
trained numbers are directly comparable.

The agent talks to a local vLLM server (Qwen3-8B, optionally + LoRA adapter) via
litellm's openai-compatible path; the user simulator stays on gpt-5.

Usage (a vLLM server must already be serving --model-name; see run_base_rollout.sh):
    python -m tau2_airline_verl.evaluation.run_tau2_eval \
        --split train --num-trials 8 \
        --model-name Qwen3-8B --api-base http://localhost:8000/v1 \
        --out notes/stage1_base_rollout.md
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

from tau2.data_model.simulation import TextRunConfig
from tau2.run import run_domain


def aggregate(results, num_trials: int) -> dict:
    """pass@1 + per-task reward variance buckets for the GO/NO-GO gate."""
    by_task: dict[str, list[float]] = defaultdict(list)
    for sim in results.simulations:
        reward = sim.reward_info.reward if sim.reward_info else 0.0
        by_task[sim.task_id].append(float(reward))

    all_rewards = [r for rs in by_task.values() for r in rs]
    n = len(all_rewards)
    solved = lambda rs: sum(1 for r in rs if r >= 1.0)  # noqa: E731
    informative = sum(1 for rs in by_task.values() if 0 < solved(rs) < len(rs))
    all_solved = sum(1 for rs in by_task.values() if rs and solved(rs) == len(rs))
    all_failed = sum(1 for rs in by_task.values() if solved(rs) == 0)
    return {
        "num_tasks": len(by_task),
        "num_trials": num_trials,
        "num_sims": n,
        "pass@1": (sum(1 for r in all_rewards if r >= 1.0) / n) if n else 0.0,
        "tasks_informative": informative,  # neither all-0 nor all-1 -> GRPO signal
        "tasks_all_solved": all_solved,
        "tasks_all_failed": all_failed,
        "per_task_pass_rate": {
            t: solved(rs) / len(rs) for t, rs in sorted(by_task.items())
        },
    }


def go_no_go_markdown(agg: dict, split: str, model: str) -> str:
    informative = agg["tasks_informative"]
    n_tasks = agg["num_tasks"]
    verdict = "GO (RL-zero viable)" if informative >= max(3, n_tasks // 5) else "NO-GO (base too weak / saturated)"
    lines = [
        f"# Stage 1 base rollout — {split} ({model})",
        "",
        f"- pass@1: **{agg['pass@1']:.3f}** over {agg['num_sims']} sims "
        f"({n_tasks} tasks × {agg['num_trials']} trials)",
        f"- informative tasks (non all-0/all-1): **{informative}/{n_tasks}**",
        f"- all-solved: {agg['tasks_all_solved']} | all-failed: {agg['tasks_all_failed']}",
        "",
        f"**GO/NO-GO: {verdict}** — needs enough tasks with reward variance under "
        f"group_size>=8 to give GRPO a gradient (plan §15).",
        "",
        "## per-task pass rate",
        "```json",
        json.dumps(agg["per_task_pass_rate"], indent=2),
        "```",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test", "base"])
    ap.add_argument("--num-trials", type=int, default=8)
    ap.add_argument("--model-name", default="Qwen3-8B", help="served model name on the vLLM server")
    ap.add_argument("--api-base", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    ap.add_argument("--user-model", default=os.environ.get("TAU2_USER_MODEL", "gpt-5"))
    ap.add_argument("--max-concurrency", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=24)
    ap.add_argument("--task-ids", nargs="*", default=None, help="optional subset of task ids")
    ap.add_argument("--save-to", default=None, help="tau2 results json name (under its sim dir)")
    ap.add_argument("--out", default=None, help="write GO/NO-GO markdown here")
    args = ap.parse_args()

    config = TextRunConfig(
        domain="airline",
        agent="llm_agent",
        llm_agent=f"openai/{args.model_name}",
        llm_args_agent={"api_base": args.api_base, "api_key": args.api_key},
        user="user_simulator",
        llm_user=args.user_model,
        llm_args_user={},  # gpt-5 reasoning: no temperature
        num_trials=args.num_trials,
        task_set_name="airline",
        task_split_name=args.split,
        task_ids=args.task_ids,
        max_concurrency=args.max_concurrency,
        max_steps=args.max_steps,
        save_to=args.save_to,
    )
    results = run_domain(config)
    agg = aggregate(results, args.num_trials)
    print(json.dumps({k: v for k, v in agg.items() if k != "per_task_pass_rate"}, indent=2))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write(go_no_go_markdown(agg, args.split, args.model_name) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
