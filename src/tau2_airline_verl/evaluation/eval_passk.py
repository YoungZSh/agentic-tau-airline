"""pass@1 / pass^k metrics over held-out rollouts (plan §15 stage 3).

Inputs are the trajectory JSONL produced by rollout (plan §10); each carries a
final `reward` (0/1) from tau2_airline_verl.env.reward.compute_reward.

pass@1     : mean single-attempt success rate.
pass^k     : fraction of tasks solved in ALL k independent attempts (tau2's
             strict reliability metric) — STAGE 3.
"""

from __future__ import annotations


def pass_at_1(rewards: list[float]) -> float:
    """Mean single-attempt success rate over a flat list of trajectory rewards."""
    if not rewards:
        return 0.0
    return sum(1.0 for r in rewards if r >= 1.0) / len(rewards)


def pass_hat_k(rewards_by_task: dict[str, list[float]], k: int) -> float:
    """Strict pass^k: fraction of tasks solved in ALL k attempts (tau2's
    reliability metric). Tasks with fewer than k attempts are skipped."""
    scored = []
    for rewards in rewards_by_task.values():
        if len(rewards) < k:
            continue
        scored.append(1.0 if all(r >= 1.0 for r in rewards[:k]) else 0.0)
    if not scored:
        return 0.0
    return sum(scored) / len(scored)


__all__ = ["pass_at_1", "pass_hat_k"]
