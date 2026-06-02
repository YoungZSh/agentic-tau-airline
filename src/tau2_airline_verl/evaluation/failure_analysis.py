"""Automatic failure-reason tagging (plan §18.4 / §17, stage 3).

Tags each failed trajectory with one or more reasons, for failure analysis and
(later) shaped reward. STAGE 3 — not yet implemented.
"""

from __future__ import annotations

from typing import Any

FAILURE_TAGS = [
    "wrong_tool",
    "wrong_arguments",
    "missing_user_info",
    "policy_violation",
    "failed_to_communicate",
    "too_many_turns",
    "user_simulator_confusion",
    "context_truncation",
]


def tag_failure(trajectory: dict[str, Any]) -> list[str]:
    """Heuristic failure tags for one failed trajectory (plan §18.4).

    Successful trajectories (reward >= 1) return []. Tags are derived from the
    per-component reward breakdown, termination reason, and tool-call stats.
    """
    if trajectory.get("reward", 0.0) >= 1.0:
        return []

    tags: list[str] = []
    comps = trajectory.get("reward_components", {}) or {}
    stats = trajectory.get("stats", {}) or {}
    term = trajectory.get("termination_reason")

    db = comps.get("DB")
    comm = comps.get("COMMUNICATE")
    if db is not None and db < 1.0:
        tags.append("wrong_arguments")  # DB end-state mismatch -> wrong action/args
    if comm is not None and comm < 1.0:
        tags.append("failed_to_communicate")
    if stats.get("num_invalid_tool_calls", 0) > 0:
        tags.append("wrong_tool")
    if term == "max_steps":
        tags.append("too_many_turns")
    if not tags:
        tags.append("policy_violation")  # failed but components don't localize it
    return tags


__all__ = ["FAILURE_TAGS", "tag_failure"]
