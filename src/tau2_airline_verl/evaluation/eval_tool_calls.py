"""Tool-call accuracy / invalid-tool-call rate (plan §15 stage 3).

Aggregates over trajectories: avg num_tool_calls, invalid_tool_call rate,
wrong-tool / wrong-arguments rate. STAGE 3 — not yet implemented.
"""

from __future__ import annotations

from typing import Any


def tool_call_stats(trajectories: list[dict[str, Any]]) -> dict:
    """Aggregate tool-call metrics across trajectories (plan §10 JSONL schema).

    Each trajectory carries a `stats` dict with num_tool_calls /
    num_invalid_tool_calls / num_turns (see rollout export).
    """
    n = len(trajectories) or 1
    tool_calls = [t.get("stats", {}).get("num_tool_calls", 0) for t in trajectories]
    invalid = [t.get("stats", {}).get("num_invalid_tool_calls", 0) for t in trajectories]
    turns = [t.get("stats", {}).get("num_turns", 0) for t in trajectories]
    total_calls = sum(tool_calls)
    return {
        "num_trajectories": len(trajectories),
        "avg_tool_calls": sum(tool_calls) / n,
        "total_tool_calls": total_calls,
        "invalid_tool_call_rate": (sum(invalid) / total_calls) if total_calls else 0.0,
        "avg_turns": sum(turns) / n,
    }


__all__ = ["tool_call_stats"]
