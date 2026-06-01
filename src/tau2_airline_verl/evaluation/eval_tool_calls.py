"""Tool-call accuracy / invalid-tool-call rate (plan §15 stage 3).

Aggregates over trajectories: avg num_tool_calls, invalid_tool_call rate,
wrong-tool / wrong-arguments rate. STAGE 3 — not yet implemented.
"""

from __future__ import annotations

from typing import Any


def tool_call_stats(trajectories: list[dict[str, Any]]) -> dict:
    """Aggregate tool-call metrics across trajectories. STAGE 3 stub."""
    raise NotImplementedError("tool-call metrics implemented in stage 3.")


__all__ = ["tool_call_stats"]
