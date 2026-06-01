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
    """Return failure tags for one trajectory (subset of FAILURE_TAGS). STAGE 3 stub."""
    raise NotImplementedError("failure analysis implemented in stage 3.")


__all__ = ["FAILURE_TAGS", "tag_failure"]
