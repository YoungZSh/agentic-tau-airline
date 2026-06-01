"""Qwen3 agent system-prompt construction (plan §6.2 / §17 qwen3_prompt.py).

The policy system prompt = tau2 airline `policy.md`. Tool schemas are passed
separately to the chat template via `tools=` (Qwen3 native / Hermes format),
NOT concatenated into the prompt text — keeping the format identical to how
Qwen3 was trained, which is the prerequisite for skipping SFT (RL-zero).
"""

from __future__ import annotations

from typing import Any, Optional

from tau2_airline_verl.env.airline_tool import airline_policy, airline_tool_schemas


def build_system_prompt(policy: Optional[str] = None) -> str:
    """Return the agent system prompt. Defaults to the airline policy.md text."""
    return policy if policy is not None else airline_policy()


def build_chat_inputs(
    messages: list[dict[str, Any]],
    tools: Optional[list[dict]] = None,
) -> tuple[list[dict[str, Any]], list[dict]]:
    """Pair messages with airline tool schemas for `apply_chat_template`.

    Returns (messages, tools); tools defaults to the airline OpenAI-format
    schemas so callers can do `tokenizer.apply_chat_template(messages, tools=tools)`.
    """
    if tools is None:
        tools = airline_tool_schemas()
    return messages, tools


__all__ = ["build_system_prompt", "build_chat_inputs"]
