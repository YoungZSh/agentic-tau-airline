"""Conversion helpers between tau2 Message objects and OpenAI chat dicts.

verl's `apply_chat_template` / tool parser operate on OpenAI-style chat dicts
and tool schemas; tau2's Environment / UserSimulator operate on tau2 Message
objects. The custom AgentLoop bridges the two through these helpers.

Stage 0: core conversions implemented; edge cases (multi tool call, audio,
multimodal) left as TODO since airline is text-only single-turn-tool.
"""

from __future__ import annotations

import json
from typing import Any

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    ToolCall,
    ToolMessage,
    UserMessage,
)


def tau2_message_to_openai(msg: Message) -> dict[str, Any]:
    """Convert a single tau2 Message to an OpenAI chat dict."""
    role = msg.role
    if role == "assistant":
        out: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if getattr(msg, "tool_calls", None):
            out["content"] = msg.content or None
            out["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in msg.tool_calls
            ]
        return out
    if role == "user":
        return {"role": "user", "content": msg.content or ""}
    if role == "tool":
        return {
            "role": "tool",
            "content": msg.content or "",
            "tool_call_id": getattr(msg, "id", None),
        }
    if role == "system":
        return {"role": "system", "content": msg.content or ""}
    raise ValueError(f"Unsupported tau2 message role: {role!r}")


def tau2_messages_to_openai(messages: list[Message]) -> list[dict[str, Any]]:
    return [tau2_message_to_openai(m) for m in messages]


def openai_tool_call_to_tau2(name: str, arguments: str | dict, call_id: str = "") -> ToolCall:
    """Build a tau2 ToolCall (requestor=assistant) from a parsed model tool call."""
    if isinstance(arguments, str):
        arguments = json.loads(arguments) if arguments.strip() else {}
    return ToolCall(id=call_id or "call_0", name=name, arguments=arguments, requestor="assistant")


def tool_message_to_text(tool_msg: ToolMessage) -> str:
    """Flatten a tau2 ToolMessage into observation text for the next prompt."""
    return tool_msg.content or ""


# TODO(stage2): builders for the assistant message tau2 expects when the policy
# emits a tool call (AssistantMessage(tool_calls=[...])) and the user-facing
# AssistantMessage(content=...), so tau2's Environment / UserSimulator can be
# driven directly from token-level model output parsed by verl's ToolParser.
__all__ = [
    "tau2_message_to_openai",
    "tau2_messages_to_openai",
    "openai_tool_call_to_tau2",
    "tool_message_to_text",
]
