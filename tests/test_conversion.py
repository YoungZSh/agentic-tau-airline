"""tau2 <-> openai message conversion round-trips for the common roles."""

from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage

from tau2_airline_verl.rollout.conversion import (
    openai_tool_call_to_tau2,
    strip_think,
    tau2_message_to_openai,
)


def test_user_and_assistant_text():
    u = tau2_message_to_openai(UserMessage(role="user", content="hi"))
    assert u == {"role": "user", "content": "hi"}
    a = tau2_message_to_openai(AssistantMessage(role="assistant", content="hello"))
    assert a["role"] == "assistant" and a["content"] == "hello"


def test_assistant_tool_call_and_tool_message():
    msg = AssistantMessage(
        role="assistant",
        content=None,
        tool_calls=[ToolCall(id="c1", name="get_reservation_details",
                             arguments={"reservation_id": "ABC123"}, requestor="assistant")],
    )
    out = tau2_message_to_openai(msg)
    assert out["tool_calls"][0]["function"]["name"] == "get_reservation_details"

    tool = tau2_message_to_openai(ToolMessage(id="c1", role="tool", content="{}", requestor="assistant"))
    assert tool["role"] == "tool"


def test_openai_tool_call_to_tau2():
    tc = openai_tool_call_to_tau2("book", '{"x": 1}', call_id="c9")
    assert tc.name == "book" and tc.arguments == {"x": 1} and tc.requestor == "assistant"


def test_strip_think():
    # normal turn: reasoning peeled off, visible reply trimmed
    assert strip_think("<think>plan the lookup</think>\n\nHello, how can I help?") == (
        "Hello, how can I help?"
    )
    # tool-turn preamble before the call survives
    assert strip_think("<think>reason</think>\n\nLet me check that.") == "Let me check that."
    # no think block: returned as-is (trimmed)
    assert strip_think("just a reply") == "just a reply"
    # degenerate turns collapse to "" so the agent loop can detect them
    assert strip_think("<think>thought but no reply</think>") == ""
    assert strip_think("<think>\n</think>") == ""
    assert strip_think("<think>truncated mid-reasoning, never closed") == ""
    assert strip_think("") == ""
    assert strip_think(None) == ""
