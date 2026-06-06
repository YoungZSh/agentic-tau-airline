"""Unit tests for Tau2Session's degenerate user-sim reply guard (_generate_user).

Bypasses the heavy Environment/UserSimulator construction (object.__new__) and
injects a scripted fake user simulator that mirrors tau2's generate_next_message
contract: it appends the incoming message + its response to state.messages and
returns (response, state). is_stop() detects the tau2 control tokens.
"""

import types

from tau2.data_model.message import AssistantMessage, UserMessage

from tau2_airline_verl.rollout.tau2_session import Tau2Session


class _FakeUser:
    def __init__(self, scripted_contents):
        self._it = iter(scripted_contents)
        self.calls = 0

    def generate_next_message(self, message, state):
        self.calls += 1
        resp = UserMessage(role="user", content=next(self._it))
        state.messages.append(message)  # mirrors tau2: input appended first
        state.messages.append(resp)
        return resp, state

    @staticmethod
    def is_stop(msg):
        c = msg.content or ""
        return any(t in c for t in ("###STOP###", "###OUT-OF-SCOPE###", "###TRANSFER###"))


def _session(scripted, max_retries=3):
    s = object.__new__(Tau2Session)
    s.user = _FakeUser(scripted)
    s.user_state = types.SimpleNamespace(messages=[])
    s.messages = []
    s._user_sim_max_retries = max_retries
    return s


def test_respond_user_retries_past_empty_replies():
    s = _session(["", "   ", "Hi, I need to cancel my flight."])
    msg, is_stop = s.respond_user(AssistantMessage(role="assistant", content="How can I help?"))
    assert msg.content == "Hi, I need to cancel my flight."
    assert is_stop is False
    assert s.user.calls == 3
    # state rolled back between attempts: only the final good attempt remains
    assert [m.content for m in s.user_state.messages] == ["How can I help?", "Hi, I need to cancel my flight."]
    assert s.messages[-1].content == "Hi, I need to cancel my flight."


def test_start_retries_past_turn0_control_token():
    # a control token as the FIRST utterance is illegitimate -> re-rolled
    s = _session(["###OUT-OF-SCOPE###", "Hi, I'm Sophia and I want compensation."])
    s._greeting = AssistantMessage(role="assistant", content="Hi! How can I help you today?")
    msg = s.start()
    assert msg.content == "Hi, I'm Sophia and I want compensation."
    assert s.user.calls == 2


def test_midconv_control_token_is_not_retried():
    # mid-conversation ###STOP### is a legitimate end signal -> honored, no retry
    s = _session(["###STOP###"])
    msg, is_stop = s.respond_user(AssistantMessage(role="assistant", content="Anything else?"))
    assert is_stop is True
    assert s.user.calls == 1


def test_exhausted_empty_retries_signals_stop():
    s = _session(["", "", "", ""], max_retries=3)  # 1 + 3 retries, all empty
    msg, is_stop = s.respond_user(AssistantMessage(role="assistant", content="?"))
    assert is_stop is True            # terminate gracefully instead of crashing
    assert s.user.calls == 4
