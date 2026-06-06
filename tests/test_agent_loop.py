"""Offline tests for Tau2AirlineAgentLoop.run() — no GPU, no API.

We bypass AgentLoopBase.__init__ (which needs a server/tokenizer/processor) and
inject fakes for the four things run() touches: apply_chat_template, the LLM
server (server_manager.generate), the tool parser, and the Tau2Session. This
pins down the part that's easy to get wrong: response_mask accumulation
(assistant=1, tool/user=0) and the prompt/response split.
"""

import asyncio
import types

import pytest

import tau2_airline_verl.rollout.agent_loop as al_mod
from tau2_airline_verl.rollout.agent_loop import Tau2AirlineAgentLoop


class _Gen:
    """Fake server_manager.generate output."""

    def __init__(self, token_ids):
        self.token_ids = token_ids
        self.log_probs = None
        self.num_preempted = 0
        self.extra_fields = {}


def _make_loop(
    generate_outputs, parse_outputs, respond_stop=True, reward=1.0,
    respond_exc=False, start_exc=False,
):
    """Build a run()-able Tau2AirlineAgentLoop with everything faked."""
    al = object.__new__(Tau2AirlineAgentLoop)
    al.max_assistant_turns = 8
    al.max_user_turns = 8
    al.max_parallel_calls = 1
    al.max_tool_response_length = 512
    al.response_length = 10_000
    al.loop = asyncio.new_event_loop()
    al.tokenizer = types.SimpleNamespace(
        encode=lambda text, add_special_tokens=False: [0] * 5,
        decode=lambda ids: "x",
        eos_token_id=2,
    )

    # apply_chat_template: every call yields a fixed 3-token chunk (mask 0 region).
    async def _act(messages, tools=None, remove_system_prompt=False):
        return [101, 102, 103]

    al.apply_chat_template = _act

    # server_manager.generate: returns the next scripted assistant chunk (mask 1).
    gen_iter = iter(generate_outputs)

    async def _gen(request_id, prompt_ids, sampling_params):
        return _Gen(next(gen_iter))

    al.server_manager = types.SimpleNamespace(generate=_gen)

    # tool parser: returns the next scripted (content, fcalls).
    parse_iter = iter(parse_outputs)

    async def _extract(token_ids):
        return next(parse_iter)

    al.tool_parser = types.SimpleNamespace(extract_tool_calls=_extract)

    # Fake Tau2Session (monkeypatched into the module + _get_task).
    class _Session:
        def __init__(self, *a, **k):
            self.messages = []  # tau2_messages_to_openai([]) -> []

        def start(self):
            if start_exc:
                raise RuntimeError("user-sim start blew up")
            return types.SimpleNamespace(content="first request")

        def tool_schemas(self):
            return []

        def record_assistant(self, content, fcalls=None):
            return types.SimpleNamespace(content=content, tool_calls=fcalls)

        def execute_tools(self, assistant_msg):
            return [types.SimpleNamespace(content='{"ok": true}', id="call_0")]

        def respond_user(self, assistant_msg):
            if respond_exc:
                raise RuntimeError("user-sim respond blew up")
            return types.SimpleNamespace(content="thanks ###STOP###"), respond_stop

        def reward(self, termination_reason):
            return {"reward": reward, "components": {"DB": 1.0, "COMMUNICATE": 1.0}}

    al_mod.Tau2Session = _Session
    al_mod._get_task = lambda task_id: object()
    return al


def _run(al):
    kwargs = {
        "raw_prompt": [{"role": "system", "content": "POLICY"}],
        "extra_info": {"task_id": "0"},
    }
    return al.loop.run_until_complete(al.run({}, **kwargs))


def _fcall():
    return types.SimpleNamespace(name="get_user_details", arguments='{"user_id": "x"}')


def test_mask_self_consistency_tool_then_user_stop():
    # turn1: assistant emits tool call (2 tok); tool obs appended (3 tok, mask 0)
    # turn2: assistant natural language (3 tok); user replies with STOP -> end
    al = _make_loop(
        generate_outputs=[[10, 11], [20, 21, 22]],
        parse_outputs=[("", [_fcall()]), ("done", [])],
        respond_stop=True,
        reward=1.0,
    )
    out = _run(al)

    # response_mask aligns 1:1 with response_ids
    assert len(out.response_mask) == len(out.response_ids)
    # assistant tokens (2 + 3) are 1; the tool-obs chunk (3) is 0
    assert sum(out.response_mask) == 5
    assert out.response_mask.count(0) == 3
    # assistant chunks come first (mask 1) then tool obs (mask 0) then 2nd assistant
    assert out.response_mask[:2] == [1, 1]
    assert out.response_mask[2:5] == [0, 0, 0]
    assert out.response_mask[5:8] == [1, 1, 1]
    assert out.reward_score == 1.0
    assert out.extra_fields["termination_reason"] == "user_stop"


def test_degenerate_empty_turn_terminates_without_user_sim():
    # A think-only assistant turn (nothing after </think>) and no tool call has no
    # user-visible reply. The loop must NOT hand it to the user simulator (tau2 would
    # reject the empty message and crash the batch) — it ends with agent_stop instead.
    al = _make_loop(
        generate_outputs=[[20, 21]],
        parse_outputs=[("<think>thinking but no reply</think>", [])],
        respond_stop=False,  # if respond_user were called, the loop would continue
        reward=0.0,
    )
    out = _run(al)
    assert out.extra_fields["termination_reason"] == "agent_stop"
    # only the (mask 1) assistant tokens accumulated; no user-obs chunk was appended
    assert out.response_mask == [1, 1]
    assert out.reward_score == 0.0


def test_user_sim_exception_midloop_excludes_rollout():
    # An assistant turn is generated, then respond_user raises mid-trajectory. The
    # exception unwinds _rollout_trajectory (its partial tokens are discarded — fine,
    # the sample is excluded anyway) and run() emits a shape-valid, zero-gradient
    # sample instead of crashing the batch.
    al = _make_loop(
        generate_outputs=[[20, 21, 22]],
        parse_outputs=[("hello", [])],
        respond_exc=True,
        reward=1.0,
    )
    out = _run(al)
    assert out.response_mask == [0]                    # zero-gradient
    assert len(out.response_ids) == 1
    assert out.reward_score == 0.0
    assert out.extra_fields["excluded"] == 1.0
    assert out.extra_fields["termination_reason"] == "unexpected_error"


def test_user_sim_exception_before_generation_excludes_rollout():
    # start() raises before any assistant token -> still a shape-valid, inert sample
    # (non-empty prompt + one masked eos), never a crash and never an empty response.
    al = _make_loop(
        generate_outputs=[[20, 21]],
        parse_outputs=[("done", [])],
        start_exc=True,
    )
    out = _run(al)
    assert out.response_mask == [0]
    assert len(out.response_ids) == 1
    assert len(out.prompt_ids) > 0
    assert out.reward_score == 0.0
    assert out.extra_fields["excluded"] == 1.0


def test_reward_failure_does_not_crash():
    al = _make_loop(
        generate_outputs=[[20, 21]],
        parse_outputs=[("done", [])],
        respond_stop=True,
        reward=0.0,
    )
    out = _run(al)
    assert out.reward_score == 0.0
    assert len(out.response_mask) == len(out.response_ids) == 2
    assert out.response_mask == [1, 1]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
