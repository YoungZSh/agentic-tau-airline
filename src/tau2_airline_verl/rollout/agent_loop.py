"""Custom verl AgentLoop for tau2 airline (Strategy B).

Why a custom AgentLoop (not verl tools+interaction): this verl version has no
`Interaction` abstraction, so the user simulator has nowhere to mount. Here the
AgentLoop owns tau2's Environment + UserSimulator (via `Tau2Session`) for the
whole trajectory (DB state persists naturally), the policy turn goes through
verl's `server_manager`, and the reward reuses tau2's official evaluator.

State machine mirrors verl's `ToolAgentLoop`
(`verl/experimental/agent_loop/tool_agent_loop.py`) for response_mask accumulation
— the one thing that's easy to get wrong:
    policy (assistant) tokens -> mask 1   (counts for loss)
    tool / user tokens        -> mask 0   (env/user, masked out)
The single addition over ToolAgentLoop is the **user-simulator branch**: when the
policy emits natural language (no tool call), tau2's UserSimulator responds
instead of terminating.

Token accumulation is byte-for-byte the ToolAgentLoop pattern: render the initial
[system, greeting, first-user] prompt once, then for each later turn append the
*incremental* tokens via `apply_chat_template(add_messages, remove_system_prompt=True)`.

reward_score returned in AgentLoopOutput is dropped by verl onto rm_scores[-1]
(`agent_loop.py:131-135`), so no custom_reward_function is needed.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
    register,
)
from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.utils.profiler import simple_timer

from tau2.data_model.simulation import TerminationReason

from tau2_airline_verl.data.splits import load_tasks
from tau2_airline_verl.env.reward import set_nl_judge_model
from tau2_airline_verl.rollout.conversion import strip_think, tau2_messages_to_openai
from tau2_airline_verl.rollout.tau2_session import Tau2Session
from tau2_airline_verl.utils.litellm_setup import configure_litellm_from_env

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Process-wide task cache (id -> Task). Loaded once per worker; airline has 50 tasks.
_TASKS_BY_ID: Optional[dict] = None


def _get_task(task_id):
    global _TASKS_BY_ID
    if _TASKS_BY_ID is None:
        _TASKS_BY_ID = {t.id: t for t in load_tasks(None)}
    key = str(task_id)
    if key not in _TASKS_BY_ID:
        raise KeyError(f"airline task_id {key!r} not found (have {len(_TASKS_BY_ID)} tasks)")
    return _TASKS_BY_ID[key]


@register("tau2_airline")
class Tau2AirlineAgentLoop(AgentLoopBase):
    """Drives one tau2 airline trajectory through verl's LLM server + tau2 env/user."""

    def __init__(self, *args, tools: Optional[Any] = None, **kwargs):
        super().__init__(*args, **kwargs)
        mt = self.rollout_config.multi_turn
        self.max_assistant_turns = mt.max_assistant_turns or 12
        self.max_user_turns = mt.max_user_turns or 12
        self.max_parallel_calls = getattr(mt, "max_parallel_calls", 1) or 1
        self.max_tool_response_length = getattr(mt, "max_tool_response_length", 512) or 512
        self.response_length = self.rollout_config.response_length
        self.tool_parser = ToolParser.get_tool_parser(mt.format, self.tokenizer)
        # airline reward is fully deterministic (DB hash + substring); the NL judge
        # never fires, but set the override once so any stray NL task uses gpt-5.
        set_nl_judge_model()
        # Size the shared sync http pool for gpt-5 (user sim) so a transient
        # PoolTimeout under concurrent rollouts can't kill the whole step. No-op
        # unless TAU2_LLM_MAX_CONNECTIONS is set; runs once per worker process.
        configure_litellm_from_env()

    def _truncate_tool_response(self, text: Optional[str]) -> str:
        """Right-truncate a tool observation to max_tool_response_length tokens."""
        if not text:
            return ""
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= self.max_tool_response_length:
            return text
        return self.tokenizer.decode(ids[: self.max_tool_response_length])

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        raw_prompt = list(kwargs["raw_prompt"])
        system_content = next(
            (m.get("content") for m in raw_prompt if m.get("role") == "system"), None
        )
        extra_info = kwargs.get("extra_info") or {}
        task_id = extra_info.get("task_id")
        if task_id is None:
            raise ValueError("tau2_airline AgentLoop requires extra_info.task_id")
        task = _get_task(task_id)

        metrics: dict[str, Any] = {}
        request_id = uuid4().hex

        # Build the whole trajectory in a helper so a failure ANYWHERE in it
        # (user-sim glitch that survives its retries, tool exec, generation) is
        # caught here and EXCLUDES this rollout from the gradient update instead of
        # crashing the whole batch: the response_mask is zeroed below so the policy
        # loss / KL / entropy (all response_mask-gated) get no contribution from it.
        # A few bad rollouts must not kill training; per-rollout retries happen first
        # (Tau2Session._generate_user, litellm num_retries), this is the backstop.
        excluded = False
        session = None
        prompt_ids: list[int] = []
        response_mask: list[int] = []
        response_logprobs: list[float] = []
        assistant_turns = 0
        user_turns = 0
        termination_reason = TerminationReason.MAX_STEPS
        try:
            (
                session,
                prompt_ids,
                response_mask,
                response_logprobs,
                assistant_turns,
                user_turns,
                termination_reason,
            ) = await self._rollout_trajectory(
                task, system_content, sampling_params, request_id, metrics
            )
        except Exception as e:  # noqa: BLE001 — exclude this rollout, never crash the batch
            logger.exception("tau2_airline rollout EXCLUDED (task=%s): %s", task_id, e)
            excluded = True
            termination_reason = TerminationReason.UNEXPECTED_ERROR

        # ---- reward via tau2 official evaluator (replays trajectory) -----------
        # reward_score is the official scalar (DB × COMMUNICATE product over the task's
        # reward_basis). db/comm are the per-component subscores; we surface all three so
        # GDPO can normalize each dimension independently within the group (extra_fields
        # below). Defaults stay 0.0 so a reward exception can never drop a key.
        reward_score = 0.0
        db_score = 0.0
        comm_score = 0.0
        reward_components: dict[str, Any] = {}
        if not excluded:
            try:
                with simple_timer("compute_score", metrics):
                    reward_info = await self.loop.run_in_executor(
                        None, session.reward, termination_reason
                    )
                reward_score = float(reward_info["reward"])
                reward_components = reward_info.get("components", {}) or {}
                db_score = float(reward_info.get("db") or 0.0)
                comm_score = float(reward_info.get("communicate") or 0.0)
            except Exception as e:  # noqa: BLE001 — never let a reward bug crash the rollout
                logger.warning("reward computation failed for task %s: %s", task_id, e)

        # ---- excluded rollout: emit a shape-valid, ZERO-gradient sample --------
        # Zeroing response_mask drops it from the policy/KL/entropy loss (all
        # response_mask-gated); reward stays 0. Guarantee a non-empty prompt + at
        # least one (masked) response token so verl's slicing/batching stay valid
        # even when the failure happened before any assistant turn was generated.
        if excluded:
            if not prompt_ids:
                prompt_ids = await self.apply_chat_template(raw_prompt)
            if not response_mask:
                prompt_ids = prompt_ids + [self.tokenizer.eos_token_id]
                response_mask = [0]
                response_logprobs = []
            else:
                response_mask = [0] * len(response_mask)

        # ---- split prompt / response (ToolAgentLoop:177-190) -------------------
        response_ids = prompt_ids[-len(response_mask):]
        prompt_only_ids = prompt_ids[: len(prompt_ids) - len(response_mask)]

        return AgentLoopOutput(
            prompt_ids=prompt_only_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length]
            if response_logprobs
            else None,
            reward_score=reward_score,
            num_turns=assistant_turns + user_turns + 1,
            metrics=AgentLoopMetrics(**metrics),
            extra_fields={
                # Per-component subscores for GDPO decoupled normalization. verl
                # flattens extra_fields["reward_extra_info"] into non_tensor_batch
                # (agent_loop.py:978-982), where algorithm.gdpo_reward_keys reads them.
                # All three keys MUST appear on every trajectory — verl takes the key
                # set from the FIRST sample only — hence the 0.0 defaults above. db_comm
                # == reward_score (the official DB×COMMUNICATE product); keeping it as a
                # 3rd normalized dimension re-aligns the summed advantage with true task
                # success while db/comm give per-dimension gradient even when the product
                # is all-zero across the group. Harmless under plain GRPO (it ignores them).
                "reward_extra_info": {
                    "db": db_score,
                    "comm": comm_score,
                    "db_comm": reward_score,
                },
                "reward_components": reward_components,
                "termination_reason": termination_reason.value,
                "assistant_turns": assistant_turns,
                "user_turns": user_turns,
                # 1.0 when this rollout hit an unrecoverable error and was dropped
                # from the gradient (response_mask zeroed) — watch its mean to catch
                # a systemic user-sim / infra failure instead of silently absorbing it.
                "excluded": 1.0 if excluded else 0.0,
            },
        )

    async def _rollout_trajectory(
        self, task, system_content, sampling_params, request_id, metrics
    ):
        """Build one trajectory: tau2 session + interleaved policy/user/tool turns.

        Returns (session, prompt_ids, response_mask, response_logprobs,
        assistant_turns, user_turns, termination_reason). Raises on any failure;
        run() catches it to EXCLUDE the rollout from the gradient update rather than
        crash the batch. Token bookkeeping mirrors verl's ToolAgentLoop: assistant
        (policy) tokens are mask 1, tool/user observations mask 0.
        """
        # --- tau2 side: env + user simulator + greeting + first user request ----
        session = Tau2Session(task, system_content=system_content)
        with simple_timer("tool_calls", metrics):
            session.start()  # blocking litellm -> first user message

        # --- verl side: initial prompt = [system, greeting, first-user] ---------
        init_messages = tau2_messages_to_openai(session.messages)
        prompt_ids: list[int] = await self.apply_chat_template(
            init_messages, tools=session.tool_schemas()
        )

        response_mask: list[int] = []
        response_logprobs: list[float] = []
        assistant_turns = 0
        user_turns = 0
        termination_reason = TerminationReason.MAX_STEPS

        while True:
            # ---- policy (assistant) turn: mask 1 -------------------------------
            with simple_timer("generate_sequences", metrics):
                output = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                )
            prompt_ids += output.token_ids
            response_mask += [1] * len(output.token_ids)
            if output.log_probs:
                response_logprobs += output.log_probs
            assistant_turns += 1

            if len(response_mask) >= self.response_length:
                termination_reason = TerminationReason.MAX_STEPS
                break

            content, fcalls = await self.tool_parser.extract_tool_calls(output.token_ids)
            fcalls = fcalls[: self.max_parallel_calls] if fcalls else []
            # The decoded content still carries the policy's <think>...</think>
            # chain-of-thought (the hermes parser only strips <tool_call> tags). That
            # reasoning must stay in the token trajectory — the mask-1 ids appended
            # above already count it toward the loss — but must NOT reach the tau2
            # side: otherwise the user simulator reads the agent's hidden reasoning
            # and the COMMUNICATE / NL_ASSERTION judges score it. Peel it off to the
            # user-visible reply for everything tau2 records or evaluates.
            visible = strip_think(content)

            if fcalls:
                # ---- tool turn: execute via tau2 env, append obs as mask 0 -----
                assistant_msg = session.record_assistant(visible, fcalls)
                with simple_timer("tool_calls", metrics):
                    tool_msgs = await self.loop.run_in_executor(
                        None, session.execute_tools, assistant_msg
                    )
                add_messages = [
                    {
                        "role": "tool",
                        "content": self._truncate_tool_response(tm.content),
                        "tool_call_id": tm.id,
                    }
                    for tm in tool_msgs
                ]
                resp_ids = await self.apply_chat_template(
                    add_messages, remove_system_prompt=True
                )
                if len(response_mask) + len(resp_ids) >= self.response_length:
                    termination_reason = TerminationReason.MAX_STEPS
                    break
                prompt_ids += resp_ids
                response_mask += [0] * len(resp_ids)
                if response_logprobs:
                    response_logprobs += [0.0] * len(resp_ids)
                if assistant_turns >= self.max_assistant_turns:
                    termination_reason = TerminationReason.MAX_STEPS
                    break
            else:
                # ---- user-simulator turn: tau2 user responds, append as mask 0 -
                # Degenerate turn: no tool calls and no user-visible reply — an empty
                # completion, or think-only with nothing after </think>. Either way
                # build_tau2_assistant("") yields an AssistantMessage with neither
                # content nor tool_calls, which tau2's UserSimulator rejects via
                # validate_message — and that AssertionError would crash the entire
                # rollout batch (one bad trajectory kills the whole step). End this
                # trajectory gracefully instead; the tokens already count toward the
                # loss (mask 1) and the evaluator scores the incomplete run, so a low
                # reward correctly penalizes the empty turn.
                if not visible:
                    termination_reason = TerminationReason.AGENT_STOP
                    break
                assistant_msg = session.record_assistant(visible, None)
                with simple_timer("tool_calls", metrics):
                    user_msg, is_stop = await self.loop.run_in_executor(
                        None, session.respond_user, assistant_msg
                    )
                user_turns += 1
                if is_stop:
                    termination_reason = TerminationReason.USER_STOP
                    break
                add_messages = [{"role": "user", "content": user_msg.content or ""}]
                resp_ids = await self.apply_chat_template(
                    add_messages, remove_system_prompt=True
                )
                if len(response_mask) + len(resp_ids) >= self.response_length:
                    termination_reason = TerminationReason.MAX_STEPS
                    break
                prompt_ids += resp_ids
                response_mask += [0] * len(resp_ids)
                if response_logprobs:
                    response_logprobs += [0.0] * len(resp_ids)
                if user_turns >= self.max_user_turns:
                    termination_reason = TerminationReason.MAX_STEPS
                    break

        return (
            session,
            prompt_ids,
            response_mask,
            response_logprobs,
            assistant_turns,
            user_turns,
            termination_reason,
        )
