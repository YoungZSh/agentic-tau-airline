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
from tau2_airline_verl.rollout.conversion import tau2_messages_to_openai
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

        # --- tau2 side: env + user simulator + greeting + first user request ----
        session = Tau2Session(task, system_content=system_content)
        with simple_timer("tool_calls", metrics):
            session.start()  # blocking litellm (gpt-5) -> first user message

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

            if fcalls:
                # ---- tool turn: execute via tau2 env, append obs as mask 0 -----
                assistant_msg = session.record_assistant(content, fcalls)
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
                assistant_msg = session.record_assistant(content, None)
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

        # ---- reward via tau2 official evaluator (replays trajectory) -----------
        reward_score = 0.0
        reward_components: dict[str, Any] = {}
        try:
            with simple_timer("compute_score", metrics):
                reward_info = await self.loop.run_in_executor(
                    None, session.reward, termination_reason
                )
            reward_score = float(reward_info["reward"])
            reward_components = reward_info.get("components", {}) or {}
        except Exception as e:  # noqa: BLE001 — never let a reward bug crash the rollout
            logger.warning("reward computation failed for task %s: %s", task_id, e)

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
                "reward_components": reward_components,
                "termination_reason": termination_reason.value,
                "assistant_turns": assistant_turns,
                "user_turns": user_turns,
            },
        )
