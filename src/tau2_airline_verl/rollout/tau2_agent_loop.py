"""Custom verl AgentLoop for tau2 airline (Strategy B) — STAGE 0 SKELETON.

Why a custom AgentLoop (not verl tools+interaction): this verl version has no
`Interaction` abstraction, so the user simulator has nowhere to mount. Here the
AgentLoop owns tau2's Environment + UserSimulator + AirlineTools for the whole
trajectory (DB state persists naturally), the policy turn goes through verl's
`server_manager`, and the reward reuses tau2's official evaluator verbatim.

This file is a SKELETON: it imports + registers cleanly and the run() signature
matches AgentLoopBase, but the body is TODO. The state machine mirrors verl's
`ToolAgentLoop` (verl/experimental/agent_loop/tool_agent_loop.py) for the
response_mask accumulation, which is the easiest thing to get wrong:
    assistant tokens -> mask 1   (policy, counts for loss)
    tool / user tokens -> mask 0 (env/user, masked out)

Implementation is finished in stage 2; see the approved plan §16.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
    register,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("tau2_airline")
class Tau2AirlineAgentLoop(AgentLoopBase):
    """Drives one tau2 airline trajectory through verl's LLM server.

    Wiring (stage 2):
      - dataset row carries `task_id`; run() loads the tau2 Task and builds a
        fresh Environment + UserSimulator per trajectory.
      - policy turn: prompt_ids = apply_chat_template(messages, tools=schemas);
        out = await self.server_manager.generate(...); mask += [1]*len.
      - tool turn: parse tool_calls; execute via tau2 env.make_tool_call
        (sync -> run_in_executor); append observation; mask += [0]*len.
      - user turn (no tool call): tau2 UserSimulator.generate_next_message
        (litellm, async await); append user msg; mask += [0]*len.
      - terminate on user stop / max turns / response_length budget.
      - reward: build SimulationRun -> reward.evaluate.compute_reward (official
        reward_basis); set reward_score; put per-component subscores in
        extra_fields for training curves.
    """

    def __init__(self, *args, tools: Optional[Any] = None, **kwargs):
        super().__init__(*args, **kwargs)
        mt = self.rollout_config.multi_turn
        self.max_user_turns = mt.max_user_turns
        self.max_assistant_turns = mt.max_assistant_turns
        self.response_length = self.rollout_config.response_length
        self.tool_parser_name = mt.format
        # TODO(stage2): self.tool_parser = ToolParser.get_tool_parser(mt.format, self.tokenizer)
        # TODO(stage2): cache airline tool schemas (tau2env.airline_tool_schemas()).

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        # ---- STAGE 0 SKELETON: not yet implemented ----
        # The signature, registration, and output shape are exercised by tests;
        # the live loop is built in stage 2.
        #
        # task = load_airline_tasks(...)[by kwargs["task_id"]]
        # env = make_airline_env(); apply_initial_state(env, task)
        # user = make_user_simulator(task)
        # messages, response_mask, response_logprobs = [...], [], []
        # state machine (see module docstring) accumulating mask 1/0 ...
        # reward = compute_reward(simulation, task)["reward"]
        # return AgentLoopOutput(prompt_ids=..., response_ids=..., response_mask=...,
        #                        reward_score=reward, num_turns=..., metrics=...)
        raise NotImplementedError(
            "Tau2AirlineAgentLoop.run is a stage-0 skeleton; implemented in stage 2."
        )

    @staticmethod
    def _empty_output() -> AgentLoopOutput:
        """Shape reference used by the mask self-consistency test (placeholder)."""
        return AgentLoopOutput(
            prompt_ids=[0],
            response_ids=[0],
            response_mask=[1],
            reward_score=0.0,
            num_turns=0,
            metrics=AgentLoopMetrics(),
        )
