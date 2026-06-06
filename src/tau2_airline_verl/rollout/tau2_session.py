"""Synchronous tau2 session — owns one airline trajectory's env + user simulator.

The custom verl AgentLoop (`agent_loop.py`) handles the async/token side (server
generation, chat-template tokenization, response_mask). This module is the
*synchronous* tau2 half: it builds a fresh `Environment` + `UserSimulator` per
trajectory, drives tau2's message exchange, and finally builds a `SimulationRun`
for the official evaluator. Keeping it sync makes it trivially unit-testable and
lets the AgentLoop wrap every blocking call (litellm user-sim, tool execution,
evaluation) in `run_in_executor` so concurrent rollouts don't stall the loop.

Trajectory shape mirrors tau2's half-duplex orchestrator
(`tau2/orchestrator/orchestrator.py`):
    [SystemMessage(policy), AssistantMessage("Hi! ..."), UserMessage(first req),
     AssistantMessage(tool_calls=...), ToolMessage(...), AssistantMessage(text), ...]

The assistant tool-call messages MUST carry structured `tool_calls` followed by
their `ToolMessage` results, because `evaluate_simulation`'s DB check replays
them via `Environment.set_state(message_history=...)` into a clean env and
compares `get_db_hash()` (see `tau2/evaluator/evaluator_env.py`).
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Optional

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.data_model.tasks import Task
from tau2.utils.utils import get_now

from tau2_airline_verl.env.airline_interaction import make_user_simulator
from tau2_airline_verl.env.airline_tool import (
    airline_policy,
    airline_tool_schemas,
    apply_initial_state,
    make_airline_env,
)
from tau2_airline_verl.env.reward import compute_reward
from tau2_airline_verl.rollout.conversion import build_tau2_assistant

# tau2's fixed conversation opener (see tau2/orchestrator/orchestrator.py:47).
DEFAULT_FIRST_AGENT_MESSAGE = "Hi! How can I help you today?"


class Tau2Session:
    """Drives one airline trajectory's tau2 state (env + user simulator).

    All methods are synchronous; the AgentLoop offloads the blocking ones
    (`start`, `execute_tools`, `respond_user`, `reward`) to a thread executor.
    """

    def __init__(
        self,
        task: Task,
        *,
        system_content: Optional[str] = None,
        user_model: Optional[str] = None,
        user_llm_args: Optional[dict] = None,
    ):
        self.task = task
        self.env = make_airline_env()
        apply_initial_state(self.env, task)
        self._tool_schemas = airline_tool_schemas(self.env)

        self.user = make_user_simulator(task, model=user_model, llm_args=user_llm_args)
        self.user_state = self.user.get_init_state()
        # Re-rolls for a degenerate user-sim reply (see _generate_user). Thinking-on
        # eliminated the spurious turn-0 control tokens but introduced a rare (~3%)
        # empty reply when the model's reasoning loops and truncates before </think>.
        self._user_sim_max_retries = int(os.environ.get("TAU2_USER_DEGENERATE_RETRIES", "3"))

        # Trajectory: system policy + tau2's fixed agent greeting (both prompt-side,
        # no loss). `system_content` lets the caller pass the exact policy text the
        # verl dataset row used, keeping the tau2 and token views identical.
        policy = system_content if system_content is not None else airline_policy(self.env)
        self._greeting = AssistantMessage(
            role="assistant", content=DEFAULT_FIRST_AGENT_MESSAGE, cost=0.0
        )
        self.messages: list[Message] = [
            SystemMessage(role="system", content=policy),
            self._greeting,
        ]
        self._start_time = get_now()

    # --- schemas -----------------------------------------------------------
    def tool_schemas(self) -> list[dict]:
        return self._tool_schemas

    # --- user simulator turns (blocking: litellm) --------------------------
    def _generate_user(
        self, message: Message, *, first: bool
    ) -> tuple[UserMessage, bool]:
        """Call the user simulator, re-rolling a *degenerate* reply.

        The local Qwen3.x user-sim (thinking on) occasionally glitches in two ways:
        (a) it returns **empty content** — a pathological reasoning loop truncated
        before `</think>`, which would become a contentless UserMessage that crashes
        tau2's `validate_message` on the next turn; (b) **only as the very first
        turn**, it emits a bare control token (###STOP### / ###OUT-OF-SCOPE### /
        ###TRANSFER###), which is never legitimate before any exchange. Both are
        stochastic, so a fresh re-roll almost always recovers. We roll the user-sim
        state back between attempts so a discarded attempt doesn't corrupt its history.

        Returns (user_message, is_stop). On the first turn an exhausted retry budget
        is treated as a real stop; mid-conversation an exhausted empty reply also
        signals stop so the loop ends gracefully instead of crashing downstream.
        """
        snapshot = len(self.user_state.messages)
        user_msg = None
        for _ in range(self._user_sim_max_retries + 1):
            del self.user_state.messages[snapshot:]  # discard a prior bad attempt
            user_msg, self.user_state = self.user.generate_next_message(
                message, self.user_state
            )
            is_stop = self.user.is_stop(user_msg)
            has_text = bool((user_msg.content or "").strip())
            degenerate = (not has_text) or (first and is_stop)
            if not degenerate:
                return user_msg, is_stop
        # Retries exhausted: surface as a stop (empty -> terminate gracefully; a
        # first-turn control token is an honest out-of-scope) so the caller ends here.
        return user_msg, True

    def start(self) -> UserMessage:
        """Generate the first real user request (user reacts to the greeting)."""
        user_msg, _ = self._generate_user(self._greeting, first=True)
        self.messages.append(user_msg)
        return user_msg

    def respond_user(self, assistant_msg: AssistantMessage) -> tuple[UserMessage, bool]:
        """User simulator responds to an agent natural-language message.

        Returns (user_message, is_stop). `is_stop` is tau2's ###STOP### /
        ###TRANSFER### / ###OUT-OF-SCOPE### signal (or an exhausted empty re-roll).
        """
        user_msg, is_stop = self._generate_user(assistant_msg, first=False)
        self.messages.append(user_msg)
        return user_msg, is_stop

    # --- agent message recording -------------------------------------------
    def record_assistant(
        self, content: Optional[str], parsed_tool_calls: Optional[list[Any]] = None
    ) -> AssistantMessage:
        """Append a tau2 AssistantMessage mirroring the policy's generated turn.

        `parsed_tool_calls` are the tool parser's FunctionCall objects (name +
        JSON-string arguments). They become structured tau2 `ToolCall`s so the
        evaluator can replay them.
        """
        msg = build_tau2_assistant(content, parsed_tool_calls)
        self.messages.append(msg)
        return msg

    # --- tool execution (blocking: env) ------------------------------------
    def execute_tools(self, assistant_msg: AssistantMessage) -> list[ToolMessage]:
        """Execute the assistant message's tool calls against the env (DB mutates
        in place) and append the resulting ToolMessages. Returns them in order."""
        results: list[ToolMessage] = []
        for tool_call in assistant_msg.tool_calls or []:
            tool_msg = self.env.get_response(tool_call)
            self.messages.append(tool_msg)
            results.append(tool_msg)
        return results

    # --- finalize + reward (blocking: evaluator replays the trajectory) ----
    def build_simulation(self, termination_reason: TerminationReason) -> SimulationRun:
        end_time = get_now()
        return SimulationRun(
            id=uuid.uuid4().hex,
            task_id=self.task.id,
            start_time=self._start_time,
            end_time=end_time,
            duration=0.0,
            termination_reason=termination_reason,
            messages=self.messages,
            trial=0,
        )

    def reward(self, termination_reason: TerminationReason) -> dict:
        """Build the SimulationRun and run tau2's official evaluator. Returns the
        flat dict from `env.reward.compute_reward` ({reward, components, ...})."""
        simulation = self.build_simulation(termination_reason)
        info = compute_reward(simulation, self.task, domain="airline")
        info["simulation"] = simulation
        return info
