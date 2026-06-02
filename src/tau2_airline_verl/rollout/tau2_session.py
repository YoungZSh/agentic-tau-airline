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
    def start(self) -> UserMessage:
        """Generate the first real user request (user reacts to the greeting)."""
        user_msg, self.user_state = self.user.generate_next_message(
            self._greeting, self.user_state
        )
        self.messages.append(user_msg)
        return user_msg

    def respond_user(self, assistant_msg: AssistantMessage) -> tuple[UserMessage, bool]:
        """User simulator responds to an agent natural-language message.

        Returns (user_message, is_stop). `is_stop` is tau2's ###STOP### /
        ###TRANSFER### / ###OUT-OF-SCOPE### signal.
        """
        user_msg, self.user_state = self.user.generate_next_message(
            assistant_msg, self.user_state
        )
        self.messages.append(user_msg)
        return user_msg, self.user.is_stop(user_msg)

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
