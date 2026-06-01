"""Airline environment factory — the single place that constructs tau2's
airline `Environment` and exposes its tools.

A fresh `Environment` owns the FlightDB; each rollout trajectory must get its
own instance so DB writes don't leak across trajectories. `task.initial_state`
is applied via `env.set_state(...)` (mirrors tau2's Orchestrator.initialize).
"""

from __future__ import annotations

from typing import Optional

from tau2.data_model.tasks import Task
from tau2.domains.airline.environment import get_environment
from tau2.environment.environment import Environment


def make_airline_env() -> Environment:
    """Construct a fresh airline Environment (default DB). Airline has no solo mode."""
    return get_environment(solo_mode=False)


def apply_initial_state(env: Environment, task: Task) -> None:
    """Apply a task's initial state to the environment (data + init actions + history)."""
    init = task.initial_state
    env.set_state(
        initialization_data=(init.initialization_data if init else None),
        initialization_actions=(init.initialization_actions if init else None),
        message_history=(init.message_history if (init and init.message_history) else []),
    )


def airline_tool_schemas(env: Optional[Environment] = None) -> list[dict]:
    """Return the airline tools as OpenAI-format function schemas.

    Used to render the policy/agent system prompt and feed verl's chat template
    / tool parser. `Tool.openai_schema` is tau2's OpenAI-compatible schema.
    """
    if env is None:
        env = make_airline_env()
    schemas = []
    for tool in env.get_tools():
        schema = getattr(tool, "openai_schema", None)
        if schema is None and hasattr(tool, "get_openai_schema"):
            schema = tool.get_openai_schema()
        schemas.append(schema)
    return schemas


def airline_policy(env: Optional[Environment] = None) -> str:
    """Return the airline policy.md text (agent system-prompt component)."""
    if env is None:
        env = make_airline_env()
    return env.get_policy()
