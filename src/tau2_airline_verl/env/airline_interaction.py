"""User-simulator factory — the single place that constructs tau2's
`UserSimulator`, wired to an external API model (decision: gpt-5) via litellm.

tau2's UserSimulator calls `litellm` under the hood (model string = litellm
model id, e.g. "gpt-5"). The API key is read from the environment
(OPENAI_API_KEY) by litellm; we don't pass it explicitly.
"""

from __future__ import annotations

import os
from typing import Optional

from tau2.data_model.tasks import Task
from tau2.user.user_simulator import UserSimulator

DEFAULT_USER_MODEL = os.environ.get("TAU2_USER_MODEL", "gpt-5")


def make_user_simulator(
    task: Task,
    model: Optional[str] = None,
    llm_args: Optional[dict] = None,
) -> UserSimulator:
    """Build a tau2 UserSimulator for a task's hidden user scenario.

    Args:
        task: tau2 Task; `task.user_scenario.instructions` defines the hidden
            scenario (reason_for_call / known_info / unknown_info / instructions).
        model: litellm model id for the user simulator (default: env TAU2_USER_MODEL / gpt-5).
        llm_args: extra LLM args (e.g. {"temperature": 0.0}); tau2 default temp is 0.0.
    """
    return UserSimulator(
        llm=model or DEFAULT_USER_MODEL,
        instructions=task.user_scenario.instructions,
        llm_args=llm_args,
        persona_config=None,
    )
