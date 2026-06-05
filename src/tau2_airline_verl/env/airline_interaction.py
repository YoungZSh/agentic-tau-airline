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


def _user_llm_args_from_env() -> dict:
    """Per-call timeout / retry budget for the gpt-5 user simulator, from env.

    These land in tau2's `generate(**llm_args)` -> litellm `completion(...)`:
    - `num_retries` overrides tau2's `DEFAULT_MAX_RETRIES` (3); set it higher so a
      transient gpt-5 timeout is retried instead of killing the whole rollout step.
    - `timeout` bounds each attempt (seconds).
    Absent env vars -> tau2/litellm defaults (so default behaviour is unchanged).
    See also utils/litellm_setup.py, which sizes the shared connection pool.
    """
    args: dict = {}
    timeout = os.environ.get("TAU2_USER_TIMEOUT")
    if timeout:
        args["timeout"] = float(timeout)
    num_retries = os.environ.get("TAU2_USER_NUM_RETRIES")
    if num_retries:
        args["num_retries"] = int(num_retries)
    return args


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
            Merged on top of the env-derived timeout/retry budget (caller wins).
    """
    merged = _user_llm_args_from_env()
    if llm_args:
        merged.update(llm_args)
    return UserSimulator(
        llm=model or DEFAULT_USER_MODEL,
        instructions=task.user_scenario.instructions,
        llm_args=merged,
        persona_config=None,
    )
