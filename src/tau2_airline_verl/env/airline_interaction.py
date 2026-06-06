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


def _user_llm_args_from_env(model: str) -> dict:
    """Per-call timeout / retry budget + thinking control for the user simulator.

    These land in tau2's `generate(**llm_args)` -> litellm `completion(...)`:
    - `num_retries` overrides tau2's `DEFAULT_MAX_RETRIES` (3); set it higher so a
      transient user-sim timeout is retried instead of killing the whole rollout step.
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
    # Thinking control for the local Qwen3.x user-sim, decoupled from the NL judge
    # (reward.py reads TAU2_DISABLE_THINKING). Default: thinking ON — the sim misjudged
    # scenarios (spurious ###OUT-OF-SCOPE###) and mirrored the agent far less when
    # allowed to reason first. Forwards chat_template_kwargs via litellm's extra_body
    # (tau2's generate() passes **llm_args straight to completion()), and REQUIRES the
    # vllm server to run with `--reasoning-parser qwen3` so <think> lands in
    # reasoning_content and `content` (the user reply tau2 reads) stays clean. Gated on
    # a local Qwen endpoint: a gpt-5 endpoint would reject the unknown body field.
    if "qwen" in model.lower():
        disable = os.environ.get("TAU2_USER_DISABLE_THINKING", "0") == "1"
        args["extra_body"] = {"chat_template_kwargs": {"enable_thinking": not disable}}
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
    effective_model = model or DEFAULT_USER_MODEL
    merged = _user_llm_args_from_env(effective_model)
    if llm_args:
        merged.update(llm_args)
    return UserSimulator(
        llm=effective_model,
        instructions=task.user_scenario.instructions,
        llm_args=merged,
        persona_config=None,
    )
