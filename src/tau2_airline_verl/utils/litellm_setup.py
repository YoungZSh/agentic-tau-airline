"""Process-level litellm tuning for the external API model (gpt-5 user simulator
and NL judge).

Why this exists: run 20260603_152218 crashed at step 36 when a gpt-5
user-simulator call raised `httpx.PoolTimeout`. Many concurrent rollouts share
litellm's *sync* OpenAI http client, whose connection pool is small, and gpt-5
(a slow reasoning model) holds each connection long enough that the pool drains
— new calls then time out *waiting for a free connection* (not on read). verl's
`asyncio.gather` has no per-rollout tolerance, so one such timeout kills the
whole training step.

litellm's sync OpenAI path adopts `litellm.client_session` as its http client
when set (litellm/llms/openai/common_utils.py `_get_sync_http_client`). We
install an httpx.Client with a much larger pool + explicit timeouts, so transient
congestion *waits* for a connection instead of failing.

All knobs are env-driven (CLAUDE.md: .env is the single source of machine
config). Per-call `timeout` / `num_retries` live in the user simulator's
llm_args (see env/airline_interaction.py); this module only sizes the shared
pool. Both are no-ops unless the corresponding env vars are set, so default
behaviour is unchanged.
"""

from __future__ import annotations

import logging
import os

import httpx
import litellm

logger = logging.getLogger(__name__)

_CONFIGURED = False


def configure_litellm_from_env() -> None:
    """Install a large-pool sync httpx client into litellm.

    Idempotent and cheap: safe to call from every AgentLoop ``__init__`` (it runs
    once per worker process). A no-op unless ``TAU2_LLM_MAX_CONNECTIONS`` is set,
    keeping the default code path untouched.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    max_conn_env = os.environ.get("TAU2_LLM_MAX_CONNECTIONS")
    if not max_conn_env:
        _CONFIGURED = True  # nothing to do; don't re-check every construction
        return

    max_conn = int(max_conn_env)
    max_keepalive = int(os.environ.get("TAU2_LLM_MAX_KEEPALIVE", "64"))
    timeout_s = float(os.environ.get("TAU2_USER_TIMEOUT", "600"))
    connect_s = float(os.environ.get("TAU2_LLM_CONNECT_TIMEOUT", "10"))

    limits = httpx.Limits(
        max_connections=max_conn,
        max_keepalive_connections=max_keepalive,
    )
    # pool=timeout_s: wait up to the full per-attempt budget for a free
    # connection instead of the httpx default that surfaces as PoolTimeout under
    # congestion. connect is kept short so a genuinely dead endpoint fails fast.
    timeout = httpx.Timeout(timeout_s, connect=connect_s, pool=timeout_s)
    litellm.client_session = httpx.Client(limits=limits, timeout=timeout)
    _CONFIGURED = True
    logger.warning(
        "litellm sync pool configured: max_connections=%d max_keepalive=%d "
        "timeout=%.0fs connect=%.0fs",
        max_conn,
        max_keepalive,
        timeout_s,
        connect_s,
    )
