"""Airline environment adapters (tau2-bench -> verl) — plan §17 `env/`.

Three thin adapters over tau2-bench (none modify tau2 source code):
  - airline_tool        : Environment + tools + policy.md
  - airline_interaction : UserSimulator (gpt-5 via litellm)
  - reward              : official tau2 evaluator + per-component subscores
"""

from tau2_airline_verl.env.airline_interaction import make_user_simulator
from tau2_airline_verl.env.airline_tool import (
    airline_policy,
    airline_tool_schemas,
    apply_initial_state,
    make_airline_env,
)
from tau2_airline_verl.env.reward import (
    compute_reward,
    reward_from_simulation,
    set_nl_judge_model,
)

__all__ = [
    "make_airline_env",
    "apply_initial_state",
    "airline_tool_schemas",
    "airline_policy",
    "make_user_simulator",
    "compute_reward",
    "reward_from_simulation",
    "set_nl_judge_model",
]
