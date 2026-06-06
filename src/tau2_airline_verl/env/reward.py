"""Reward adapter — wraps tau2's official evaluator.

Decision (locked):
- Official component basis: each task's own `evaluation_criteria.reward_basis`
  gates the final reward (product of per-component rewards). We call
  `evaluate_simulation(..., EvaluationType.ALL)`, which is exactly that.
- The final scalar reward = `RewardInfo.reward`.
- Per-component subscores are exposed for training curves via
  `RewardInfo.reward_breakdown` (dict[RewardType, float]).
- NL_ASSERTION judge model is overridden from tau2's default (gpt-4.1) to gpt-5.

The NL judge reads `DEFAULT_LLM_NL_ASSERTIONS` from the evaluator module's
namespace at call time, so we patch that symbol in-place.
"""

from __future__ import annotations

import os
from typing import Optional

import tau2.evaluator.evaluator_nl_assertions as _nl_mod
from tau2.data_model.simulation import RewardInfo, SimulationRun
from tau2.data_model.tasks import RewardType, Task
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation

# All gating components tau2 supports; we always surface them (None when not evaluated).
_ALL_COMPONENTS = [
    RewardType.DB,
    RewardType.ENV_ASSERTION,
    RewardType.ACTION,
    RewardType.COMMUNICATE,
    RewardType.NL_ASSERTION,
]


def set_nl_judge_model(model: Optional[str] = None) -> str:
    """Override the LLM used for NL_ASSERTION judging (tau2 default is gpt-4.1).

    Patches the evaluator module's symbols in-place: the model name, and — when the
    judge is the local Qwen3.x server (TAU2_DISABLE_THINKING=1) — injects
    `extra_body.chat_template_kwargs.enable_thinking=false` into the judge's llm_args
    so its verdict isn't buried in a <think> block (which would break parsing).
    Re-enabling thinking would require a vllm reasoning parser, so it's off by default.

    Returns the model now in effect. Call once at startup.
    """
    model = model or os.environ.get("TAU2_NL_JUDGE_MODEL", "gpt-5")
    _nl_mod.DEFAULT_LLM_NL_ASSERTIONS = model
    if os.environ.get("TAU2_DISABLE_THINKING", "1") == "1":
        args = dict(getattr(_nl_mod, "DEFAULT_LLM_NL_ASSERTIONS_ARGS", {}) or {})
        extra = dict(args.get("extra_body") or {})
        ctk = dict(extra.get("chat_template_kwargs") or {})
        ctk["enable_thinking"] = False
        extra["chat_template_kwargs"] = ctk
        args["extra_body"] = extra
        _nl_mod.DEFAULT_LLM_NL_ASSERTIONS_ARGS = args
    return model


def reward_from_simulation(
    simulation: SimulationRun,
    task: Task,
    *,
    domain: str = "airline",
    solo_mode: bool = False,
    evaluation_type: EvaluationType = EvaluationType.ALL,
) -> RewardInfo:
    """Run tau2's official evaluator on a completed simulation. Returns RewardInfo."""
    return evaluate_simulation(
        simulation=simulation,
        task=task,
        evaluation_type=evaluation_type,
        solo_mode=solo_mode,
        domain=domain,
    )


def compute_reward(
    simulation: SimulationRun,
    task: Task,
    *,
    domain: str = "airline",
    solo_mode: bool = False,
) -> dict:
    """Official reward + per-component subscores, in a flat dict for logging/curves.

    Returns:
        {
          "reward": float,                      # final scalar (product over reward_basis)
          "reward_basis": ["DB","COMMUNICATE"], # the task's gating components
          "components": {"DB": 1.0, "COMMUNICATE": 0.0, ...},  # only those evaluated
          "db": float|None, "communicate": float|None,
          "action": float|None, "nl_assertion": float|None, "env_assertion": float|None,
          "raw": RewardInfo,
        }
    """
    info = reward_from_simulation(
        simulation, task, domain=domain, solo_mode=solo_mode
    )
    breakdown = info.reward_breakdown or {}
    components = {rt.value: breakdown[rt] for rt in _ALL_COMPONENTS if rt in breakdown}

    basis = (
        [rt.value for rt in task.evaluation_criteria.reward_basis]
        if task.evaluation_criteria is not None
        else None
    )
    return {
        "reward": float(info.reward),
        "reward_basis": basis,
        "components": components,
        "db": breakdown.get(RewardType.DB),
        "env_assertion": breakdown.get(RewardType.ENV_ASSERTION),
        "action": breakdown.get(RewardType.ACTION),
        "communicate": breakdown.get(RewardType.COMMUNICATE),
        "nl_assertion": breakdown.get(RewardType.NL_ASSERTION),
        "raw": info,
    }
