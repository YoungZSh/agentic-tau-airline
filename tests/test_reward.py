"""Reward adapter wiring (deterministic, no API calls).

A simulation that terminated prematurely (not AGENT_STOP/USER_STOP) is scored 0
by tau2's evaluator without touching the env or the NL judge — so we can assert
the adapter's contract offline.
"""

import tau2.evaluator.evaluator_nl_assertions as _nl_mod
from tau2.data_model.simulation import SimulationRun, TerminationReason

from tau2_airline_verl.data.splits import load_tasks
from tau2_airline_verl.reward.evaluate import compute_reward, set_nl_judge_model


def _premature_sim(task_id: str) -> SimulationRun:
    return SimulationRun(
        id="test-sim",
        task_id=task_id,
        start_time="2026-01-01T00:00:00",
        end_time="2026-01-01T00:00:01",
        duration=1.0,
        termination_reason=TerminationReason.MAX_STEPS,
        messages=[],
    )


def test_set_nl_judge_model_overrides_symbol():
    model = set_nl_judge_model("gpt-5")
    assert model == "gpt-5"
    assert _nl_mod.DEFAULT_LLM_NL_ASSERTIONS == "gpt-5"


def test_compute_reward_contract_on_premature_sim():
    task = load_tasks("train")[0]
    sim = _premature_sim(task.id)
    res = compute_reward(sim, task, domain="airline")

    assert res["reward"] == 0.0
    assert 0.0 <= res["reward"] <= 1.0
    # contract: all subscore keys present (None when not evaluated)
    for k in ["db", "communicate", "nl_assertion", "action", "env_assertion"]:
        assert k in res
    assert "components" in res and isinstance(res["components"], dict)
