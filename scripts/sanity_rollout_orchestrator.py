#!/usr/bin/env python
"""Stage-0 MAIN ACCEPTANCE: end-to-end sanity rollout WITHOUT verl.

Uses tau2's own Orchestrator to run one airline task with an external-API agent
and user simulator (gpt-5), then scores it with tau2's official evaluator
(per-component, NL judge = gpt-5). Proves the env + user-sim + reward chain is
wired up before touching the verl training stack.

Requires OPENAI_API_KEY (in .env). Run in the tau2verl env:
    python scripts/sanity_rollout_orchestrator.py --task 0
"""

from __future__ import annotations

import argparse
import json
import os

from dotenv import load_dotenv

load_dotenv()  # OPENAI_API_KEY etc. for litellm

from tau2.agent.llm_agent import LLMAgent  # noqa: E402
from tau2.orchestrator.orchestrator import Orchestrator  # noqa: E402

from tau2_airline_verl.data.splits import load_tasks  # noqa: E402
from tau2_airline_verl.reward.evaluate import compute_reward, set_nl_judge_model  # noqa: E402
from tau2_airline_verl.tau2env.factory import apply_initial_state, make_airline_env  # noqa: E402
from tau2_airline_verl.usersim.factory import make_user_simulator  # noqa: E402

DOMAIN = "airline"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=None, help="task id (default: first train task)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--agent-model", default=os.environ.get("TAU2_AGENT_MODEL", "gpt-5"))
    ap.add_argument("--user-model", default=os.environ.get("TAU2_USER_MODEL", "gpt-5"))
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--llm-args", default="{}", help='JSON LLM args, e.g. {"temperature":0}')
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set. Copy .env.example -> .env and fill it in.")

    nl_model = set_nl_judge_model()  # gpt-5 (override tau2 default gpt-4.1)
    llm_args = json.loads(args.llm_args)

    tasks = load_tasks(args.split)
    task = tasks[0] if args.task is None else next(t for t in tasks if t.id == args.task)
    print(f"== Task {task.id} | split={args.split} "
          f"| reward_basis={[r.value for r in task.evaluation_criteria.reward_basis]} ==")
    print(f"agent={args.agent_model}  user={args.user_model}  nl_judge={nl_model}\n")

    env = make_airline_env()
    apply_initial_state(env, task)
    agent = LLMAgent(
        tools=env.get_tools(), domain_policy=env.get_policy(),
        llm=args.agent_model, llm_args=llm_args,
    )
    user = make_user_simulator(task, model=args.user_model, llm_args=llm_args)

    orch = Orchestrator(
        domain=DOMAIN, agent=agent, user=user, environment=env, task=task,
        max_steps=args.max_steps, solo_mode=False,
    )
    sim = orch.run()

    print(f"\n-- transcript ({len(sim.messages)} messages) --")
    for m in sim.messages:
        tc = getattr(m, "tool_calls", None)
        suffix = f"  [tool_calls: {[c.name for c in tc]}]" if tc else ""
        content = (m.content or "")[:300]
        print(f"[{m.role}] {content}{suffix}")

    print(f"\ntermination_reason: {sim.termination_reason}")
    result = compute_reward(sim, task, domain=DOMAIN)
    print("\n== REWARD ==")
    print(f"final reward: {result['reward']}")
    print(f"reward_basis: {result['reward_basis']}")
    print(f"components:   {result['components']}")
    print(f"db={result['db']} communicate={result['communicate']} "
          f"nl_assertion={result['nl_assertion']} action={result['action']}")


if __name__ == "__main__":
    main()
