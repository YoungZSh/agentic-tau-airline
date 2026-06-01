#!/usr/bin/env python
"""Stage-0 understanding-notes script (read-only, no API calls).

Dumps: split stats, reward_basis distribution, airline tool list + schemas,
a sample task structure, and the EvaluationType enum. Writes a human-readable
report to notes/airline_understanding.md.

Run in the tau2verl env:
    python scripts/understand_airline.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from tau2.evaluator.evaluator import EvaluationType

from tau2_airline_verl.data.splits import load_tasks, summarize_splits
from tau2_airline_verl.tau2env.factory import airline_tool_schemas, make_airline_env

NOTES = Path(__file__).resolve().parents[1] / "notes" / "airline_understanding.md"


def main() -> None:
    lines: list[str] = ["# Airline understanding notes (stage 0)\n"]

    # 1) splits
    splits = summarize_splits()
    lines.append("## Task splits\n")
    lines.append("```json\n" + json.dumps(splits, indent=2) + "\n```\n")
    print("splits:", splits)

    # 2) reward_basis distribution over all tasks
    tasks = load_tasks(None)
    basis_counter: Counter[str] = Counter()
    nl_tasks, comm_tasks, db_tasks, action_tasks = [], [], [], []
    for t in tasks:
        ec = t.evaluation_criteria
        basis = tuple(rt.value for rt in ec.reward_basis) if ec else ()
        basis_counter[basis] += 1
        if ec:
            names = {rt.value for rt in ec.reward_basis}
            if "NL_ASSERTION" in names:
                nl_tasks.append(t.id)
            if "COMMUNICATE" in names:
                comm_tasks.append(t.id)
            if "DB" in names:
                db_tasks.append(t.id)
            if "ACTION" in names:
                action_tasks.append(t.id)
    lines.append("## reward_basis distribution\n")
    lines.append(f"- total tasks: {len(tasks)}\n")
    for basis, n in basis_counter.most_common():
        lines.append(f"- `{list(basis)}`: {n} tasks\n")
    lines.append(
        f"\nTasks gating on: DB={len(db_tasks)}, COMMUNICATE={len(comm_tasks)}, "
        f"NL_ASSERTION={len(nl_tasks)} (ids={nl_tasks}), ACTION={len(action_tasks)}\n"
    )
    print("reward_basis:", dict(basis_counter), "| NL tasks:", nl_tasks)

    # 3) airline tools + schemas
    env = make_airline_env()
    tools = env.get_tools()
    schemas = airline_tool_schemas(env)
    lines.append("## Airline tools\n")
    lines.append(f"- num tools: {len(tools)}\n")
    for tool, schema in zip(tools, schemas):
        name = getattr(tool, "name", "?")
        lines.append(f"- **{name}**\n")
    lines.append("\n### Tool schemas (OpenAI format)\n")
    lines.append("```json\n" + json.dumps(schemas, indent=2, default=str) + "\n```\n")
    print("num tools:", len(tools), "names:", [getattr(t, "name", "?") for t in tools])

    # 4) sample task structure
    sample = load_tasks("train")[0]
    lines.append("## Sample task (train[0])\n")
    lines.append("```\n" + str(sample) + "\n```\n")

    # 5) EvaluationType enum
    lines.append("## EvaluationType enum\n")
    lines.append("```\n" + "\n".join(f"{e.name} = {e.value}" for e in EvaluationType) + "\n```\n")
    print("EvaluationType:", [e.value for e in EvaluationType])

    NOTES.parent.mkdir(parents=True, exist_ok=True)
    NOTES.write_text("".join(lines))
    print(f"\nWrote notes -> {NOTES}")


if __name__ == "__main__":
    main()
