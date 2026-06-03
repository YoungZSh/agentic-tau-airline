"""Airline task split loading (train / held-out).

Custom 40/10 stratified split maintained HERE — not in the read-only tau2
submodule. The split ids live in `airline_split.json` next to this file. They
were produced by stratifying the 50 airline tasks over gpt-5-labelled
*functional categories* (cancellation / booking / flight_change /
baggage_passenger / compensation / insurance / other) so the held-out `test`
split (10 tasks) covers every scenario category at least once; leftover test
seats go to the largest categories. See the `_meta` block inside that JSON.

We deliberately bypass tau2's own `get_tasks_split()` (which reads
`split_tasks.json` inside the pinned, read-only submodule). Instead we load ALL
airline tasks and filter by our local id lists. `train`(40) and `test`(10) are
disjoint; `base` is their union (all 50). Disjointness is checked in
`summarize_splits`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from tau2.data_model.tasks import Task
from tau2.domains.airline.environment import get_tasks

DEFAULT_TRAIN_SPLIT = "train"
DEFAULT_EVAL_SPLIT = "test"

# Local split definition (sibling file). Keys beginning with "_" (e.g. `_meta`,
# `_categories`) are documentation only and are NOT returned as splits.
_SPLIT_FILE = Path(__file__).with_name("airline_split.json")
_SPLIT_KEYS = ("train", "test", "base")


def get_splits() -> dict[str, list[str]]:
    """Return {split_name: [task_id, ...]} from the local stratified split file."""
    data = json.loads(_SPLIT_FILE.read_text())
    return {k: list(data[k]) for k in _SPLIT_KEYS if k in data}


def get_task_categories() -> dict[str, str]:
    """Return {task_id: functional_category} — the gpt-5 labels the split was
    stratified over (cancellation / booking / flight_change / ...). Used to
    assert that `test` covers every scenario category."""
    data = json.loads(_SPLIT_FILE.read_text())
    return dict(data.get("_categories", {}))


def load_tasks(split: Optional[str] = None) -> list[Task]:
    """Load airline tasks for a split name. `None` loads all tasks (no filtering).

    Loads every airline task from tau2, then filters by our local id lists —
    this is what lets us redefine the split without touching the submodule.
    """
    tasks = get_tasks(task_split_name=None)
    if split is None:
        return tasks
    splits = get_splits()
    if split not in splits:
        raise ValueError(f"Invalid split name: {split}. Valid splits are: {list(splits)}")
    ids = set(splits[split])
    return [task for task in tasks if task.id in ids]


def load_train_tasks() -> list[Task]:
    return load_tasks(DEFAULT_TRAIN_SPLIT)


def load_eval_tasks() -> list[Task]:
    return load_tasks(DEFAULT_EVAL_SPLIT)


def summarize_splits() -> dict:
    """Counts + pairwise-disjointness check across splits (excluding the all/base split)."""
    splits = get_splits()
    counts = {name: len(ids) for name, ids in splits.items()}

    # The "base" split is the union of everything; exclude from disjointness check.
    partition_names = [n for n in splits if n != "base"]
    overlaps: dict[str, list[str]] = {}
    for i, a in enumerate(partition_names):
        for b in partition_names[i + 1 :]:
            inter = sorted(set(splits[a]) & set(splits[b]))
            if inter:
                overlaps[f"{a}&{b}"] = inter

    train_eval_disjoint = not (
        set(splits.get(DEFAULT_TRAIN_SPLIT, [])) & set(splits.get(DEFAULT_EVAL_SPLIT, []))
    )
    return {
        "counts": counts,
        "split_names": list(splits.keys()),
        "train_split": DEFAULT_TRAIN_SPLIT,
        "eval_split": DEFAULT_EVAL_SPLIT,
        "train_eval_disjoint": train_eval_disjoint,
        "overlaps": overlaps,
    }
