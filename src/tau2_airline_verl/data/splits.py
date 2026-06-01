"""Airline task split loading (train / held-out).

Thin wrapper over tau2's own split machinery. The airline split file
(`split_tasks.json`) defines: train(30), test(20), base(50=all).

Methodology: we train on `train` and report all metrics on the held-out
`test` split. The two are disjoint (verified in `summarize_splits`).
"""

from __future__ import annotations

from typing import Optional

from tau2.data_model.tasks import Task
from tau2.domains.airline.environment import get_tasks, get_tasks_split

DEFAULT_TRAIN_SPLIT = "train"
DEFAULT_EVAL_SPLIT = "test"


def get_splits() -> dict[str, list[str]]:
    """Return {split_name: [task_id, ...]} as defined by the airline split file."""
    return get_tasks_split()


def load_tasks(split: Optional[str] = None) -> list[Task]:
    """Load airline tasks for a split name. `None` loads all tasks (no filtering)."""
    return get_tasks(task_split_name=split)


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
