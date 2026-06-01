"""train/held-out split: loadable, expected sizes, disjoint."""

from tau2_airline_verl.data.splits import (
    DEFAULT_EVAL_SPLIT,
    DEFAULT_TRAIN_SPLIT,
    load_tasks,
    summarize_splits,
)


def test_split_summary_disjoint():
    s = summarize_splits()
    assert s["train_eval_disjoint"] is True
    assert s["overlaps"] == {}
    assert DEFAULT_TRAIN_SPLIT in s["counts"]
    assert DEFAULT_EVAL_SPLIT in s["counts"]


def test_splits_loadable_and_have_reward_basis():
    train = load_tasks(DEFAULT_TRAIN_SPLIT)
    test = load_tasks(DEFAULT_EVAL_SPLIT)
    assert len(train) > 0 and len(test) > 0
    assert set(t.id for t in train).isdisjoint(t.id for t in test)
    t = train[0]
    assert t.user_scenario is not None
    assert t.evaluation_criteria is not None
    assert len(t.evaluation_criteria.reward_basis) > 0
