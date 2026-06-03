"""train/held-out split: loadable, expected 40/10 sizes, disjoint, scenario-covered."""

from tau2_airline_verl.data.splits import (
    DEFAULT_EVAL_SPLIT,
    DEFAULT_TRAIN_SPLIT,
    get_splits,
    get_task_categories,
    load_tasks,
    summarize_splits,
)


def test_split_summary_disjoint():
    s = summarize_splits()
    assert s["train_eval_disjoint"] is True
    assert s["overlaps"] == {}
    assert DEFAULT_TRAIN_SPLIT in s["counts"]
    assert DEFAULT_EVAL_SPLIT in s["counts"]


def test_split_sizes_are_40_10():
    s = summarize_splits()
    assert s["counts"]["train"] == 40
    assert s["counts"]["test"] == 10
    assert s["counts"]["base"] == 50


def test_test_split_covers_every_scenario_category():
    """Stratified by functional category -> held-out test must touch each one."""
    cats = get_task_categories()
    splits = get_splits()
    assert len(cats) == 50
    all_categories = set(cats.values())
    test_categories = {cats[task_id] for task_id in splits["test"]}
    missing = all_categories - test_categories
    assert not missing, f"test split misses scenario categories: {missing}"


def test_splits_loadable_and_have_reward_basis():
    train = load_tasks(DEFAULT_TRAIN_SPLIT)
    test = load_tasks(DEFAULT_EVAL_SPLIT)
    assert len(train) > 0 and len(test) > 0
    assert set(t.id for t in train).isdisjoint(t.id for t in test)
    t = train[0]
    assert t.user_scenario is not None
    assert t.evaluation_criteria is not None
    assert len(t.evaluation_criteria.reward_basis) > 0
