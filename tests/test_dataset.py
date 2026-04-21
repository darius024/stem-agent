from pathlib import Path

import pytest

from stem_agent.eval.dataset import Question, load_sample, make_splits


def _write_sample(tmp_path: Path, n: int) -> Path:
    entries = [
        {"id": f"q{i}", "question": f"question {i}", "answer": f"a{i}", "type": "bridge"}
        for i in range(n)
    ]
    path = tmp_path / "sample.json"
    path.write_text(__import__("json").dumps(entries))
    return path


def test_load_sample_parses_expected_fields(tmp_path: Path) -> None:
    path = _write_sample(tmp_path, 3)
    loaded = load_sample(path)
    assert loaded == [
        Question(id=f"q{i}", question=f"question {i}", answer=f"a{i}", type="bridge")
        for i in range(3)
    ]


def test_splits_are_disjoint_and_sized(tmp_path: Path) -> None:
    questions = load_sample(_write_sample(tmp_path, 50))
    splits = make_splits(questions, scout_size=5, self_check_size=10, seed=0)

    assert len(splits.scout) == 5
    assert len(splits.self_check) == 10
    assert len(splits.test) == 35

    all_ids = [q.id for q in splits.scout + splits.self_check + splits.test]
    assert len(all_ids) == len(set(all_ids)) == 50


def test_splits_are_seed_stable(tmp_path: Path) -> None:
    questions = load_sample(_write_sample(tmp_path, 50))
    a = make_splits(questions, scout_size=5, self_check_size=10, seed=7)
    b = make_splits(questions, scout_size=5, self_check_size=10, seed=7)
    c = make_splits(questions, scout_size=5, self_check_size=10, seed=8)

    assert [q.id for q in a.test] == [q.id for q in b.test]
    assert [q.id for q in a.test] != [q.id for q in c.test]


def test_splits_reject_sizes_that_leave_no_test(tmp_path: Path) -> None:
    questions = load_sample(_write_sample(tmp_path, 10))
    with pytest.raises(ValueError):
        make_splits(questions, scout_size=5, self_check_size=5)
