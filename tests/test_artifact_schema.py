from pathlib import Path

from stem_agent.specialization.artifact import Artifact, load, save


def test_round_trip(tmp_path: Path) -> None:
    a = Artifact(
        query_style_rules=["be specific"],
        source_preferences={"prefer": ["wikipedia.org"], "avoid": []},
        stopping_heuristic="stop when two sources agree",
        typical_hop_count=3,
        few_shots=[{"question": "q", "trajectory": "t", "answer": "a"}],
        system_prompt="sp",
    )
    p = tmp_path / "artifact.json"
    save(a, p)
    b = load(p)
    assert b == a


def test_load_rejects_missing_keys(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text('{"system_prompt": "x"}')
    try:
        load(p)
    except ValueError:
        return
    raise AssertionError("expected ValueError for missing keys")
