from stem_agent.eval.metrics import exact_match, token_f1


def test_exact_match_ignores_case_and_articles() -> None:
    assert exact_match("The Eiffel Tower", "eiffel tower") == 1.0
    assert exact_match("Paris", "London") == 0.0


def test_token_f1_partial_overlap() -> None:
    f1 = token_f1("Barack Hussein Obama", "Barack Obama")
    assert 0.7 < f1 < 0.9


def test_token_f1_zero_on_disjoint() -> None:
    assert token_f1("cat", "dog") == 0.0
