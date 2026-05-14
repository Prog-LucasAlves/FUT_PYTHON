import pandas as pd

from lay_0x1.lay0x1_core import count_goals_until


def test_count_goals_until_empty():
    assert count_goals_until([], 45) == 0


def test_count_goals_until_basic_integers():
    assert count_goals_until([10, 20, 30, 45, 60, 90], 45) == 4


def test_count_goals_until_basic_strings():
    assert count_goals_until(["10", "20", "30", "45", "60", "90"], 45) == 4


def test_count_goals_until_with_quotes():
    assert count_goals_until(["10'", "20'", "30'", "45'", "60'", "90'"], 45) == 4


def test_count_goals_until_extra_time():
    # 45+2 parses to 47.0, so it's > 45, but <= 47
    assert count_goals_until(["45+2", "90+5"], 45) == 0
    assert count_goals_until(["45+2", "90+5"], 47) == 1
    assert count_goals_until(["45+2", "90+5"], 95) == 2


def test_count_goals_until_invalid_and_missing():
    assert count_goals_until([10, None, "", pd.NA, float("nan"), "invalid", 20], 45) == 2
    # Check bounds
    assert count_goals_until([10, None, "", pd.NA, float("nan"), "invalid", 20], 15) == 1


def test_count_goals_until_all_before_or_after():
    # All before
    assert count_goals_until([10, 15, 20], 45) == 3
    # All after
    assert count_goals_until([50, 60, 70], 45) == 0


def test_count_goals_until_exact_minute():
    # Strict equality <= minute_limit
    assert count_goals_until([45], 45) == 1
    assert count_goals_until([45.5], 45) == 0  # assuming float works this way
    assert count_goals_until([45.0], 45) == 1


def test_count_goals_until_floats():
    assert count_goals_until([10.5, 20.2, 30.0], 25) == 2
