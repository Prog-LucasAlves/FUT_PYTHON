import pandas as pd
import pytest

from lay_0x1.lay0x1_core import get_goal_interval_stats


def dummy_normalize_team_name(name):
    return name


def dummy_normalize_goal_minute(minute):
    if minute is None or pd.isna(minute):
        return None
    try:
        if isinstance(minute, str) and "+" in minute:
            minute = minute.split("+")[0]
        return int(minute)
    except (ValueError, TypeError):
        return None


def test_get_goal_interval_stats_basic():
    data = {
        "Norm_Home": ["TeamA", "TeamB", "TeamA", "TeamC"],
        "Norm_Away": ["TeamB", "TeamA", "TeamC", "TeamA"],
        "Min_Goals_H": [[10, "45+2"], [], [75], [12, 60]],
        "Min_Goals_A": [[85], [22, 35], [], []],
    }
    df = pd.DataFrame(data)

    stats = get_goal_interval_stats(
        df_games=df,
        home_team="TeamA",
        away_team="TeamB",
        normalize_team_name_fn=dummy_normalize_team_name,
        normalize_goal_minute_fn=dummy_normalize_goal_minute,
    )

    assert "home_attack" in stats
    assert "away_attack" in stats
    assert "home_combined" in stats
    assert "away_combined" in stats

    assert stats["home_sample"] == 4  # TeamA is in all 4 games
    assert stats["away_sample"] == 2  # TeamB is in 2 games

    home_attack = stats["home_attack"]
    assert pytest.approx(home_attack["0-15'"]) == (1 / 4) * 100
    assert pytest.approx(home_attack["15-30'"]) == (1 / 4) * 100
    assert pytest.approx(home_attack["30-45'"]) == (2 / 4) * 100
    assert pytest.approx(home_attack["60-75'"]) == (1 / 4) * 100


def test_get_goal_interval_stats_empty():
    df = pd.DataFrame(columns=["Norm_Home", "Norm_Away", "Min_Goals_H", "Min_Goals_A"])
    stats = get_goal_interval_stats(
        df_games=df,
        home_team="TeamA",
        away_team="TeamB",
        normalize_team_name_fn=dummy_normalize_team_name,
        normalize_goal_minute_fn=dummy_normalize_goal_minute,
    )

    assert stats["home_sample"] == 0
    assert stats["away_sample"] == 0


def test_get_goal_interval_stats_list_vs_string():
    data = {
        "Norm_Home": ["TeamA"],
        "Norm_Away": ["TeamB"],
        "Min_Goals_H": [[10, 20]],
        "Min_Goals_A": [None],
    }
    df = pd.DataFrame(data)

    stats = get_goal_interval_stats(
        df_games=df,
        home_team="TeamA",
        away_team="TeamB",
        normalize_team_name_fn=dummy_normalize_team_name,
        normalize_goal_minute_fn=dummy_normalize_goal_minute,
    )

    assert pytest.approx(stats["home_attack"]["0-15'"]) == 100.0
    assert pytest.approx(stats["home_attack"]["15-30'"]) == 100.0
    assert pytest.approx(stats["home_attack"]["30-45'"]) == 0.0
