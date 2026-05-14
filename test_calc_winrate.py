import math
import sys
from unittest.mock import MagicMock

# Mock pandas before importing calc_winrate
mock_pd = MagicMock()
sys.modules["pandas"] = mock_pd
mock_pd.NA = "PD_NA"


def side_effect_isna(val):
    if val is None or val == "PD_NA":
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    return False


mock_pd.isna.side_effect = side_effect_isna

# ruff: noqa: E402
from calc_winrate import get_score_at_75, get_team_averages


def test_get_score_at_75_null_cases():
    assert get_score_at_75(None) == 0
    assert get_score_at_75("PD_NA") == 0
    assert get_score_at_75(float("nan")) == 0
    assert get_score_at_75("") == 0
    assert get_score_at_75("[]") == 0
    assert get_score_at_75("  ") == 0


def test_get_score_at_75_valid_string():
    assert get_score_at_75("10, 20, 75") == 3
    assert get_score_at_75("10, 20, 76") == 2
    assert get_score_at_75("80, 90") == 0


def test_get_score_at_75_string_with_brackets_and_quotes():
    assert get_score_at_75("[10, '20', 75]") == 3
    assert get_score_at_75("['10', '80']") == 1


def test_get_score_at_75_list_input():
    assert get_score_at_75([10, 20, 75]) == 3
    assert get_score_at_75(["10", "80"]) == 1


def test_get_score_at_75_stoppage_time():
    assert get_score_at_75("45+2, 75+1") == 2
    assert get_score_at_75("74+5, 76+1") == 1


def test_get_score_at_75_invalid_values():
    assert get_score_at_75("abc, 10, def") == 1
    assert get_score_at_75(123) == 0  # Non string/list input


def test_get_score_at_75_boundaries():
    assert get_score_at_75("74") == 1
    assert get_score_at_75("75") == 1
    assert get_score_at_75("76") == 0


def test_get_team_averages_basic():
    # We are using mock_pd so we need to construct a MagicMock that behaves like the dataframe.

    # Let's mock home_goals
    mock_df = MagicMock()

    mock_home_groupby = MagicMock()
    mock_away_groupby = MagicMock()

    def groupby_side_effect(col):
        if col == "Home":
            return mock_home_groupby
        elif col == "Away":
            return mock_away_groupby

    mock_df.groupby.side_effect = groupby_side_effect

    # We also need to mock the dataframe slices
    mock_home_goals_series = MagicMock()
    mock_home_games_series = MagicMock()
    mock_away_goals_series = MagicMock()
    mock_away_games_series = MagicMock()

    mock_home_groupby.__getitem__.side_effect = lambda key: mock_home_goals_series if key == "Goals_H_FT" else mock_home_games_series
    mock_away_groupby.__getitem__.side_effect = lambda key: mock_away_goals_series if key == "Goals_A_FT" else mock_away_games_series

    mock_home_goals_sum = MagicMock()
    mock_home_games_count = MagicMock()
    mock_away_goals_sum = MagicMock()
    mock_away_games_count = MagicMock()

    mock_home_goals_series.sum.return_value = mock_home_goals_sum
    mock_home_games_series.count.return_value = mock_home_games_count
    mock_away_goals_series.sum.return_value = mock_away_goals_sum
    mock_away_games_series.count.return_value = mock_away_games_count

    mock_total_goals = MagicMock()
    mock_total_games = MagicMock()

    mock_home_goals_sum.add.return_value = mock_total_goals
    mock_home_games_count.add.return_value = mock_total_games

    mock_division_result = MagicMock()
    mock_total_goals.__truediv__.return_value = mock_division_result

    mock_division_result.to_dict.return_value = {"TeamA": 2.0, "TeamB": 1.0}

    result = get_team_averages(mock_df)

    assert result == {"TeamA": 2.0, "TeamB": 1.0}
    mock_df.groupby.assert_any_call("Home")
    mock_df.groupby.assert_any_call("Away")
    mock_home_goals_sum.add.assert_called_once_with(mock_away_goals_sum, fill_value=0)
    mock_home_games_count.add.assert_called_once_with(mock_away_games_count, fill_value=0)
    mock_total_goals.__truediv__.assert_called_once_with(mock_total_games)
    mock_division_result.to_dict.assert_called_once()
