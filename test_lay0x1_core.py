import pandas as pd
from lay0x1_core import get_h2h_stats


def test_get_h2h_stats_empty_dataframe():
    df_games = pd.DataFrame(columns=["Date", "Norm_Home", "Norm_Away", "Goals_H_FT", "Goals_A_FT"])
    result = get_h2h_stats(df_games, "Team A", "Team B")
    assert result is None


def test_get_h2h_stats_no_match():
    data = {"Date": ["2023-01-01"], "Norm_Home": ["team_c"], "Norm_Away": ["team_d"], "Goals_H_FT": [1], "Goals_A_FT": [1]}
    df_games = pd.DataFrame(data)
    result = get_h2h_stats(df_games, "Team A", "Team B", lambda x: x.lower().replace(" ", "_"))
    assert result is None


def test_get_h2h_stats_valid_stats():
    data = {"Date": ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01", "2023-05-01"], "Norm_Home": ["teama", "teamb", "teama", "teamb", "teama"], "Norm_Away": ["teamb", "teama", "teamb", "teama", "teamb"], "Goals_H_FT": [0, 1, 1, 0, 0], "Goals_A_FT": [1, 0, 0, 0, 0]}
    df_games = pd.DataFrame(data)

    # Matches:
    # 1. 2023-01-01: H=0, A=1 (score_0x1)
    # 2. 2023-02-01: H=1, A=0 (score_1x0)
    # 3. 2023-03-01: H=1, A=0 (score_1x0)
    # 4. 2023-04-01: H=0, A=0 (score_0x0)
    # 5. 2023-05-01: H=0, A=0 (score_0x0)

    result = get_h2h_stats(df_games, "Team A", "Team B")

    assert result is not None, "get_h2h_stats returned None for valid stats"
    assert result["total"] == 5
    assert result["score_0x1"] == 1
    assert result["score_1x0"] == 2
    assert result["score_0x0"] == 2
    assert result["score_0x1_pct"] == 20.0

    expected_top_scores = {"1x0": 40.0, "0x0": 40.0, "0x1": 20.0}
    assert result["top_scores"] == expected_top_scores

    # Verify sorting
    assert result["games"].iloc[0]["Date"] == "2023-05-01"
    assert result["games"].iloc[-1]["Date"] == "2023-01-01"


def test_get_h2h_stats_default_normalizer():
    # testing that it uses normalize_team_name if not provided
    from lay0x1_core import normalize_team_name

    data = {"Date": ["2023-01-01"], "Norm_Home": [normalize_team_name("Team A")], "Norm_Away": [normalize_team_name("Team B")], "Goals_H_FT": [0], "Goals_A_FT": [1]}
    df_games = pd.DataFrame(data)
    result = get_h2h_stats(df_games, "Team A", "Team B")
    assert result is not None, "get_h2h_stats returned None for valid stats with default normalizer"
    assert result["total"] == 1
    assert result["score_0x1"] == 1
