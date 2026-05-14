testing-get-h2h-stats-12003657326519340250
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
=======
testing-normalize-team-name-13824944143605608504
import numpy as np
import pandas as pd

from lay_0x1.lay0x1_core import normalize_team_name


def test_normalize_team_name_nulls():
    assert normalize_team_name(pd.NA) == ""
    assert normalize_team_name(np.nan) == ""
    assert normalize_team_name(None) == ""


def test_normalize_team_name_standard():
    assert normalize_team_name("Manchester United") == "manchesterunited"
    assert normalize_team_name("REAL MADRID") == "realmadrid"
    assert normalize_team_name("Arsenal") == "arsenal"


def test_normalize_team_name_accents():
    assert normalize_team_name("São Paulo") == "saopaulo"
    assert normalize_team_name("Grêmio") == "gremio"
    assert normalize_team_name("1. FC Köln") == "1fckoln"
    assert normalize_team_name("VfL Osnabrück") == "vflosnabruck"
    assert normalize_team_name("Fenerbahçe") == "fenerbahce"


def test_normalize_team_name_punctuation():
    assert normalize_team_name("A.C. Milan") == "acmilan"
    assert normalize_team_name("Paris St-Germain") == "parisstgermain"
    assert normalize_team_name("Boca Juniors / A") == "bocajuniorsa"
    assert normalize_team_name("Team_A_Reserves") == "teamareserves"
    assert normalize_team_name("O'Higgins") == "ohiggins"


def test_normalize_team_name_whitespace():
    assert normalize_team_name("  Chelsea  ") == "chelsea"
    assert normalize_team_name("Inter   Milan") == "intermilan"
    assert normalize_team_name("\tJuventus\n") == "juventus"


def test_normalize_team_name_numeric():
    assert normalize_team_name(1860) == "1860"
    assert normalize_team_name("1860 München") == "1860munchen"
    assert normalize_team_name("Schalke 04") == "schalke04"
=======
test-count-goals-after-12096316838107574234
import pandas as pd
from lay0x1_core import count_goals_after


def test_count_goals_after_empty():
    assert count_goals_after([], 45) == 0


def test_count_goals_after_none_and_invalid():
    # Includes pd.NA which might be evaluated by pd.isna within normalize_goal_minute
    assert count_goals_after([None, "invalid", pd.NA], 45) == 0


def test_count_goals_after_all_before_limit():
    assert count_goals_after([10, 20, 30], 45) == 0
    assert count_goals_after(["10", "45"], 45) == 0
    # 45 is not > 45
    assert count_goals_after([45], 45) == 0


def test_count_goals_after_all_after_limit():
    assert count_goals_after([46, 60, 90], 45) == 3
    assert count_goals_after(["45+1", "90+5"], 45) == 2


def test_count_goals_after_mixed():
    assert count_goals_after([10, 45, 46, 60], 45) == 2
    assert count_goals_after([15, "45+2", "50", "90+3"], 45) == 3


def test_count_goals_after_with_stoppage_time():
    # "45+2" parses to 47.0, which is > 45, but not > 50
    assert count_goals_after(["45+2"], 45) == 1
    assert count_goals_after(["45+2"], 50) == 0
    # "90+5" parses to 95.0, which is > 90
    assert count_goals_after(["90+5"], 90) == 1


def test_count_goals_after_floats():
    assert count_goals_after([45.5, 46.0], 45) == 2
    assert count_goals_after([44.5, 45.0], 45) == 0


def test_count_goals_after_exactly_at_limit():
    # Should not be counted as it's not strictly greater
    assert count_goals_after([60], 60) == 0
=======

import numpy as np
import pandas as pd

from lay_0x1.lay0x1_core import format_minutes


def test_format_minutes_null_and_zero():
    assert format_minutes(pd.NA) == "N/A"
    assert format_minutes(None) == "N/A"
    assert format_minutes(0) == "N/A"
    assert format_minutes(0.0) == "N/A"
    assert format_minutes(float("nan")) == "N/A"
    assert format_minutes(np.nan) == "N/A"


def test_format_minutes_positive():
    assert format_minutes(45) == "45'00\""
    assert format_minutes(45.5) == "45'30\""
    assert format_minutes(0.5) == "0'30\""
    assert format_minutes(90.25) == "90'15\""


def test_format_minutes_negative():
    assert format_minutes(-45) == "-45'00\""
    assert format_minutes(-45.5) == "-45'-30\""
    assert format_minutes(-0.5) == "0'-30\""
=======
import pandas as pd

from lay_0x1.lay0x1_core import normalize_goal_minute


def test_normalize_goal_minute_nulls():
    assert normalize_goal_minute(pd.NA) is None
    assert normalize_goal_minute(None) is None


def test_normalize_goal_minute_numeric():
    assert normalize_goal_minute(45) == 45.0
    assert normalize_goal_minute(45.5) == 45.5


def test_normalize_goal_minute_empty_string():
    assert normalize_goal_minute("") is None
    assert normalize_goal_minute("   ") is None
    assert normalize_goal_minute("'") is None


def test_normalize_goal_minute_valid_strings():
    assert normalize_goal_minute("45") == 45.0
    assert normalize_goal_minute("45'") == 45.0
    assert normalize_goal_minute(" 90 ' ") == 90.0
    assert normalize_goal_minute("45+2") == 47.0
    assert normalize_goal_minute("90+5'") == 95.0
    assert normalize_goal_minute("45.5") == 45.5


def test_normalize_goal_minute_exception_paths():
    # Covers except Exception blocks
    assert normalize_goal_minute("abc") is None
    assert normalize_goal_minute("45+abc") is None
    assert normalize_goal_minute("abc+5") is None
    assert normalize_goal_minute("+") is None
    assert normalize_goal_minute("++") is None
    assert normalize_goal_minute("45+2+3") is None  # split gets 45, 2+3, then float('2+3') throws ValueError

main
main
main
