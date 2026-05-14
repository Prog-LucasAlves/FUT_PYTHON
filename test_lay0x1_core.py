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
