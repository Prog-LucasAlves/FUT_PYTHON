
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

