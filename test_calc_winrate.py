import sys
from unittest.mock import MagicMock

# Save real pandas
try:
    real_pandas = sys.modules.get("pandas")
except KeyError:
    real_pandas = None

# Mock pandas before importing calc_winrate
mock_pd = MagicMock()
sys.modules["pandas"] = mock_pd
mock_pd.NA = "PD_NA"


def side_effect_isna(val):
    import math

    if val is None or val == "PD_NA":
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    return False


mock_pd.isna.side_effect = side_effect_isna

from calc_winrate import get_score_at_75  # noqa: E402


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


# Restore pandas
if real_pandas is not None:
    sys.modules["pandas"] = real_pandas
else:
    del sys.modules["pandas"]
