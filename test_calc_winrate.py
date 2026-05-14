import sys
from unittest.mock import MagicMock

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

from calc_winrate import get_bin, get_score_at_75  # noqa: E402


def test_get_bin_null_cases():
    bins = [1.0, 1.5, 2.0]
    labels = ["Low", "High"]
    assert get_bin(None, bins, labels) is None
    assert get_bin("PD_NA", bins, labels) is None
    assert get_bin(float("nan"), bins, labels) is None


def test_get_bin_within_bins():
    bins = [1.0, 1.3, 1.5, 1.7, 2.0]
    labels = ["<1.3", "1.3-1.5", "1.5-1.7", "1.7-2.0"]
    assert get_bin(1.2, bins, labels) == "<1.3"
    assert get_bin(1.4, bins, labels) == "1.3-1.5"
    assert get_bin(1.6, bins, labels) == "1.5-1.7"
    assert get_bin(1.9, bins, labels) == "1.7-2.0"


def test_get_bin_boundaries():
    bins = [1.0, 1.3, 1.5, 1.7, 2.0]
    labels = ["<1.3", "1.3-1.5", "1.5-1.7", "1.7-2.0"]
    # The condition is bins[i] < val <= bins[i+1]
    # So 1.3 should fall in the first bin, 1.5 in the second, etc.
    assert get_bin(1.3, bins, labels) == "<1.3"
    assert get_bin(1.5, bins, labels) == "1.3-1.5"
    assert get_bin(1.7, bins, labels) == "1.5-1.7"
    assert get_bin(2.0, bins, labels) == "1.7-2.0"


def test_get_bin_out_of_bounds():
    bins = [1.0, 1.3, 1.5, 1.7, 2.0]
    labels = ["<1.3", "1.3-1.5", "1.5-1.7", "1.7-2.0"]
    # Values lower than the first bin should fall back to the last label
    assert get_bin(0.5, bins, labels) == "1.7-2.0"
    assert get_bin(1.0, bins, labels) == "1.7-2.0"  # <= bins[0] is not caught in the loop
    # Values higher than the last bin should also fall back to the last label
    assert get_bin(2.5, bins, labels) == "1.7-2.0"


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
