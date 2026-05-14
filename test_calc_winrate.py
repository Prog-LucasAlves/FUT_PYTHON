from unittest.mock import patch

import pytest

from calc_winrate import get_score_at_75



from calc_winrate import get_score_at_75  # noqa: E402




def side_effect_isna(val):
    import math

    if val is None or val == "PD_NA":
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    return False
 perf-optimize-pandas-filter-10079845275900928112



mock_pd.isna.side_effect = side_effect_isna

remove-unused-render-badge-10106806103153642180
from calc_winrate import get_score_at_75  # noqa: E402

=======
remove-unused-render-callout-15998711541426404829
from calc_winrate import get_score_at_75  # noqa: E402

=======
=======
 code-health-refactor-load-historical-data-8712844606160724658

mock_pd.isna.side_effect = side_effect_isna

from calc_winrate import get_score_at_75  # noqa: E402
=======

@pytest.fixture(autouse=True)
def mock_pandas_isna():
    with patch("pandas.isna", side_effect=side_effect_isna):
        with patch("pandas.NA", "PD_NA"):
            yield
 main

main
main
main

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
