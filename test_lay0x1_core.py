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
