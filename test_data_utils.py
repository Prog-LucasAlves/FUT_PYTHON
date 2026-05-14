import os

# Adjusting python path to allow importing lay_0x1 modules if needed
import sys
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "lay_0x1")))

from data_utils import _validate_columns


def test_validate_columns_all_present():
    df = pd.DataFrame({"A": [1], "B": [2], "C": [3]})
    with patch("streamlit.sidebar.warning") as mock_warning:
        result = _validate_columns(df, ["A", "B"], "test_source")
        assert result is True
        mock_warning.assert_not_called()


def test_validate_columns_missing_some():
    df = pd.DataFrame({"A": [1]})
    with patch("streamlit.sidebar.warning") as mock_warning:
        result = _validate_columns(df, ["A", "B", "C"], "test_source")
        assert result is False
        mock_warning.assert_called_once_with("test_source: colunas ausentes B, C")


def test_validate_columns_missing_all():
    df = pd.DataFrame({"X": [1]})
    with patch("streamlit.sidebar.warning") as mock_warning:
        result = _validate_columns(df, ["A", "B"], "test_source")
        assert result is False
        mock_warning.assert_called_once_with("test_source: colunas ausentes A, B")


def test_validate_columns_empty_required():
    df = pd.DataFrame({"A": [1]})
    with patch("streamlit.sidebar.warning") as mock_warning:
        result = _validate_columns(df, [], "test_source")
        assert result is True
        mock_warning.assert_not_called()
