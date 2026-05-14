improve-testing-data-utils-5532844099891188372
from unittest.mock import MagicMock

from lay_0x1.data_utils import _read_csv_file


def test_read_csv_file_success(mocker):
    # Mock pd.read_csv to return a dummy value
    dummy_df = MagicMock()
    mock_read_csv = mocker.patch("lay_0x1.data_utils.pd.read_csv", return_value=dummy_df)

    file_path = "dummy_path.csv"
    sep = ","

    result = _read_csv_file(file_path, sep=sep)

    # Verify pd.read_csv was called with the correct arguments
    mock_read_csv.assert_called_once_with(file_path, sep=sep)

    # Verify the return value is the dummy DataFrame
    assert result is dummy_df


def test_read_csv_file_exception(mocker):
    # Mock pd.read_csv to raise an Exception
    error_msg = "Test Exception"
    mock_read_csv = mocker.patch("lay_0x1.data_utils.pd.read_csv", side_effect=Exception(error_msg))

    # Mock st.sidebar.warning
    mock_warning = mocker.patch("lay_0x1.data_utils.st.sidebar.warning")

    # Create a mock file_path object with a 'name' attribute
    mock_file_path = MagicMock()
    mock_file_path.name = "dummy_file.csv"

    result = _read_csv_file(mock_file_path, sep=";")

    # Verify pd.read_csv was called
    mock_read_csv.assert_called_once_with(mock_file_path, sep=";")

    # Verify st.sidebar.warning was called with the correct formatted message
    mock_warning.assert_called_once_with(f"Falha ao ler dummy_file.csv: {error_msg}")

    # Verify the return value is None
    assert result is None
=======
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
main
