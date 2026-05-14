test/getDataDay-integration-11262913659135874285
import datetime
from unittest.mock import MagicMock, patch

import pandas as pd

import main


@patch("main.requests.get")
@patch("main.datetime")
@patch("main.pd.read_csv")
def test_getDataDay_success(mock_read_csv, mock_datetime, mock_get):
    """Testa o caso de sucesso onde a API retorna status 200 para ambas as datas."""
    mock_date_hoje = datetime.date(2023, 1, 1)
    mock_date_amanha = datetime.date(2023, 1, 2)
    mock_datetime.date.today.return_value = mock_date_hoje
    mock_datetime.timedelta.side_effect = datetime.timedelta

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b"col1,col2\n1,2"
    mock_get.return_value = mock_response

    # Instead of MagicMock, use a real DataFrame mock
    mock_df = pd.DataFrame({"col1": [1], "col2": [2]})
    # Mocking to_csv inside the real DataFrame might be tricky so we can patch the DataFrame directly
    mock_df.to_csv = MagicMock()
    mock_read_csv.return_value = mock_df

    result = main.getDataDay()

    # The function actually doesn't return anything in the success case! Let's check the code:
    # ```
    # if response.status_code == 200:
    #     df = pd.read_csv(io.BytesIO(response.content))
    #     df.to_csv(...)
    # else:
    #     return pd.DataFrame()
    # ```
    # Oh! It returns None implicitly on success.
    assert result is None
    assert mock_get.call_count == 2

    expected_calls = [
        ((f"https://api.futpythontrader.com/api/dados/jogos-do-dia/betfair/{mock_date_hoje}/download/",), {"headers": main.HEADERS}),
        ((f"https://api.futpythontrader.com/api/dados/jogos-do-dia/betfair/{mock_date_amanha}/download/",), {"headers": main.HEADERS}),
    ]
    mock_get.assert_has_calls(expected_calls, any_order=False)

    assert mock_read_csv.call_count == 2
    assert mock_df.to_csv.call_count == 2


@patch("main.requests.get")
@patch("main.datetime")
def test_getDataDay_error(mock_datetime, mock_get):
    """Testa o caso de erro onde a API retorna status diferente de 200."""
    mock_date_hoje = datetime.date(2023, 1, 1)
    mock_datetime.date.today.return_value = mock_date_hoje
    mock_datetime.timedelta.side_effect = datetime.timedelta

    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.text = "Not Found"
    mock_get.return_value = mock_response

    result = main.getDataDay()

    # It returns an empty pandas DataFrame. We can test isinstance now without MagicMock issue.
    assert isinstance(result, pd.DataFrame)
    assert result.empty
    assert mock_get.call_count == 1
=======
test-integration-main-getDataTotalBetfair-8008597018345525812
from unittest.mock import MagicMock, patch

from main import HEADERS, getDataTotalBetfair


@patch("main.pd.read_csv")
@patch("main.requests.get")
def test_getDataTotalBetfair_success(mock_requests_get, mock_read_csv):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b"col1,col2\nval1,val2\n"
    mock_requests_get.return_value = mock_response

    mock_df = MagicMock()
    mock_df.empty = False
    mock_df.__len__.return_value = 1

    # Mocking columns logic
    mock_columns = MagicMock()
    mock_df.columns = mock_columns

    mock_series1 = MagicMock()
    mock_index = MagicMock()
    mock_series2 = MagicMock()

    mock_columns.to_series.return_value = mock_series1
    mock_series1.index = mock_index
    mock_index.to_series.return_value = mock_series2

    mock_read_csv.return_value = mock_df

    df = getDataTotalBetfair()

    # Verify requests.get was called correctly
    mock_requests_get.assert_called_once_with(
        "https://api.futpythontrader.com/api/dados/betfair/download/",
        headers=HEADERS,
    )

    # Verify return dataframe
    assert df is mock_df

    # Verify to_csv calls on the mocked df and mocked series
    mock_df.to_csv.assert_called_once_with("data_total/dados_betfair.csv", index=False, sep=";")
    mock_series2.to_csv.assert_called_once_with("data_total/columns_betfair.csv", header=False, index=False)


@patch("main.requests.get")
@patch("pandas.DataFrame.to_csv")
@patch("pandas.Series.to_csv")
def test_getDataTotalBetfair_failure(mock_series_to_csv, mock_df_to_csv, mock_requests_get):
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.text = "Not Found"
    mock_requests_get.return_value = mock_response

    df = getDataTotalBetfair()

    # Verify requests.get was called
    mock_requests_get.assert_called_once_with(
        "https://api.futpythontrader.com/api/dados/betfair/download/",
        headers=HEADERS,
    )

    # Verify empty dataframe returned
    assert df.empty

    # Verify to_csv was NOT called (using the patch since it doesn't return a mocked read_csv df here)
    mock_df_to_csv.assert_not_called()
    mock_series_to_csv.assert_not_called()
=======
import os
from unittest.mock import MagicMock, patch

import main


def test_getDataTotalfootystats_success(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data_total", exist_ok=True)

    with patch("main.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        # The code uses pd.read_csv without specifying separator, defaults to comma.
        # But writes with sep=";". So let's provide comma separated.
        mock_response.content = b"col1,col2\n1,2"
        mock_get.return_value = mock_response

        df = main.getDataTotalfootystats()

        mock_get.assert_called_once()
        assert not df.empty
        assert len(df) == 1
        assert list(df.columns) == ["col1", "col2"]
        assert os.path.exists(f"data_total/dados_{main.FONTE2}.csv")
        assert os.path.exists(f"data_total/columns_{main.FONTE2}.csv")


def test_getDataTotalfootystats_error():
    with patch("main.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        mock_get.return_value = mock_response

        df = main.getDataTotalfootystats()

        mock_get.assert_called_once()
        assert df.empty
main
main
