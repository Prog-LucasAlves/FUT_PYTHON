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
