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
