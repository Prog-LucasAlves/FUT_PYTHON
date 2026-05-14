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
