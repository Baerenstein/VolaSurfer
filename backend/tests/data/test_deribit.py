import pytest
import requests
from unittest.mock import Mock, patch
from data.exchanges.deribit import DeribitAPI


@pytest.fixture
def deribit_api():
    return DeribitAPI()


@pytest.fixture
def mock_successful_response():
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"result": [], "success": True}
    return mock_response


@pytest.fixture
def mock_failed_response():
    mock_response = Mock()
    mock_response.status_code = 500
    mock_response.raise_for_status.side_effect = requests.exceptions.RequestException(
        "Test error"
    )
    return mock_response


class TestDeribitAPI:
    def test_test_connection_success(self, deribit_api, mock_successful_response):
        with patch("requests.Session.get", return_value=mock_successful_response):
            assert deribit_api.test_connection() is True

    def test_test_connection_failure(self, deribit_api, mock_failed_response):
        with patch("requests.Session.get", return_value=mock_failed_response):
            assert deribit_api.test_connection() is False

    def test_get_options_success(self, deribit_api):
        mock_options = [
            {"instrument_name": "BTC-24JUN22-30000-C"},
            {"instrument_name": "BTC-24JUN22-35000-P"},
        ]
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": mock_options, "success": True}

        with patch("requests.get", return_value=mock_response):
            result = deribit_api.get_options("BTC")
            assert result == {"options": mock_options}

    def test_get_options_failure(self, deribit_api, mock_failed_response):
        with patch("requests.get", return_value=mock_failed_response):
            result = deribit_api.get_options("BTC")
            assert result is None

    def test_get_option_data_success(self, deribit_api):
        mock_data = {
            "result": {
                "mark_price": 1000.0,
                "mark_iv": 80.0,
                "greeks": {"delta": 0.5, "gamma": 0.001, "vega": 0.2, "theta": -0.1},
            }
        }
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_data

        with patch("requests.get", return_value=mock_response):
            result = deribit_api.get_option_data("BTC-24JUN22-30000-C")
            assert result == {
                "last_price": 1000.0,
                "implied_vol": 80.0,
                "delta": 0.5,
                "gamma": 0.001,
                "vega": 0.2,
                "theta": -0.1,
            }

    def test_get_option_data_failure(self, deribit_api, mock_failed_response):
        with patch("requests.get", return_value=mock_failed_response):
            result = deribit_api.get_option_data("BTC-24JUN22-30000-C")
            assert result is None

    def test_get_last_price_success(self, deribit_api):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": {"mark_price": 50000.0}}

        with patch("requests.get", return_value=mock_response):
            result = deribit_api.get_last_price("BTC")
            assert result == 50000.0

    def test_get_last_price_failure(self, deribit_api, mock_failed_response):
        with patch("requests.get", return_value=mock_failed_response):
            result = deribit_api.get_last_price("BTC")
            assert result is None
