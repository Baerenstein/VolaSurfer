import pytest
from fastapi.testclient import TestClient
from backend.server.app import app
from unittest.mock import patch

client = TestClient(app)


def test_get_vol_surface_history_default_limit():
    response = client.get("/api/v1/vol_surface/history")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) <= 100  # Default limit


def test_get_vol_surface_history_custom_limit():
    response = client.get("/api/v1/vol_surface/history?limit=50")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) <= 50


def test_get_vol_surface_history_invalid_limit():
    response = client.get("/api/v1/vol_surface/history?limit=-1")
    assert response.status_code == 422  # Unprocessable Entity for invalid limit


@patch('backend.server.app.store.get_last_n_surfaces', return_value=[])
def test_get_vol_surface_history_no_data(mock_get_last_n_surfaces):
    # Assuming the database is empty or no surfaces are available
    response = client.get("/api/v1/vol_surface/history")
    assert response.status_code == 404  # Not Found if no data is available
    mock_get_last_n_surfaces.assert_called_once() 