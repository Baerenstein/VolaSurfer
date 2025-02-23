import pytest
import pandas as pd
from datetime import datetime, timedelta
from data.storage.postgres_store import PostgresStore


@pytest.fixture(scope="module")
def postgres_store():
    """Fixture to create a PostgresStore instance for testing."""
    db_uri = "postgresql://user:password@localhost/test_db"  # Update with your test DB credentials
    store = PostgresStore(db_uri)
    yield store
    store.conn.close()  # Ensure the connection is closed after tests


@pytest.mark.skip
def test_initialize_tables(postgres_store):
    """Test that the tables are initialized correctly."""
    with postgres_store.conn.cursor() as cursor:
        cursor.execute("SELECT to_regclass('options_data');")
        options_data_exists = cursor.fetchone()[0] is not None
        cursor.execute("SELECT to_regclass('underlying_data');")
        underlying_data_exists = cursor.fetchone()[0] is not None

    assert options_data_exists
    assert underlying_data_exists


@pytest.mark.skip
def test_store_options_chain(postgres_store):
    """Test storing options chain data."""
    options_df = pd.DataFrame(
        {
            "base_currency": ["USD"],
            "expiry_date": [datetime.now() + timedelta(days=30)],
            "delta": [0.5],
            "gamma": [0.1],
            "implied_vol": [0.2],
            "last_price": [10.0],
            "moneyness": [1.0],
            "option_type": ["call"],
            "strike": [100],
            "theta": [-0.01],
            "vega": [0.1],
        }
    )

    postgres_store.store_options_chain(options_df, symbol="AAPL")

    # Verify that the data was stored
    stored_data = postgres_store.get_options_chain(symbol="AAPL")
    assert len(stored_data) == 1
    assert stored_data[0].symbol == "AAPL"


@pytest.mark.skip
def test_store_underlying(postgres_store):
    """Test storing underlying asset data."""
    underlying_df = pd.DataFrame({"price": [150.0], "volume": [1000]})

    postgres_store.store_underlying(underlying_df, symbol="AAPL")

    # Verify that the data was stored
    stored_data = postgres_store.get_underlying_data(symbol="AAPL")
    assert len(stored_data) == 1
    assert stored_data["symbol"].iloc[0] == "AAPL"
