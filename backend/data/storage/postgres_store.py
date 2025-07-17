from datetime import datetime
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import psycopg2
from psycopg2 import extras
from typing import List, Optional
import warnings

from data.storage.base_store import BaseStore
from data.utils.data_schemas import OptionContract, VolSurface, VolMetrics

warnings.filterwarnings(
    "ignore", message=".*pandas only supports SQLAlchemy connectable.*"
)
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")


class PostgresStore(BaseStore):
    def __init__(self, db_uri: str):
        """Initialize PostgreSQL connection and create tables"""
        self.db_uri = db_uri
        self.conn = psycopg2.connect(db_uri)
        self.engine = create_engine(db_uri)
        self.initialize_tables()

    def _limit_decimals(self, value: float, decimals: int = 4) -> float:
        """Helper function to limit decimal places"""
        if value is None or np.isnan(value):
            return None
        return round(float(value), decimals)

    def initialize_tables(self):
        """Initialize required tables in PostgreSQL"""
        self.initialize_assets_table()
        self.initialize_options_data_table()
        self.initialize_underlying_data_table()
        self.initialize_volatility_metrics_table()
        self.initialize_surfaces_table()
        self.initialize_surface_points_table()
        self.initialize_models_table()
        self.initialize_model_parameters_table()
        self.initialize_model_surfaces_table()
        self.initialize_model_surface_points_table()

    def initialize_assets_table(self):
        """Create assets table"""
        with self.conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS assets (
                    id SERIAL PRIMARY KEY,
                    asset_type VARCHAR NOT NULL,
                    ticker VARCHAR NOT NULL UNIQUE
                )
            """)

    def initialize_options_data_table(self):
        """Create options_data table"""
        with self.conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS options_data (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP,
                    asset_id INTEGER REFERENCES assets(id) ON DELETE CASCADE,
                    symbol VARCHAR,
                    strike DOUBLE PRECISION,
                    moneyness DOUBLE PRECISION,
                    option_type VARCHAR,
                    expiry_date DATE,
                    days_to_expiry DOUBLE PRECISION,
                    last_price DOUBLE PRECISION,
                    implied_vol DOUBLE PRECISION,
                    delta DOUBLE PRECISION,
                    gamma DOUBLE PRECISION,
                    vega DOUBLE PRECISION,
                    theta DOUBLE PRECISION,
                    base_currency VARCHAR,
                    snapshot_id VARCHAR
                )
            """)

    def initialize_underlying_data_table(self):
        """Create underlying_data table"""
        with self.conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS underlying_data (
                    id SERIAL PRIMARY KEY,
                    asset_id INTEGER REFERENCES assets(id) ON DELETE CASCADE,
                    symbol VARCHAR,
                    price DOUBLE PRECISION,
                    timestamp TIMESTAMP
                )
            """)

    def initialize_volatility_metrics_table(self):
        """Create volatility_metrics table"""
        with self.conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS volatility_metrics (
                    id SERIAL PRIMARY KEY,
                    asset_id INTEGER REFERENCES assets(id) ON DELETE CASCADE,
                    timestamp TIMESTAMP,
                    implied_vol_index DOUBLE PRECISION,
                    historical_vols JSONB,
                    vol_spread DOUBLE PRECISION,
                    sample_size INTEGER,
                    calculation_method VARCHAR,
                    metadata JSONB
                )
            """)

    def initialize_surfaces_table(self):
        """Create surfaces table"""
        with self.conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS surfaces (
                    id SERIAL PRIMARY KEY,
                    asset_id INTEGER REFERENCES assets(id) ON DELETE CASCADE,
                    timestamp TIMESTAMP,
                    method TEXT NOT NULL,
                    snapshot_id VARCHAR,
                    source_type VARCHAR
                );
                CREATE INDEX IF NOT EXISTS idx_surfaces_timestamp ON surfaces(timestamp);
            """)

    def initialize_surface_points_table(self):
        """Create surface_points table"""
        with self.conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS surface_points (
                    id SERIAL PRIMARY KEY,
                    surface_id INTEGER REFERENCES surfaces(id) ON DELETE CASCADE,
                    strike NUMERIC NOT NULL,
                    moneyness NUMERIC NOT NULL,
                    maturity TIMESTAMP NOT NULL,
                    days_to_expiry NUMERIC NOT NULL,
                    implied_vol NUMERIC NOT NULL,
                    option_type VARCHAR NOT NULL
                )
            """)

    def initialize_models_table(self):
        """Create models table"""
        with self.conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR,
                    description TEXT,
                    parameters_schema JSONB
                )
            """)

    def initialize_model_parameters_table(self):
        """Create model_parameters table"""
        with self.conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_parameters (
                    id SERIAL PRIMARY KEY,
                    model_id INTEGER REFERENCES models(id) ON DELETE CASCADE,
                    asset_id INTEGER REFERENCES assets(id) ON DELETE CASCADE,
                    parameters JSONB,
                    timestamp TIMESTAMP,
                    calibration_data JSONB
                )
            """)

    def initialize_model_surfaces_table(self):
        """Create model_surfaces table"""
        with self.conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_surfaces (
                    id SERIAL PRIMARY KEY,
                    model_id INTEGER REFERENCES models(id) ON DELETE CASCADE,
                    asset_id INTEGER REFERENCES assets(id) ON DELETE CASCADE,
                    timestamp TIMESTAMP,
                    parameters_id INTEGER REFERENCES model_parameters(id) ON DELETE CASCADE,
                    source_surface_id INTEGER REFERENCES surfaces(id) ON DELETE CASCADE
                )
            """)

    def initialize_model_surface_points_table(self):
        """Create model_surface_points table"""
        with self.conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_surface_points (
                    id SERIAL PRIMARY KEY,
                    model_surface_id INTEGER REFERENCES model_surfaces(id) ON DELETE CASCADE,
                    strike NUMERIC NOT NULL,
                    moneyness NUMERIC NOT NULL,
                    maturity TIMESTAMP NOT NULL,
                    days_to_expiry NUMERIC NOT NULL,
                    implied_vol NUMERIC NOT NULL,
                    option_type VARCHAR NOT NULL
                )
            """)

    def store_underlying(self, last_price: float, asset_type: str, symbol: str):
        """Store underlying asset data"""
        asset_id = self.get_or_create_asset(asset_type, symbol)
        underlying_df = pd.DataFrame(
            {"asset_id": [asset_id], 'symbol': [symbol], "price": [last_price], "timestamp": [datetime.now()]}
        )
        with self.conn.cursor() as cursor:
            for _, row in underlying_df.iterrows():
                cursor.execute(
                    """
                    INSERT INTO underlying_data (asset_id, symbol, price, timestamp)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (row["asset_id"], row["symbol"], row["price"], row["timestamp"]),
                )
        self.conn.commit()

    def get_underlying_data(self, symbol: str) -> pd.DataFrame:
        """Retrieve the latest underlying data for a symbol"""
        query = "SELECT * FROM underlying_data WHERE symbol = %s"
        df = pd.read_sql_query(query, self.conn, params=(symbol,))
        return df

    def store_options_chain(self, options_df: pd.DataFrame):
        """Store options chain data with limited decimal places"""
        for idx, row in options_df.iterrows():

            with self.conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO options_data (
                        timestamp, asset_id, symbol, option_type, base_currency,
                        expiry_date, days_to_expiry, strike, moneyness,
                        last_price, implied_vol, delta, gamma, vega, theta,
                        snapshot_id
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        row["timestamp"],
                        row["asset_id"],
                        row["symbol"],
                        row["option_type"],
                        row["base_currency"],
                        row["expiry_date"],
                        self._limit_decimals(row["days_to_expiry"]),
                        self._limit_decimals(row["strike"]),
                        self._limit_decimals(row["moneyness"]),
                        self._limit_decimals(row["last_price"]),
                        self._limit_decimals(row["implied_vol"]),
                        self._limit_decimals(row.get("delta")),
                        self._limit_decimals(row.get("gamma")),
                        self._limit_decimals(row.get("vega")),
                        self._limit_decimals(row.get("theta")),
                        row["snapshot_id"],
                    )
                )
        self.conn.commit()

    def get_options_chain(self, symbol: str) -> List[OptionContract]:
        """Retrieve options chain using SQLAlchemy"""
        query = text("SELECT * FROM options_data WHERE symbol = :symbol")
        df = pd.read_sql_query(query, self.engine, params={"symbol": symbol})

        contracts = [
            OptionContract(
                symbol=row["symbol"],
                expiry_date=row["expiry_date"],
                days_to_expiry=self._limit_decimals(row["days_to_expiry"]),
                strike=self._limit_decimals(row["strike"]),
                moneyness=self._limit_decimals(row["moneyness"]),
                option_type=row["option_type"],
                base_currency=row["base_currency"],
                last_price=self._limit_decimals(row["last_price"]),
                implied_vol=self._limit_decimals(row["implied_vol"]),
                delta=self._limit_decimals(row["delta"]),
                gamma=self._limit_decimals(row["gamma"]),
                vega=self._limit_decimals(row["vega"]),
                theta=self._limit_decimals(row["theta"]),
                snapshot_id=row["snapshot_id"],
            )
            for _, row in df.iterrows()
        ]
        return contracts

    def store_surface(self, vol_surface: VolSurface) -> int:
        """Store for surfaces"""
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO surfaces (timestamp, method, snapshot_id, asset_id)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        vol_surface.timestamp,
                        vol_surface.method,
                        vol_surface.snapshot_id,
                        vol_surface.asset_id,
                    ),
                )

                surface_id = cur.fetchone()[0]

                points_data = [
                    (
                        surface_id,
                        self._limit_decimals(strike),
                        self._limit_decimals(moneyness),
                        maturity,
                        self._limit_decimals(dte),
                        self._limit_decimals(impl_vol),
                        option_type,
                    )
                    for strike, moneyness, maturity, dte, impl_vol, option_type in zip(
                        vol_surface.strikes,
                        vol_surface.moneyness,
                        vol_surface.maturities,
                        vol_surface.days_to_expiry,
                        vol_surface.implied_vols,
                        vol_surface.option_type,
                    )
                ]

                extras.execute_values(
                    cur,
                    """
                    INSERT INTO surface_points 
                        (surface_id, strike, moneyness, maturity, days_to_expiry, implied_vol, option_type)
                    VALUES %s
                    """,
                    points_data,
                    page_size=100,
                )

                self.conn.commit()
                return surface_id

        except Exception as e:
            self.conn.rollback()
            raise RuntimeError(f"Failed to store surface: {str(e)}")

    def get_vol_surfaces(
        self, timestamp: datetime, snapshot_id: Optional[str] = None
    ) -> VolSurface:
        """
        Retrieve a VolSurface object from the surfaces table.

        Args:
            timestamp: Timestamp to retrieve
            snapshot_id: Optional snapshot ID

        Returns:
            VolSurface: Retrieved volatility surface
        """
        with self.conn.cursor() as cur:
            query = """
                SELECT sp.strike, sp.moneyness, sp.maturity, sp.days_to_expiry, 
                    sp.implied_vol, sp.option_type, s.method
                FROM surfaces s
                JOIN surface_points sp ON s.id = sp.surface_id
                WHERE s.timestamp = %s
            """
            params = [timestamp]

            if snapshot_id:
                query += " AND s.snapshot_id = %s"
                params.append(snapshot_id)

            cur.execute(query, params)
            rows = cur.fetchall()

            if not rows:
                return None

            strikes = [row[0] for row in rows]
            moneyness = [row[1] for row in rows]
            maturities = [row[2] for row in rows]
            days_to_expiry = [row[3] for row in rows]
            implied_vols = [row[4] for row in rows]
            option_type = [row[5] for row in rows]
            method = rows[0][6]
            return VolSurface(
                timestamp=timestamp,
                method=method,
                strikes=strikes,
                moneyness=moneyness,
                maturities=maturities,
                days_to_expiry=days_to_expiry,
                implied_vols=implied_vols,
                option_type=option_type,
                snapshot_id=snapshot_id,
            )

    def get_latest_vol_surface(self):
        """
        Retrieve the latest volatility surface data for calls, including timestamp for logging.
        Returns:
            dict: Surface data with timestamp (as ISO string), moneyness, days_to_expiry, and implied_vols
        """
        with self.conn.cursor() as cur:
            query = """
            WITH latest_surface AS (
                SELECT id, timestamp
                FROM surfaces
                ORDER BY timestamp DESC
                LIMIT 1
            )
            SELECT 
                ls.timestamp,
                vsp.moneyness::float,
                vsp.days_to_expiry::float,
                vsp.implied_vol::float
            FROM surface_points vsp
            INNER JOIN latest_surface ls ON vsp.surface_id = ls.id
            WHERE vsp.option_type = 'c'
            ORDER BY vsp.days_to_expiry, vsp.moneyness;
            """
            
            cur.execute(query)
            rows = cur.fetchall()

            if not rows:
                return None

            # First row's timestamp will be the same for all rows
            timestamp = rows[0][0]
            moneyness = [row[1] for row in rows]
            days_to_expiry = [row[2] for row in rows]
            implied_vols = [row[3] for row in rows]

            return {
                "timestamp": timestamp.isoformat(),
                "moneyness": moneyness,
                "days_to_expiry": days_to_expiry,
                "implied_vols": implied_vols
            }

    def get_last_n_surfaces(self, limit: int = 100) -> List[VolSurface]:
        """
        Retrieve the last N volatility surfaces ordered by timestamp descending.

        Args:
            limit: Number of surfaces to retrieve

        Returns:
            List of VolSurface objects
        """
        query = """
        SELECT s.id, s.timestamp, s.method, s.snapshot_id, sp.strike, sp.moneyness, sp.maturity, 
               sp.days_to_expiry, sp.implied_vol, sp.option_type
        FROM surfaces s
        JOIN surface_points sp ON s.id = sp.surface_id
        ORDER BY s.timestamp DESC
        LIMIT %s
        """
        with self.conn.cursor() as cur:
            cur.execute(query, (limit,))
            rows = cur.fetchall()

        surfaces = []
        current_surface_id = None
        current_surface = None

        for row in rows:
            surface_id, timestamp, method, snapshot_id, strike, moneyness, maturity, dte, implied_vol, option_type = row

            if surface_id != current_surface_id:
                if current_surface:
                    surfaces.append(current_surface)

                current_surface = VolSurface(
                    timestamp=timestamp,
                    method=method,
                    snapshot_id=snapshot_id,
                    strikes=[],
                    moneyness=[],
                    maturities=[],
                    days_to_expiry=[],
                    implied_vols=[],
                    option_type=[]
                )
                current_surface_id = surface_id

            current_surface.strikes.append(strike)
            current_surface.moneyness.append(moneyness)
            current_surface.maturities.append(maturity)
            current_surface.days_to_expiry.append(dte)
            current_surface.implied_vols.append(implied_vol)
            current_surface.option_type.append(option_type)

        if current_surface:
            surfaces.append(current_surface)

        return surfaces

    def get_or_create_asset(self, asset_type: str, ticker: str) -> int:
        """Get the asset ID or create a new asset if it doesn't exist"""
        with self.conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO assets (asset_type, ticker)
                VALUES (%s, %s)
                ON CONFLICT (ticker) DO NOTHING
                RETURNING id
            """, (asset_type, ticker))
            
            asset_id = cursor.fetchone()
            if asset_id is None:
                cursor.execute("SELECT id FROM assets WHERE ticker = %s", (ticker,))
                asset_id = cursor.fetchone()
            
            return asset_id[0]