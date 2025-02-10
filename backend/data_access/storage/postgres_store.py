from datetime import datetime
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import psycopg2
from psycopg2 import extras
from typing import List, Optional
import warnings

from data_access.storage.base_store import BaseStore
from data_access.utils.data_schemas import OptionContract, VolSurface, VolMetrics

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
        self.initialize_options_data_table()
        self.initialize_underlying_data_table()
        self.initialize_vol_surface_table()
        self.initialize_vol_surface_points_table()

    def initialize_options_data_table(self):
        """Create options_data table"""
        with self.conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS options_data (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP,
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
                    symbol VARCHAR,
                    price DOUBLE PRECISION,
                    timestamp TIMESTAMP
                )
            """)

    def initialize_vol_surface_table(self):
        """Create vol_surfaces table with increased VARCHAR length"""
        with self.conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS vol_surfaces (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL,
                    method TEXT NOT NULL,
                    snapshot_id TEXT,
                    UNIQUE(timestamp, snapshot_id)
                )
            """)

    # VolSurface TODO: add option type parameter
    def initialize_vol_surface_points_table(self):
        """Create vol_surface_points table"""
        with self.conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS vol_surface_points (
                    id SERIAL PRIMARY KEY,
                    vol_surface_id INTEGER REFERENCES vol_surfaces(id) ON DELETE CASCADE,
                    strike NUMERIC NOT NULL,
                    moneyness NUMERIC NOT NULL,
                    maturity TIMESTAMP NOT NULL,
                    days_to_expiry NUMERIC NOT NULL,
                    implied_vol NUMERIC NOT NULL,
                    option_type VARCHAR NOT NULL
                )
            """)

    # TODO the symbol should be used to store the data in the correct table
    def store_options_chain(self, options_df: pd.DataFrame, symbol: str):
        """Store options chain data with limited decimal places"""

        numeric_columns = [
            "days_to_expiry",
            "strike",
            "moneyness",
            "last_price",
            "implied_vol",
            "delta",
            "gamma",
            "vega",
            "theta",
        ]

        for col in numeric_columns:
            if col in options_df.columns:
                options_df[col] = options_df[col].apply(
                    lambda x: self._limit_decimals(x)
                )

        with self.conn.cursor() as cursor:
            for idx, row in options_df.iterrows():
                cursor.execute(
                    """
                    INSERT INTO options_data (
                        timestamp, symbol, option_type, base_currency,
                        expiry_date, days_to_expiry, strike, moneyness,
                        last_price, implied_vol, delta, gamma, vega, theta,
                        snapshot_id
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        row["timestamp"],
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
                    ),
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

    def store_underlying(self, last_price: float, symbol: str):
        """Store underlying asset data"""
        underlying_df = pd.DataFrame(
            {"symbol": [symbol], "price": [last_price], "timestamp": [datetime.now()]}
        )

        with self.conn.cursor() as cursor:
            for _, row in underlying_df.iterrows():
                cursor.execute(
                    """
                    INSERT INTO underlying_data (symbol, price, timestamp)
                    VALUES (%s, %s, %s)
                """,
                    (row["symbol"], row["price"], row["timestamp"]),
                )
            self.conn.commit()

    def get_underlying_data(self, symbol: str) -> pd.DataFrame:
        """Retrieve the latest underlying data for a symbol"""
        query = "SELECT * FROM underlying_data WHERE symbol = %s"
        df = pd.read_sql_query(query, self.conn, params=(symbol,))
        return df

    def store_vol_surface(self, vol_surface: VolSurface) -> int:
        """Store volatility surface with limited decimal places"""
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO vol_surfaces (timestamp, method, snapshot_id)
                    VALUES (%s, %s, %s)
                    RETURNING id
                """,
                    (
                        vol_surface.timestamp,
                        vol_surface.method,
                        vol_surface.snapshot_id,
                    ),
                )

                vol_surface_id = cur.fetchone()[0]

                points_data = [
                    (
                        vol_surface_id,
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
                    INSERT INTO vol_surface_points 
                        (vol_surface_id, strike, moneyness, maturity, days_to_expiry, implied_vol, option_type)
                    VALUES %s
                    """,
                    points_data,
                    page_size=100,
                )

                self.conn.commit()
                return vol_surface_id

        except Exception as e:
            self.conn.rollback()
            raise RuntimeError(f"Failed to store vol surface: {str(e)}")

    # VolSurface TODO: add option type parameter
    def get_vol_surfaces(
        self, timestamp: datetime, snapshot_id: Optional[str] = None
    ) -> VolSurface:
        """
        Retrieve a VolSurface object from the database.

        Args:
            timestamp: Timestamp to retrieve
            snapshot_id: Optional snapshot ID

        Returns:
            VolSurface: Retrieved volatility surface
        """
        with self.conn.cursor() as cur:
            query = """
                SELECT vsp.strike, vsp.moneyness, vsp.maturity, vsp.days_to_expiry, 
                    vsp.implied_vol, vsp.option_type, vs.method
                FROM vol_surfaces vs
                JOIN vol_surface_points vsp ON vs.id = vsp.vol_surface_id
                WHERE vs.timestamp = %s
            """
            params = [timestamp]

            if snapshot_id:
                query += " AND vs.snapshot_id = %s"
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

    def get_current_vol_surface(self) -> VolSurface:
        """
        Retrieve the most recent VolSurface object from the database.
        Returns float values instead of Decimal.
        """
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT timestamp, snapshot_id 
                FROM vol_surfaces 
                ORDER BY timestamp DESC 
                LIMIT 1
            """)
            latest = cur.fetchone()

            if not latest:
                return None

            latest_timestamp, latest_snapshot_id = latest

            query = """
                SELECT vsp.strike::float, vsp.moneyness::float, vsp.maturity, 
                    vsp.days_to_expiry::float, vsp.implied_vol::float, vsp.option_type, 
                    vs.method
                FROM vol_surfaces vs
                JOIN vol_surface_points vsp ON vs.id = vsp.vol_surface_id
                WHERE vs.timestamp = %s AND vs.snapshot_id = %s
            """

            cur.execute(query, (latest_timestamp, latest_snapshot_id))
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
                timestamp=latest_timestamp,
                method=method,
                strikes=strikes,
                moneyness=moneyness,
                maturities=maturities,
                days_to_expiry=days_to_expiry,
                implied_vols=implied_vols,
                option_type=option_type,
                snapshot_id=latest_snapshot_id,
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
                FROM vol_surfaces
                ORDER BY timestamp DESC
                LIMIT 1
            )
            SELECT 
                ls.timestamp,
                vsp.moneyness::float,
                vsp.days_to_expiry::float,
                vsp.implied_vol::float
            FROM vol_surface_points vsp
            INNER JOIN latest_surface ls ON vsp.vol_surface_id = ls.id
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