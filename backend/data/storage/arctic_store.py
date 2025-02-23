from datetime import datetime, timedelta
import pandas as pd

# import arcticdb as adb
from collections import defaultdict
from typing import Optional

from data.storage.base_store import BaseStore
from data.utils.data_schemas import VolSurface


class ArcticStore(BaseStore):
    def __init__(self, uri: str = "lmdb://tmp/trading_data"):
        self.arctic = adb.Arctic(uri)
        self.initialize_libraries()
        self.temp_storage = defaultdict(list)
        self.last_dump_time = datetime.now()

    def initialize_libraries(self):
        """Initialize required libraries in ArcticDB"""
        libraries = ["options_chain", "underlying_data"]
        for lib in libraries:
            self.arctic.get_library(lib, create_if_missing=True)

    def store_options_chain(self, options_df: pd.DataFrame, symbol: str):
        """Store options chain data"""
        library = self.arctic["options_chain"]
        key = f"{symbol}_options_chain"
        # Write the updated data back to the library
        library.append(key, options_df, metadata=None)

    def store_underlying(self, last_price: float, symbol: str):
        """Store underlying asset data"""
        underlying_df = pd.DataFrame(
            {"symbol": [symbol], "price": [last_price], "timestamp": [datetime.now()]}
        )

        library = self.arctic["underlying_data"]
        key = f"{symbol}_underlying"
        library.write(key, underlying_df)

    def get_options_chain(self, symbol: str) -> pd.DataFrame:
        """Retrieve the latest options chain for a symbol"""
        library = self.arctic["options_chain"]
        key = f"{symbol}_options_chain"
        try:
            return library.read(key).data
        except Exception as e:
            print(f"Error reading options chain for {symbol}: {e}")
            return None

    def get_underlying_data(self, symbol: str) -> pd.DataFrame:
        """Retrieve the latest underlying data for a symbol"""
        library = self.arctic["underlying_data"]
        key = f"{symbol}_underlying"
        try:
            return library.read(key).data
        except Exception as e:
            print(f"Error reading underlying data for {symbol}: {e}")
            return None

    def store_vol_surface(self, vol_surface: VolSurface, symbol: str) -> str:
        """
        Save a VolSurface object to ArcticDB.

        Args:
            vol_surface: VolSurface object to save

        Returns:
            str: The key used to store the vol surface
        """
        # Create a library for vol surfaces if it doesn't exist
        library = self.arctic.get_library("vol_surfaces", create_if_missing=True)

        # Convert vol surface data to DataFrame
        surface_data = pd.DataFrame(
            {
                "strike": vol_surface.strikes,
                "moneyness": vol_surface.moneyness,
                "maturity": vol_surface.maturities,
                "days_to_expiry": vol_surface.days_to_expiry,
                "implied_vol": vol_surface.implied_vols,
                "option_type": vol_surface.option_type,
            }
        )

        # Create a unique key for the vol surface
        key = f"{symbol}_vol_surface_{vol_surface.timestamp.strftime('%Y%m%d_%H%M%S')}"
        if vol_surface.snapshot_id:
            key += f"_{vol_surface.snapshot_id}"

        # Store metadata
        metadata = {
            "timestamp": vol_surface.timestamp,
            "method": vol_surface.method,
            "snapshot_id": vol_surface.snapshot_id,
        }

        # Write to ArcticDB
        library.write(key, surface_data, metadata=metadata)

        return key

    def get_vol_surfaces(
        self,
        timestamp: Optional[datetime],
        symbol: str,
        snapshot_id: Optional[str] = None,
    ) -> Optional[VolSurface]:
        """
        Retrieve a VolSurface object from ArcticDB.
        If timestamp is None, returns the latest available surface.

        Args:
            timestamp: Optional timestamp to retrieve, if None gets latest
            symbol: Symbol to retrieve surface for
            snapshot_id: Optional snapshot ID

        Returns:
            VolSurface: Retrieved volatility surface or None if not found
        """
        library = self.arctic.get_library("vol_surfaces")

        # Get all symbols for this currency
        all_symbols = library.list_symbols()
        matching_symbols = [
            sym for sym in all_symbols if sym.startswith(f"{symbol}_vol_surface_")
        ]

        if not matching_symbols:
            return None

        if timestamp is not None:
            # Format the timestamp part of the key
            time_key = timestamp.strftime("%Y%m%d_%H%M%S")

            if snapshot_id:
                # Look for exact match with snapshot_id
                key = f"{symbol}_vol_surface_{time_key}_{snapshot_id}"
                if key in matching_symbols:
                    matching_symbols = [key]
            else:
                # Filter for surfaces at this timestamp
                matching_symbols = [sym for sym in matching_symbols if time_key in sym]
        else:
            # Sort by timestamp (which is part of the key) to get the latest
            matching_symbols.sort(reverse=True)

        if not matching_symbols:
            return None

        # Get the first (latest) matching vol surface
        key = matching_symbols[0]
        version = library.read(key)

        # Extract data and metadata
        surface_data = version.data
        metadata = version.metadata

        # Reconstruct VolSurface object
        return VolSurface(
            timestamp=metadata["timestamp"],
            method=metadata["method"],
            strikes=surface_data["strike"].tolist(),
            moneyness=surface_data["moneyness"].tolist(),
            maturities=surface_data["maturity"].tolist(),
            days_to_expiry=surface_data["days_to_expiry"].tolist(),
            implied_vols=surface_data["implied_vol"].tolist(),
            option_type=surface_data["option_type"],
            snapshot_id=metadata["snapshot_id"],
        )
