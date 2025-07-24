from abc import ABC, abstractmethod
import pandas as pd
from datetime import datetime
from typing import Optional, List
from data.utils.data_schemas import VolSurface


class BaseStore(ABC):

    @abstractmethod
    def store_underlying(self, underlying_df: pd.DataFrame, symbol: str):
        """Store underlying asset data"""
        pass

    @abstractmethod
    def get_underlying_data(self, symbol: str) -> pd.DataFrame:
        """Retrieve the latest underlying data for a symbol"""
        pass
    
    @abstractmethod
    def store_options_chain(self, options_df: pd.DataFrame, symbol: str):
        """Store options chain data"""
        pass

    @abstractmethod
    def get_options_chain(self, symbol: str) -> pd.DataFrame:
        """Retrieve the latest options chain for a symbol"""
        pass
    
    @abstractmethod
    def get_latest_vol_surface(self):
        """Retrieve the latest surface data"""
        pass
    
    @abstractmethod
    def get_available_assets(self):
        """Get all available assets from the database"""
        pass

    @abstractmethod
    def get_options_by_snapshot(self, snapshot_id: str, asset_id: str) -> pd.DataFrame:
        """Get options data for a specific snapshot and asset"""
        pass

    @abstractmethod
    def get_latest_snapshot_id(self, asset_id: str) -> Optional[str]:
        """Get the latest snapshot ID for a given asset"""
        pass

    @abstractmethod
    def store_calibration_results(self, asset_id: str, snapshot_id: str, calibration_data: dict) -> int:
        """Store calibration results in the database"""
        pass

    @abstractmethod
    def get_calibration_history(self, asset_id: str, lookback_days: int = 30) -> List[dict]:
        """Get calibration history for performance analysis"""
        pass

    @abstractmethod
    def get_surface_quality_metrics(self, snapshot_id: str, asset_id: str) -> dict:
        """Calculate surface quality metrics from options data"""
        pass