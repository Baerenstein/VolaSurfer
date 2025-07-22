from abc import ABC, abstractmethod
import pandas as pd
from datetime import datetime
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