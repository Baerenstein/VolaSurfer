from abc import ABC, abstractmethod
import pandas as pd
from datetime import datetime
from data_access.utils.data_schemas import VolSurface


class BaseStore(ABC):
    @abstractmethod
    def store_options_chain(self, options_df: pd.DataFrame, symbol: str):
        """Store options chain data"""
        pass

    @abstractmethod
    def store_underlying(self, underlying_df: pd.DataFrame, symbol: str):
        """Store underlying asset data"""
        pass

    @abstractmethod
    def store_vol_surface(self, vol_surface: VolSurface):
        """Store surface data"""
        pass

    @abstractmethod
    def get_options_chain(self, symbol: str) -> pd.DataFrame:
        """Retrieve the latest options chain for a symbol"""
        pass

    @abstractmethod
    def get_underlying_data(self, symbol: str) -> pd.DataFrame:
        """Retrieve the latest underlying data for a symbol"""
        pass

    @abstractmethod
    def get_vol_surfaces(self, timestamp: datetime) -> VolSurface:
        """Retrieve the latest surface data for a timestamp"""
        pass

    @abstractmethod
    def get_current_vol_surface(self) -> VolSurface:
        """Retrieve the latest surface data"""
        pass
    
    @abstractmethod
    def get_latest_vol_surface(self):
        """Retrieve the latest surface data"""
        pass
