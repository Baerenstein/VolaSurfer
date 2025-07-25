from enum import Enum
from pydantic_settings import BaseSettings
from typing import Tuple


class StorageType(Enum):
    ARCTIC = "arctic"
    POSTGRES = "postgres"


class StorageConfig(BaseSettings):
    """Configuration for data storage"""

    STORAGE_TYPE: StorageType = StorageType.POSTGRES
    ARCTIC_URI: str = "lmdb://tmp/trading_data"
    POSTGRES_URI: str = "postgresql://mikeb:postgres@localhost:5432/optionsdb"  # Update with your credentials


class Settings(BaseSettings):
    """Main configuration"""

    STORAGE: StorageConfig = StorageConfig()

    #TODO these are not used anywhere yet
    MONEYNESS_RANGE: Tuple[float, float] = (0.9, 1.1)
    DAYS_TO_EXPIRY_RANGE: Tuple[int, int] = (0, 10)
    class Config:
        env_file = ".env"
