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

    # Default moneyness range for surface filtering
    MIN_MONEYNESS: float = 0.6
    MAX_MONEYNESS: float = 1.4
    # Default maturity (days to expiry) range for surface filtering
    MIN_MATURITY: int = 0
    MAX_MATURITY: int = 30
    class Config:
        env_file = ".env"
