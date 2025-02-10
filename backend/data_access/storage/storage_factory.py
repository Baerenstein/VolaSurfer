from infrastructure.settings import StorageType, Settings
from .base_store import BaseStore
from .postgres_store import PostgresStore
# from .arctic_store import ArcticStore


class StorageFactory:
    @staticmethod
    def create_storage(settings: Settings) -> BaseStore:
        if settings.STORAGE.STORAGE_TYPE == StorageType.POSTGRES:
            return PostgresStore(settings.STORAGE.POSTGRES_URI)
        # if settings.STORAGE.STORAGE_TYPE == StorageType.ARCTIC:
        #     return ArcticStore(settings.STORAGE.ARCTIC_URI)
        else:
            raise ValueError(f"Unknown storage type: {settings.STORAGE.STORAGE_TYPE}")
