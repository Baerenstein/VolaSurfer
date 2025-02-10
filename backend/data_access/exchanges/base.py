from abc import ABC, abstractmethod
from typing import Dict


class ExchangeAPI(ABC):
    @abstractmethod
    def test_connection(self) -> bool:
        pass

    @abstractmethod
    def get_options(self, currency: str) -> Dict:
        pass

    @abstractmethod
    def get_last_price(self, currency: str) -> float:
        pass

    @abstractmethod
    def get_option_data(self, instrument_name: str) -> Dict:
        pass
