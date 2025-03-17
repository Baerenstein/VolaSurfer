from data.utils.data_schemas import OptionContract, MarketState
from infrastructure.utils.logging import setup_logger
from data.exchanges.deribit import DeribitAPI
from datetime import datetime
from data.exchanges import ExchangeAPI
from typing import TypedDict
import pandas as pd
from core.VolatilityEngine import VolatilityEngine


# {'instrument_name': 'BTC-3MAR25-79000-C', 'last_price': 0.1556, 'implied_vol': 177.52, 'delta': 0.9927, 'gamma': 0.0, 'vega': 0.74712, 'theta': -66.31527}
class InstrumentData(TypedDict):
    instrument_name: str
    last_price: float
    implied_vol: float
    delta: float
    gamma: float
    vega: float
    theta: float

class OptionsData(TypedDict):
    chain: list[InstrumentData]
    last_price: float
    asset_id: int

class Provider:
    async def initialize_instruments(self):
        raise NotImplementedError
    async def update_market_data(self):
        raise NotImplementedError

class DeribitProvider(Provider):
    def __init__(self, currency: str, store):
        self.exchange_api = DeribitAPI()
        self.store = store
        self.currency = currency
        self.min_expiry_days = 1
        self.max_expiry_days = 90
        self.min_moneyness = 0.5
        self.max_moneyness = 1.5
        self.vol_engine = VolatilityEngine()
        self.asset_type = "crypto"
        self.state = MarketState(
            last_update=datetime.now(),
            active_instruments=set(),
            last_price=float
        )
        self.logger = setup_logger("deribit_provider")
        self.instruments_initialized = False

    async def initialize_instruments(self):
        """Initialize the set of instruments to monitor"""
        current_time = datetime.now()
        self.asset_id = self.store.get_or_create_asset(self.asset_type, self.currency)

        # add time check for when was the last time
        if self.instruments_initialized:
            self.logger.info("Instruments have already been initialized.")
            return

        current_price = await self.get_last_price()
        instruments = self.exchange_api.get_options(self.currency)

        # this below should be optimised
        filtered = []
        for inst in instruments.get("options", []):
            option_data = self.exchange_api.get_option_data(inst["instrument_name"])
            if not option_data or not option_data["last_price"] > 0:
                continue
            parts = inst["instrument_name"].split("-")
            if len(parts) < 4:
                continue
            try:
                expiry_date = datetime.strptime(parts[1], "%d%b%y")
                days_to_expiry = (expiry_date.date() - current_time.date()).days
                strike = float(parts[2])
                moneyness = strike / current_price
                if (
                    self.min_expiry_days <= days_to_expiry <= self.max_expiry_days
                    and self.min_moneyness <= moneyness <= self.max_moneyness
                ):
                    filtered.append(inst)
            except (ValueError, IndexError) as e:
                self.logger.warning(
                    f"Error parsing instrument {inst['instrument_name']}: {e}"
                )
                continue
        
        if filtered:
            self.state.active_instruments.update(
                symbol["instrument_name"] for symbol in filtered
            )
            self.logger.info(
                f"Added {len(filtered)} filtered instruments for {self.currency}"
            )
            self.instruments_initialized = True

    async def process_currency_updates(self) -> OptionsData:
        """Process updates for a specific currency using OptionContract objects"""
        print(f"{datetime.now()}: Starting to process currency updates\n")
        last_price = await self.get_last_price()
        print("fetched current price, next up options chain")
        options_chain = self._get_options_chain()
        return {"last_price": last_price, "chain": options_chain, "asset_id": self.asset_id}

    async def get_last_price(self) -> float:
        print("GETTING LAST PRICE", self.currency)
        try:
            current_price = self.exchange_api.get_last_price(self.currency)
            if not current_price:
                raise ValueError(f"Could not get price for {self.currency}")

            self.state.last_price = current_price
            print(f"{datetime.now()}: Last price: {self.state.last_price}")
            return current_price

        except Exception as e:
            self.logger.error(f"Error getting last price for {self.currency}: {e}")
            return None
        
      
    def _get_options_chain(self):
        print("GETTING OPTIONS CHAIN")
        current_time = datetime.now()
        data_points = []
        for symbol in self.state.active_instruments:
            if self.currency in symbol:
                option_data = self.exchange_api.get_option_data(symbol)
                if option_data is None:
                    raise LookupError(f"No option data found for symbol: {symbol}")

                parts = symbol.split("-")
                if len(parts) < 4:
                    raise ValueError(f"Invalid symbol format: {symbol}")
                expiry_date = datetime.strptime(parts[1], "%d%b%y")
                days_to_expiry = (expiry_date.date() - current_time.date()).days
                strike = float(parts[2])
                price = self.state.last_price
                moneyness = strike / price if price else None

                option = OptionContract(
                    timestamp=current_time,
                    asset_id=self.asset_id,
                    base_currency=self.currency,
                    symbol=symbol,
                    expiry_date=expiry_date,
                    days_to_expiry=days_to_expiry,
                    strike=strike,
                    moneyness=moneyness,
                    option_type=parts[3].lower(),
                    last_price=option_data.get("last_price", 0),
                    implied_vol=option_data.get("implied_vol", 0),
                    bid_price=option_data.get("bid_price"),
                    ask_price=option_data.get("ask_price"),
                    delta=option_data.get("delta"),
                    gamma=option_data.get("gamma"),
                    vega=option_data.get("vega"),
                    theta=option_data.get("theta"),
                    open_interest=option_data.get("open_interest"),
                    snapshot_id=current_time.isoformat(), # change snapshot id method
                )

                data_points.append(option)
                options_dicts = [vars(option) for option in data_points]
                options_chain = pd.DataFrame(options_dicts)
                options_chain["bid_price"].fillna(0, inplace=True)
                options_chain["ask_price"].fillna(0, inplace=True)
                options_chain["open_interest"].fillna(0, inplace=True)

        return options_chain
    
    