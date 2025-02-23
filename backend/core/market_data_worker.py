import asyncio
from datetime import datetime, timedelta
import pandas as pd
import signal
import warnings

from infrastructure.settings import Settings
from data_access.exchanges.deribit import DeribitAPI
from core.volatility_engine import VolatilityEngine
from data_access.storage import StorageFactory
from infrastructure.utils.logging import setup_logger
from data_access.utils.data_schemas import OptionContract, MarketState, VolSurface

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")


class MarketDataWorker:
    def __init__(
        self,
        exchange_api: DeribitAPI,
        vol_engine: VolatilityEngine,
        settings: Settings = Settings(),
        currencies: str = "ETH",
        asset_type: str = "crypto",
    ):
        self.settings = settings
        self.exchange_api = exchange_api
        self.vol_engine = vol_engine
        self.currencies = currencies
        self.asset_type = asset_type

        self.max_retries = 1
        self.store = StorageFactory.create_storage(settings)
        self.state = MarketState(
            last_update=datetime.now(), active_instruments=set(), last_price=float
        )

        self.min_expiry_days = 1
        self.max_expiry_days = 90
        self.min_moneyness = 0.5
        self.max_moneyness = 1.5

        self.logger = setup_logger("market_data_worker")
        self.instruments_initialized = False

    async def get_last_price(self, currency: str) -> float:
        try:
            current_price = self.exchange_api.get_last_price(currency)
            if not current_price:
                raise ValueError(f"Could not get price for {currency}")

            self.state.last_price = current_price
            print(f"{datetime.now()}: Last price: {self.state.last_price}")
            return current_price

        except Exception as e:
            self.logger.error(f"Error getting last price for {currency}: {e}")
            return None

    async def initialize_instruments(self):
        """Initialize the set of instruments to monitor"""
        if self.instruments_initialized:
            self.logger.info("Instruments have already been initialized.")
            return

        for currency in self.currencies:
            # Get current price for moneyness calculation
            current_price = await self.get_last_price(currency)
            if not current_price:
                self.logger.error(f"Could not get current price for {currency}")
                continue

            instruments = self.exchange_api.get_options(currency)
            current_time = datetime.now()

            filtered = []
            for inst in instruments.get("options", []):
                option_data = self.exchange_api.get_option_data(inst["instrument_name"])
                if not option_data or not option_data["last_price"] > 0:
                    continue

                # Parse instrument name for filtering
                parts = inst["instrument_name"].split("-")
                if len(parts) < 4:
                    continue

                # Calculate days to expiry
                try:
                    expiry_date = datetime.strptime(parts[1], "%d%b%y")
                    days_to_expiry = (expiry_date.date() - current_time.date()).days

                    # Calculate moneyness
                    strike = float(parts[2])
                    moneyness = strike / current_price

                    # Apply filters:
                    # 1. Days to expiry between 1 and 90 days
                    # 2. Moneyness between 0.5 and 1.5 (50% to 150% of current price)
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
                    f"Added {len(filtered)} filtered instruments for {currency}"
                )

        self.instruments_initialized = True

    async def process_market_updates(self):
        """Processing loop for market updates"""
        self.logger.info(f"{datetime.now()}: Starting market data processing...")

        try:
            if not self._check_market_state() and not any(
                currency in ["BTC", "ETH"] for currency in self.currencies
            ):
                await asyncio.sleep(60)

            print(f"{datetime.now()}: Processing updates for {self.currencies}\n")
            for currency in self.currencies:
                await self._process_currency_updates(currency)

            self.state.last_update = datetime.now()
            self.state.error_count = 0
            print(f"{datetime.now()}: Last update: {self.state.last_update}\n")

        except Exception as e:
            await self._handle_error(e)

    async def _process_currency_updates(self, currency: str):
        """Process updates for a specific currency using OptionContract objects"""
        print(f"{datetime.now()}: Starting to process currency updates\n")

        try:
            last_price = await self.get_last_price(currency)
            asset_id = self.store.get_or_create_asset(self.asset_type, currency)
            self.store.store_underlying(last_price, self.asset_type, currency)
            print(f"{datetime.now()}: Underlying data stored successfully\n")

            market_data_points = []
            current_time = datetime.now()

            for symbol in self.state.active_instruments:
                if currency in symbol:
                    option_data = self.exchange_api.get_option_data(symbol)
                    if option_data is None:
                        raise LookupError(f"No option data found for symbol: {symbol}")

                    parts = symbol.split("-")
                    if len(parts) < 4:
                        raise ValueError(f"Invalid symbol format: {symbol}")

                    expiry_date = datetime.strptime(parts[1], "%d%b%y")
                    days_to_expiry = (expiry_date.date() - current_time.date()).days
                    strike = float(parts[2])
                    forward = self.state.last_price
                    moneyness = strike / forward if forward else None

                    option = OptionContract(
                        timestamp=current_time,
                        asset_id=asset_id,
                        base_currency=currency,
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
                        snapshot_id=current_time.isoformat(),
                    )

                    market_data_points.append(option)

            if market_data_points:
                print(f"{datetime.now()}: Storing options chain\n")
                options_dicts = [vars(option) for option in market_data_points]
                combined_df = pd.DataFrame(options_dicts)

                combined_df["bid_price"].fillna(0, inplace=True)
                combined_df["ask_price"].fillna(0, inplace=True)
                combined_df["open_interest"].fillna(0, inplace=True)

                self.store.store_options_chain(combined_df)

                print(f"{datetime.now()}: adding data to volatility engine\n")
                for option in market_data_points:
                    self.vol_engine.add_market_data(
                        timestamp=option.timestamp,
                        strike=option.strike,
                        moneyness=option.moneyness,
                        option_type=option.option_type,
                        expiry_date=option.expiry_date,
                        days_to_expiry=option.days_to_expiry,
                        implied_vol=option.implied_vol,
                        delta=option.delta,
                        gamma=option.gamma,
                        vega=option.vega,
                        theta=option.theta,
                        snapshot_id=option.snapshot_id,
                    )

                print(f"{datetime.now()}: Processing surface data\n")
                vol_surface = self.vol_engine.get_latest_volatility_surface(
                    current_time.isoformat()
                )

                if vol_surface:
                    print(f"{datetime.now()}: Storing surface")

                    if asset_id is None:
                        raise ValueError(f"Failed to get or create asset_id for {currency}")
                    vol_surface.asset_id = asset_id

                    self.store.store_surface(vol_surface)

                    print(f"{datetime.now()}: Retrieving surface from storage")

                    retrieved_surface = self.store.get_vol_surfaces(
                        vol_surface.timestamp, vol_surface.snapshot_id
                    )

                    if retrieved_surface:
                        print(
                            f"{datetime.now()}: Successfully retrieved surface"
                        )

                    else:
                        print(
                            f"{datetime.now()}: Warning - Could not retrieve stored surface"
                        )
                else:
                    print(
                        f"{datetime.now()}: Warning - No surface generated"
                    )

        except ValueError as ve:
            self.logger.error(f"Value error processing {currency} updates: {ve}")
            raise
        except LookupError as le:
            self.logger.error(f"Lookup error processing {currency} updates: {le}")
            raise
        except Exception as e:
            self.logger.error(f"General error processing {currency} updates: {e}")
            raise

    def _check_market_state(self) -> bool:
        """Check if market is in normal operating state"""
        current_time = datetime.now()
        is_weekend = current_time.weekday() >= 5
        updates_stale = (current_time - self.state.last_update).total_seconds() > 60
        too_many_errors = self.state.error_count >= self.max_retries

        return not (is_weekend or updates_stale or too_many_errors)

    async def _handle_error(self, error: Exception):
        """Handle processing errors"""
        self.state.error_count += 1
        self.logger.error(f"Processing error: {error}")

        if self.state.error_count >= self.max_retries:
            self.logger.critical("Max retries exceeded. Initiating shutdown...")
            self.is_running = False
        else:
            wait_time = min(300, 2**self.state.error_count)
            self.logger.info(f"Retrying in {wait_time} seconds...")
            await asyncio.sleep(wait_time)

    async def run(self, interval_minutes=5):
        """
        Main entry point for the market data worker.
        Runs the market data processing loop every X minutes.

        Args:
            interval_minutes (int): Time in minutes between each processing cycle
        """
        self.logger.info(
            f"Starting market data worker with {interval_minutes} minute intervals"
        )

        # Initialize instruments once at startup
        await self.initialize_instruments()

        while True:
            try:
                cycle_start = datetime.now()
                self.logger.info("Starting new data collection cycle")

                await self.process_market_updates()

                # Calculate time until next run
                cycle_duration = (datetime.now() - cycle_start).total_seconds()
                sleep_time = max(0, interval_minutes * 60 - cycle_duration)

                if sleep_time > 0:
                    self.logger.info(
                        f"Cycle completed in {cycle_duration:.1f}s. Sleeping for {sleep_time:.1f}s"
                    )
                    await asyncio.sleep(sleep_time)
                else:
                    self.logger.warning(
                        f"Cycle took {cycle_duration:.1f}s, longer than the {interval_minutes} minute interval"
                    )
            except asyncio.CancelledError:
                self.logger.info("Worker shutdown requested")
                break
            except Exception as e:
                self.logger.error(f"Error in processing cycle: {e}")
                await self._handle_error(e)
                if self.state.error_count >= self.max_retries:
                    break


async def main():
    settings = Settings()

    exchange_api = DeribitAPI()
    vol_engine = VolatilityEngine()
    currencies = ["ETH"]

    worker = MarketDataWorker(
        exchange_api=exchange_api,
        vol_engine=vol_engine,
        settings=settings,
        currencies=currencies,
    )

    await worker.run(interval_minutes=1)


if __name__ == "__main__":
    asyncio.run(main())


# get_option_data is currently run twice, once in _fetch_instruments and once in _process_currency_updates
# this leads to a lot of redundant calls to the API and slows down the algorithm
# instead it should call the api periodically and use a websocket to get the updates.