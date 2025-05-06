import asyncio
from datetime import datetime, timedelta
import pandas as pd
import signal
import warnings

from infrastructure.settings import Settings
from data.exchanges.deribit import DeribitAPI
from core.VolatilityEngine import VolatilityEngine
from data.storage import StorageFactory
from infrastructure.utils.logging import setup_logger
from data.utils.data_schemas import OptionContract, MarketState 
from data.providers import Provider, DeribitProvider

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")


class MarketDataEngine:
    """
    A class for managing market data collection, processing, and storage for cryptocurrency options.

    This engine is responsible for:
    - Initializing and maintaining a list of active instruments
    - Collecting market data at regular intervals
    - Processing and calculating volatility surfaces
    - Storing market data and derived analytics

    Attributes:
        settings (Settings): Configuration settings for the engine
        exchange_api (DeribitAPI): API interface for the exchange
        vol_engine (VolatilityEngine): Engine for volatility calculations
        currency (str): The cryptocurrency being monitored (e.g., "BTC", "ETH")
        asset_type (str): Type of asset (default: "crypto")
        asset_id (int): Unique identifier for the asset
        max_retries (int): Maximum number of retry attempts for failed operations
        min_expiry_days (int): Minimum days to expiry for monitored options
        max_expiry_days (int): Maximum days to expiry for monitored options
        min_moneyness (float): Minimum moneyness ratio for monitored options
        max_moneyness (float): Maximum moneyness ratio for monitored options
    """

    def __init__(
        self,
        exchange_api: DeribitAPI,
        vol_engine: VolatilityEngine,
        settings: Settings = Settings(),
        currency: str = "ETH",
        asset_type: str = "crypto",
        asset_id: int = 0,
        store = None,
        provider: Provider = None,
    ):
        self.settings = settings
        self.exchange_api = exchange_api
        self.vol_engine = vol_engine
        self.currency = currency
        self.asset_type = asset_type
        self.asset_id = asset_id

        self.max_retries = 1
        self.store = store
        if self.store is None:
            self.store = StorageFactory.create_storage(self.settings)
        self.provider = provider
        self.state = MarketState(
            last_update=datetime.now(), active_instruments=set(), last_price=float
        )

        self.min_expiry_days = 1
        self.max_expiry_days = 90
        self.min_moneyness = 0.5
        self.max_moneyness = 1.5

        self.logger = setup_logger("market_data_worker")
        self.instruments_initialized = False

    
    def _store_data(self, last_price, options_chain, vol_surface):
        self.store.store_underlying(last_price, self.asset_type, self.currency)
        print(f"{datetime.now()}: Underlying data stored successfully\n")
        print(f"{datetime.now()}: Storing options chain\n")
        self.store.store_options_chain(options_chain)
        self.store.store_surface(vol_surface)

    def _get_vol_surface(self, option_chain, asset_id):
        """
        Generate the volatility surface from options chain data.

        Args:
            option_chain (pd.DataFrame): Options chain data

        Returns:
            VolatilitySurface: Object containing the calculated volatility surface,
                              including term structure and skew information
        """
        print("GETTING VOL SURFACE")
        print(f"Option chain type: {type(option_chain)}")
        print(f"Option chain length: {len(option_chain) if option_chain is not None else 'None'}")
        
        for row in option_chain.itertuples():
            # print(f"Processing option: {row.symbol}")
            self.vol_engine.add_market_data(
                timestamp=row.timestamp,
                strike=row.strike,
                moneyness=row.moneyness,
                option_type=row.option_type,
                expiry_date=row.expiry_date,
                days_to_expiry=row.days_to_expiry,
                implied_vol=row.implied_vol,
                delta=row.delta,
                gamma=row.gamma,
                vega=row.vega,
                theta=row.theta,
                snapshot_id=row.snapshot_id,
                asset_id=asset_id,
            )

        print(f"All options processed, generating surface...")
        
        # snapshot_id = datetime.now().isoformat()
        # print(f"Using snapshot_id: {snapshot_id}")
        snapshot_id = self.vol_engine.get_latest_snapshot_id()
        vol_surface = self.vol_engine.get_volatility_surface(snapshot_id, asset_id)
        print(f"Vol surface generated: {vol_surface is not None}")
        if vol_surface:
            print(f"Surface contains {len(vol_surface.strikes)} points")

        skews = self.vol_engine.get_skews(vol_surface)
        print(f"Skew generated: {skews is not None}")
        print(f"Skew contains {skews}")

        term_structure = self.vol_engine._get_term_structure(vol_surface)
        print(f"Term structure contains {term_structure}")

        # # get implied volatility index
        implied_volatility_index = self.vol_engine.get_implied_volatility_index(vol_surface)
        print(f"Implied volatility index generated: {implied_volatility_index}")

        surface_metrics = self.vol_engine.get_surface_metrics(vol_surface)
        print(f"Surface metrics generated: {surface_metrics}")

        return vol_surface
   
    async def process_market_updates(self):
        """Processing loop for market updates"""
        self.logger.info(f"{datetime.now()}: Starting market data processing...")

        try:
            # if not self._check_market_state():
            #     await asyncio.sleep(60)

            options_data = await self.provider.process_currency_updates()
            surface = self._get_vol_surface(options_data["chain"], options_data["asset_id"])
            print("... almost done, surfaces are about to be surfed")
            self._store_data(options_data["last_price"], options_data["chain"], surface)
            self.state.last_update = datetime.now()
            self.state.error_count = 0
            print(f"{datetime.now()}: Last update: {self.state.last_update}\n")

        except Exception as e:
            await self._handle_error(e)

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
            wait_time = min(120, 2**self.state.error_count)
            self.logger.info(f"Retrying in {wait_time} seconds...")
            await asyncio.sleep(wait_time)

    async def run(self, interval_minutes=5):
        """
        Main entry point for the market data worker.

        Args:
            interval_minutes (int): Time in minutes between each processing cycle

        This method:
        - Initializes instruments
        - Runs continuous processing loop
        - Handles scheduling and error recovery
        - Manages graceful shutdown

        Returns:
            None
        """
        self.logger.info(
            f"Starting market data worker with {interval_minutes} minute intervals"
        )

        # Initialize instruments once at startup
        # TODO should be called only once per day
        await self.provider.initialize_instruments()

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
    store = StorageFactory.create_storage(settings)
    exchange_api = DeribitAPI()
    provider = DeribitProvider(currency="BTC", store=store)
    vol_engine = VolatilityEngine()
    currency = "BTC"

    worker = MarketDataEngine(
        exchange_api=exchange_api,
        vol_engine=vol_engine,
        settings=settings,
        currency=currency,
        store=store,
        provider=provider,
    )

    await worker.run(interval_minutes=2)


if __name__ == "__main__":
    asyncio.run(main())


# get_option_data is currently run twice, once in _fetch_instruments and once in _process_currency_updates
# this leads to a lot of redundant calls to the API and slows down the algorithm
# instead it should call the api periodically and use a websocket to get the updates.