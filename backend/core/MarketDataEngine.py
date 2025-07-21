import asyncio
from datetime import datetime, timedelta
import pandas as pd
import signal
import warnings
import argparse
import sys

from infrastructure.settings import Settings
from data.exchanges.deribit import DeribitAPI
from core.VolatilityEngine import VolatilityEngine
from data.storage import StorageFactory
from infrastructure.utils.logging import setup_logger
from data.utils.data_schemas import OptionContract, MarketState 

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
        asset_id: int = 0
    ):
        self.settings = settings
        self.exchange_api = exchange_api
        self.vol_engine = vol_engine
        self.currency = currency
        self.asset_type = asset_type
        self.asset_id = asset_id

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

    async def initialize_instruments(self):
        """
        Initialize the set of instruments to monitor.

        This method:
        - Fetches the current price of the underlying asset
        - Retrieves available options from the exchange
        - Filters options based on expiry and moneyness criteria
        - Updates the active instruments set

        Returns:
            None
        """
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
          
    async def process_market_updates(self):
        """
        Process market updates for all active instruments.

        This method coordinates the main data collection and processing cycle:
        - Checks market state
        - Processes currency updates
        - Updates last update timestamp
        - Handles any errors that occur during processing

        Returns:
            None

        Raises:
            Exception: If any error occurs during processing
        """
        self.logger.info(f"{datetime.now()}: Starting market data processing...")

        try:
            # if not self._check_market_state():
            #     await asyncio.sleep(60)

            await self._process_currency_updates()

            self.state.last_update = datetime.now()
            self.state.error_count = 0
            print(f"{datetime.now()}: Last update: {self.state.last_update}\n")

        except Exception as e:
            await self._handle_error(e)

    async def _process_currency_updates(self):
        """
        Process updates for the current currency.

        This method:
        - Fetches the current price
        - Gets the options chain
        - Calculates the volatility surface
        - Stores the collected and calculated data

        Returns:
            None
        """
        print(f"{datetime.now()}: Starting to process currency updates\n")
        last_price = await self.get_last_price()
        options_chain = self._get_options_chain()
        surface = self._get_vol_surface(options_chain)
        self._store_data(last_price, options_chain, surface)
   
    async def get_last_price(self) -> float:
        """
        Fetch the current price of the underlying asset.

        Returns:
            float: The current price of the underlying asset

        Raises:
            ValueError: If unable to get the current price
        """
        # print("GETTING LAST PRICE")
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
        """
        Collect and process options chain data for active instruments.

        Returns:
            pd.DataFrame: DataFrame containing options chain data with columns:
                - timestamp: Time of data collection
                - symbol: Option instrument identifier
                - strike: Strike price
                - expiry_date: Option expiration date
                - option_type: Put or Call
                - last_price: Last traded price
                - implied_vol: Implied volatility
                - greeks: Delta, Gamma, Vega, Theta
                - other market data
        """
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

    def _get_vol_surface(self, option_chain):
        """
        Generate the volatility surface from options chain data.

        Args:
            option_chain (pd.DataFrame): Options chain data

        Returns:
            VolatilitySurface: Object containing the calculated volatility surface,
                              including term structure and skew information
        """
        # print("GETTING VOL SURFACE")
        # print(f"Option chain type: {type(option_chain)}")
        # print(f"Option chain length: {len(option_chain) if option_chain is not None else 'None'}")
        
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
                asset_id=self.asset_id,
            )

        # print(f"All options processed, generating surface...")
        
        # snapshot_id = datetime.now().isoformat()
        # print(f"Using snapshot_id: {snapshot_id}")
        snapshot_id = self.vol_engine.get_latest_snapshot_id()
        vol_surface = self.vol_engine.get_volatility_surface(snapshot_id, self.asset_id)
        # print(f"Vol surface generated: {vol_surface is not None}")
        # if vol_surface:
        #     print(f"Surface contains {len(vol_surface.strikes)} points")

        # skews = self.vol_engine.get_skews(vol_surface)
        # print(f"Skew generated: {skews is not None}")
        # print(f"Skew contains {skews}")

        # term_structure = self.vol_engine._get_term_structure(vol_surface)
        # print(f"Term structure generated: {term_structure is not None}")
        # print(f"Term structure contains {term_structure}")

        # # get implied volatility index
        # implied_volatility_index = self.vol_engine.get_implied_volatility_index(vol_surface)
        # print(f"Implied volatility index generated: {implied_volatility_index}")

        # surface_metrics = self.vol_engine.get_surface_metrics(vol_surface)
        # print(f"Surface metrics generated: {surface_metrics is not None}")
        # print(f"Surface metrics contains {surface_metrics}")

        return vol_surface

    def _store_data(self, last_price, options_chain, vol_surface):
        """
        Store collected and calculated market data.

        Args:
            last_price (float): Current price of the underlying
            options_chain (pd.DataFrame): Options chain data
            vol_surface (VolatilitySurface): Calculated volatility surface

        Returns:
            None
        """
        print("STORING DATA")
        self.store.store_underlying(last_price, self.asset_type, self.currency)
        print(f"{datetime.now()}: Underlying data stored successfully\n")

        print(f"{datetime.now()}: Storing options chain\n")
        self.store.store_options_chain(options_chain)
        self.store.store_surface(vol_surface)

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
        - Handles scheduling and error recovery

        Returns:
            None
        """
        self.logger.info(
            f"Starting {self.currency} market data worker with {interval_minutes} minute intervals"
        )

        # Initialize instruments once at startup
        # TODO should be called only once per day
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


def parse_arguments():
    """Parse command line arguments for currency selection."""
    parser = argparse.ArgumentParser(
        description="Market Data Engine for cryptocurrency options",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m core.MarketDataEngine -BTC              # Run for Bitcoin (1 min intervals)
  python -m core.MarketDataEngine -ETH --5min       # Run for Ethereum (5 min intervals)
  python -m core.MarketDataEngine BTC --1hr         # Run for Bitcoin (1 hour intervals)
  python -m core.MarketDataEngine --currency BTC -i 10  # Custom 10 minute intervals
  python -m core.MarketDataEngine -BTC --15min      # Bitcoin with 15 minute intervals
        """
    )
    
    # Support multiple ways to specify currency
    parser.add_argument(
        'currency', 
        nargs='?', 
        default=None,
        help='Currency symbol (BTC, ETH, etc.)'
    )
    
    parser.add_argument(
        '--currency', '-c',
        dest='currency_option',
        help='Currency symbol (alternative to positional argument)'
    )
    
    parser.add_argument(
        '--interval', '-i',
        type=int,
        default=None,
        help='Update interval in minutes (default: 1)'
    )
    
    # Preset interval options
    interval_group = parser.add_mutually_exclusive_group()
    interval_group.add_argument('--1min', action='store_const', const=1, dest='preset_interval', help='1 minute intervals')
    interval_group.add_argument('--5min', action='store_const', const=5, dest='preset_interval', help='5 minute intervals')
    interval_group.add_argument('--15min', action='store_const', const=15, dest='preset_interval', help='15 minute intervals')
    interval_group.add_argument('--30min', action='store_const', const=30, dest='preset_interval', help='30 minute intervals')
    interval_group.add_argument('--1hr', action='store_const', const=60, dest='preset_interval', help='1 hour intervals')
    
    # Add support for -BTC, -ETH style arguments
    parser.add_argument('-BTC', action='store_const', const='BTC', dest='crypto_flag')
    parser.add_argument('-ETH', action='store_const', const='ETH', dest='crypto_flag')
    
    args = parser.parse_args()
    
    # Determine currency from various argument sources
    currency = None
    if args.crypto_flag:
        currency = args.crypto_flag
    elif args.currency_option:
        currency = args.currency_option.upper()
    elif args.currency:
        currency = args.currency.upper()
    else:
        # Default to ETH if no currency specified
        currency = "ETH"
        print("No currency specified, defaulting to ETH")
    
    # Determine interval from various sources
    interval = None
    if args.preset_interval:
        interval = args.preset_interval
    elif args.interval:
        if args.interval < 1:
            parser.error("Interval must be at least 1 minute")
        elif args.interval > 1440:  # 24 hours
            parser.error("Interval cannot exceed 1440 minutes (24 hours)")
        interval = args.interval
    else:
        # Default to 1 minute if no interval specified
        interval = 1
    
    return currency, interval

async def main():
    try:
        currency, interval = parse_arguments()
        interval_text = f"{interval} minute" if interval == 1 else f"{interval} minutes"
        if interval >= 60:
            hours = interval // 60
            remaining_mins = interval % 60
            if remaining_mins == 0:
                interval_text = f"{hours} hour" if hours == 1 else f"{hours} hours"
            else:
                interval_text = f"{hours}h {remaining_mins}m"
        
        print(f"Starting Market Data Engine for {currency} with {interval_text} intervals")
        
        settings = Settings()
        exchange_api = DeribitAPI()
        vol_engine = VolatilityEngine()

        worker = MarketDataEngine(
            exchange_api=exchange_api,
            vol_engine=vol_engine,
            settings=settings,
            currency=currency,
        )

        await worker.run(interval_minutes=interval)
        
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())


# get_option_data is currently run twice, once in _fetch_instruments and once in _process_currency_updates
# this leads to a lot of redundant calls to the API and slows down the algorithm
# instead it should call the api periodically and use a websocket to get the updates.