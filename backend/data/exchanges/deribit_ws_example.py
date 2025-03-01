from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
import asyncio
import aiohttp
import json
import websockets

@dataclass
class OptionContract:
    symbol: str
    underlying: str
    strike: float
    expiry: datetime
    option_type: str
    bid: float
    ask: float
    last_price: float
    open_interest: float
    volume: Optional[float] = None
    implied_volatility: Optional[float] = None
    last_update: datetime = datetime.now()

@dataclass
class OptionsFilter:
    strike_min: Optional[float] = None
    strike_max: Optional[float] = None
    expiry_min: Optional[datetime] = None
    expiry_max: Optional[datetime] = None
    option_types: Optional[List[str]] = None

    def __post_init__(self):
        if self.option_types:
            valid_types = {'CALL', 'PUT'}
            self.option_types = [ot.upper() for ot in self.option_types]
            if not all(ot in valid_types for ot in self.option_types):
                raise ValueError("option_types must be 'CALL' or 'PUT'")

    def match_contract_params(self, strike: float, expiry: datetime, option_type: str) -> bool:
        strike_match = (not self.strike_min or strike >= self.strike_min) and \
                      (not self.strike_max or strike <= self.strike_max)
        
        expiry_match = (not self.expiry_min or expiry >= self.expiry_min) and \
                      (not self.expiry_max or expiry <= self.expiry_max)
        
        type_match = not self.option_types or option_type in self.option_types
        return strike_match and expiry_match and type_match

class BaseOptionsDataStreamer(ABC):
    def __init__(self):
        self.contracts: Dict[str, OptionContract] = {}
        self.strikes_by_expiry: Dict[str, Dict[datetime, List[float]]] = {}
        self.active_underlyings: List[str] = []
    
    @abstractmethod
    async def fetch_contracts(self, underlying: str, options_filter: Optional[OptionsFilter] = None):
        pass
    
    @abstractmethod
    async def start_streaming(self, underlyings: List[str], options_filter: Optional[OptionsFilter] = None):
        pass
    
    @abstractmethod
    async def close(self):
        pass
    
    def get_contracts_by_expiry(self, underlying: str, expiry: datetime) -> List[OptionContract]:
        return [contract for contract in self.contracts.values() 
                if contract.underlying == underlying and contract.expiry == expiry]
    
    def get_contracts_by_strike(self, underlying: str, strike: float) -> List[OptionContract]:
        return [contract for contract in self.contracts.values() 
                if contract.underlying == underlying and contract.strike == strike]
    
    def get_strikes_for_expiry(self, underlying: str, expiry: datetime) -> List[float]:
        return self.strikes_by_expiry.get(underlying, {}).get(expiry, [])
    
    def get_all_expiries(self, underlying: str) -> List[datetime]:
        return sorted(self.strikes_by_expiry.get(underlying, {}).keys())
    
    def get_active_underlyings(self) -> List[str]:
        return self.active_underlyings

class BinanceOptionsDataStreamer(BaseOptionsDataStreamer):
    def __init__(self):
        super().__init__()
        self.ws = None
        self.base_rest_url = "https://eapi.binance.com/eapi/v1"
        self.ws_url = "wss://nbstream.binance.com/eoptions/stream"
        self.request_window = 1  
        self.max_requests_per_window = 50  
        self.requests_made = 0
        self.last_request_time = datetime.now()
        self.last_ticker_log = datetime.now()
        print(f"Initialized BinanceOptionsDataStreamer")

    async def _wait_for_rate_limit(self):
        current_time = datetime.now()
        time_passed = (current_time - self.last_request_time).total_seconds()
        
        if time_passed < self.request_window:
            if self.requests_made >= self.max_requests_per_window:
                sleep_time = self.request_window - time_passed
                await asyncio.sleep(sleep_time)
                self.requests_made = 0
                self.last_request_time = datetime.now()
        else:
            self.requests_made = 0
            self.last_request_time = current_time

    async def fetch_contracts(self, underlying: str, options_filter: Optional[OptionsFilter] = None):
        print(f"Fetching contracts for {underlying} with filter: {options_filter}")
        async with aiohttp.ClientSession() as session:
            await self._wait_for_rate_limit()
            self.requests_made += 1
            url = f"{self.base_rest_url}/exchangeInfo"
            async with session.get(url) as response:
                data = await response.json()
                
                for symbol_data in data['optionSymbols']:
                    if symbol_data['underlying'] == underlying:
                        expiry = datetime.fromtimestamp(symbol_data['expiryDate']/1000)
                        strike = float(symbol_data['strikePrice'])
                        option_type = 'CALL' if symbol_data['symbol'].endswith('-C') else 'PUT'

                        if options_filter and not options_filter.match_contract_params(strike, expiry, option_type):
                            continue
                        
                        if underlying not in self.strikes_by_expiry:
                            self.strikes_by_expiry[underlying] = {}
                        if expiry not in self.strikes_by_expiry[underlying]:
                            self.strikes_by_expiry[underlying][expiry] = []
                            
                        self.strikes_by_expiry[underlying][expiry].append(strike)
                        
                        print(f"Fetching ticker for {symbol_data['symbol']}")
                        await self._wait_for_rate_limit()
                        self.requests_made += 1
                        ticker_url = f"{self.base_rest_url}/ticker"
                        async with session.get(ticker_url, params={'symbol': symbol_data['symbol']}) as ticker_response:
                            ticker = await ticker_response.json()
                            
                            self.contracts[symbol_data['symbol']] = OptionContract(
                                symbol=symbol_data['symbol'],
                                underlying=underlying,
                                strike=strike,
                                expiry=expiry,
                                option_type=option_type,
                                bid=float(ticker[0]['bidPrice']),
                                ask=float(ticker[0]['askPrice']), 
                                last_price=float(ticker[0]['lastPrice']),
                                open_interest=float(ticker[0]['volume']),
                                volume=float(ticker[0]['volume'])
                            )

        for expiry in self.strikes_by_expiry[underlying]:
            self.strikes_by_expiry[underlying][expiry] = sorted(set(self.strikes_by_expiry[underlying][expiry]))
        print(f"Fetched {len(self.contracts)} contracts for {underlying}")
    
    async def market_data_handler(self, msg):
        data = json.loads(msg)
        if 'stream' not in data:
            return
            
        market_data = data['data']
        symbol = market_data['s']
        
        if symbol in self.contracts:
            contract = self.contracts[symbol]
            contract.bid = float(market_data['b'])
            contract.ask = float(market_data['a'])
            contract.last_price = float(market_data['c'])
            contract.volume = float(market_data['V'])
            contract.last_update = datetime.now()
            print(f"{symbol}: bid={contract.bid}, ask={contract.ask}, oi={contract.open_interest}, volume={contract.volume}")
            
    
    async def start_streaming(self, underlyings: List[str], options_filter: Optional[OptionsFilter] = None):
        print(f"Starting stream for {underlyings}")
        
        for underlying in underlyings:
            if underlying not in self.active_underlyings:
                self.active_underlyings.append(underlying)
                await self.fetch_contracts(underlying, options_filter)
        
        streams = [f"{symbol}@ticker" for symbol in self.contracts.keys()]
        print(f"Connecting to WebSocket with {len(streams)} streams")
        combined_streams = "/".join(streams)
        ws_url = f"{self.ws_url}?streams={combined_streams}"
        print(f"WebSocket URL: {ws_url}")
        self.ws = await websockets.connect(ws_url)
        print("WebSocket connection established")
        
        while True:
            try:
                message = await self.ws.recv()
                await self.market_data_handler(message)

            except Exception as e:
                print(f"WebSocket error: {e}")
                await asyncio.sleep(1)
                try:
                    print("Attempting to reconnect...")
                    self.ws = await websockets.connect(ws_url)
                    print("Successfully reconnected")
                except Exception as e:
                    print(f"Reconnection failed: {e}")
    
    async def close(self):
        if self.ws:
            print("Closing WebSocket connection")
            await self.ws.close()
            print("WebSocket connection closed")
            
class DeribitOptionsDataStreamer(BaseOptionsDataStreamer):
    def __init__(self):
        super().__init__()
        self.ws = None
        self.base_rest_url = "https://www.deribit.com/api/v2"
        self.ws_url = "wss://www.deribit.com/ws/api/v2"
        self.request_window = 1
        self.max_requests_per_window = 50
        self.requests_made = 0
        self.last_request_time = datetime.now()
        
    async def _wait_for_rate_limit(self):
        current_time = datetime.now()
        time_passed = (current_time - self.last_request_time).total_seconds()
        
        if time_passed < self.request_window:
            if self.requests_made >= self.max_requests_per_window:
                sleep_time = self.request_window - time_passed
                await asyncio.sleep(sleep_time)
                self.requests_made = 0
                self.last_request_time = datetime.now()
        else:
            self.requests_made = 0
            self.last_request_time = current_time

    async def fetch_contracts(self, underlying: str, options_filter: Optional[OptionsFilter] = None):
        currency = underlying.replace("USDT", "").replace("USD", "")
        async with aiohttp.ClientSession() as session:
            await self._wait_for_rate_limit()
            self.requests_made += 1
            url = f"{self.base_rest_url}/public/get_instruments"
            params = {
                "currency": currency,
                "kind": "option",
                "expired": "false"
            }
            
            async with session.get(url, params=params) as response:
                data = await response.json()
                instruments = data["result"]
                
                with open('data/deribit_get_instruments.json', 'w') as f:
                    json.dump(data, f, indent=4)
                
                for instrument in instruments:
                    expiry = datetime.fromtimestamp(instrument["expiration_timestamp"] / 1000)
                    strike = float(instrument["strike"])
                    option_type = instrument["option_type"].upper()
                    
                    if options_filter and not options_filter.match_contract_params(strike, expiry, option_type):
                        continue
                    
                    if underlying not in self.strikes_by_expiry:
                        self.strikes_by_expiry[underlying] = {}
                    if expiry not in self.strikes_by_expiry[underlying]:
                        self.strikes_by_expiry[underlying][expiry] = []
                        
                    self.strikes_by_expiry[underlying][expiry].append(strike)
                    
                    await self._wait_for_rate_limit()
                    self.requests_made += 1
                    ticker_url = f"{self.base_rest_url}/public/ticker"
                    ticker_params = {"instrument_name": instrument["instrument_name"]}
                    
                    async with session.get(ticker_url, params=ticker_params) as ticker_response:
                        ticker = await ticker_response.json()
                        ticker_data = ticker["result"]
                        
                        self.contracts[instrument["instrument_name"]] = OptionContract(
                            symbol=instrument["instrument_name"],
                            underlying=underlying,
                            strike=strike,
                            expiry=expiry,
                            option_type=option_type,
                            bid=float(ticker_data["bid_price"] or 0),
                            ask=float(ticker_data["ask_price"] or 0),
                            last_price=float(ticker_data["last_price"] or 0),
                            open_interest=float(ticker_data["open_interest"] or 0),
                            volume=float(ticker_data["volume"] or 0),
                            implied_volatility=float(ticker_data["mark_iv"] or 0)
                        )

        for expiry in self.strikes_by_expiry[underlying]:
            self.strikes_by_expiry[underlying][expiry] = sorted(set(self.strikes_by_expiry[underlying][expiry]))

    async def market_data_handler(self, msg):
        if msg.get("method") == "subscription" and msg.get("params", {}).get("channel", "").startswith("ticker"):
            data = msg["params"]["data"]
            instrument_name = data["instrument_name"]
            
            if instrument_name in self.contracts:
                contract = self.contracts[instrument_name]
                contract.bid = float(data["best_bid_price"] or 0)
                contract.ask = float(data["best_ask_price"] or 0)
                contract.last_price = float(data["last_price"] or 0)
                contract.volume = float(data["stats"]["volume"] or 0)
                contract.open_interest = float(data["open_interest"] or 0)
                contract.implied_volatility = float(data.get("mark_iv", 0) or 0)
                contract.last_update = datetime.now()

    async def start_streaming(self, underlyings: List[str], options_filter: Optional[OptionsFilter] = None):
        for underlying in underlyings:
            if underlying not in self.active_underlyings:
                self.active_underlyings.append(underlying)
                await self.fetch_contracts(underlying, options_filter)
        
        channels = [f"ticker.{symbol}.raw" for symbol in self.contracts.keys()]
        subscribe_msg = {
            "jsonrpc": "2.0",
            "method": "public/subscribe",
            "params": {
                "channels": channels
            },
            "id": 1
        }
        
        self.ws = await websockets.connect(self.ws_url)
        await self.ws.send(json.dumps(subscribe_msg))
        
        while True:
            try:
                message = await self.ws.recv()
                await self.market_data_handler(json.loads(message))
            except Exception as e:
                await asyncio.sleep(1)
                try:
                    self.ws = await websockets.connect(self.ws_url)
                    await self.ws.send(json.dumps(subscribe_msg))
                except Exception:
                    continue

    async def close(self):
        if self.ws:
            await self.ws.close()

async def main():
    options_filter = OptionsFilter(
        strike_min=1800.0,
        strike_max=2000.0,
        expiry_min=datetime.now(),
        expiry_max=datetime.now().replace(month=datetime.now().month + 1),
        option_types=['CALL']
    )
        
    # streamer = BinanceOptionsDataStreamer()
    streamer = DeribitOptionsDataStreamer()
    await streamer.start_streaming(['ETHUSDT'], options_filter)
    
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        await streamer.close()

if __name__ == "__main__":
    asyncio.run(main())