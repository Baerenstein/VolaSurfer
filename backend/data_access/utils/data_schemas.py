from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, List, Set


@dataclass
class MarketState:
    last_update: datetime
    last_price: float
    active_instruments: Set[str]
    last_regime: Optional[str] = None  # what regime is meant by this?
    is_market_open: bool = True
    error_count: int = 0


@dataclass
class OptionContract:
    timestamp: datetime
    asset_id: str
    base_currency: str
    symbol: str
    expiry_date: datetime
    days_to_expiry: int
    strike: float
    moneyness: float
    option_type: str
    last_price: float
    implied_vol: float
    bid_price: Optional[float] = None
    ask_price: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    vega: Optional[float] = None
    theta: Optional[float] = None
    open_interest: Optional[float] = None
    snapshot_id: Optional[str] = None

    def to_dict(self):
        return {
            "timestamp": self.timestamp.isoformat(),
            "base_currency": self.base_currency,
            "symbol": self.symbol,
            "expiry_date": self.expiry_date.isoformat(),
            "days_to_expiry": self.days_to_expiry,
            "strike": self.strike,
            "moneyness": self.moneyness,
            "option_type": self.option_type,
            "last_price": self.last_price,
            "implied_vol": self.implied_vol,
            "bid_price": self.bid_price,
            "ask_price": self.ask_price,
            "delta": self.delta,
            "gamma": self.gamma,
            "vega": self.vega,
            "theta": self.theta,
            "open_interest": self.open_interest,
            "snapshot_id": self.snapshot_id,
        }


@dataclass
class UnderlyingAsset:
    symbol: str
    last_price: float
    timestamp: datetime
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume_24h: Optional[float] = None


# TODO
@dataclass
class VolPoint:
    strike: float
    expiry_date: datetime
    moneyness: float
    forward: float
    implied_vol: float
    bid_vol: Optional[float]
    ask_vol: Optional[float]
    volume: Optional[int]
    open_interest: Optional[int]
    greeks: Dict[str, float]
    vol_error: Optional[float]


# TODO
@dataclass
class VolatilityPoint:
    strike: float
    moneyness: float
    expiry_date: datetime
    days_to_expiry: int
    implied_vol: float
    option_type: str
    timestamp: datetime
    delta: Optional[float] = None
    gamma: Optional[float] = None
    vega: Optional[float] = (None,)
    theta: Optional[float] = (None,)
    snapshot_id: Optional[str] = None


@dataclass
class VolSurface:
    timestamp: datetime
    method: str
    strikes: List[float]  # List of strike prices
    moneyness: List[float]
    maturities: List[datetime]  # List of expiry_date dates
    days_to_expiry: List[int]
    implied_vols: List[List[float]]
    option_type: List[str]
    snapshot_id: Optional[str] = None

    def to_dict(self):
        return {
            "timestamp": self.timestamp.isoformat(),
            "method": self.method,
            "strikes": self.strikes,
            "moneyness": self.moneyness,
            "maturities": [maturity.isoformat() for maturity in self.maturities],
            "days_to_expiry": self.days_to_expiry,
            "implied_vols": self.implied_vols,
            "snapshot_id": self.snapshot_id,
        }


# TODO
@dataclass
class VolMetrics:
    timestamp: datetime
    avg_vol: float
    term_structure_slope: float
    put_call_skew: float
    wing_risk_metrics: Dict[str, float]
    historical_percentiles: Dict[str, float]


# TODO refactor to use dataclass
# from dataclasses import dataclass
# from datetime import datetime
# from typing import Optional, Dict, List, Union

# # Base Metadata Class
# @dataclass
# class BaseMetadata:
#     timestamp: datetime
#     snapshot_id: Optional[str] = None

# # Separate Greeks Management
# @dataclass
# class Greeks:
#     delta: Optional[float] = None
#     gamma: Optional[float] = None
#     vega: Optional[float] = None
#     theta: Optional[float] = None

# # Pricing Information
# @dataclass
# class PricingInfo:
#     last_price: float
#     bid_price: Optional[float] = None
#     ask_price: Optional[float] = None

# # Specialized Option Contract
# @dataclass
# class OptionContract(BaseMetadata, PricingInfo, Greeks):
#     symbol: str
#     expiry_date: datetime
#     strike: float
#     option_type: str
#     base_currency: str
#     days_to_expiry: int
#     moneyness: float
#     implied_vol: float
#     open_interest: Optional[float] = None

# # Volatility Point with Modular Design
# @dataclass
# class VolPoint(BaseMetadata, Greeks):
#     strike: float
#     expiry_date: datetime
#     moneyness: float
#     forward: float
#     implied_vol: float
#     bid_vol: Optional[float] = None
#     ask_vol: Optional[float] = None
#     volume: Optional[int] = None
#     open_interest: Optional[int] = None
#     vol_error: Optional[float] = None

# # Advanced Volatility Surface
# @dataclass
# class VolSurface(BaseMetadata):
#     method: str
#     data: Dict[str, Union[List[float], List[datetime], List[List[float]]]] = None

# # Enhanced Volatility Metrics
# @dataclass
# class VolMetrics(BaseMetadata):
#     avg_vol: float
#     term_structure_slope: float
#     metrics: Dict[str, Union[float, Dict[str, float]]] = None
