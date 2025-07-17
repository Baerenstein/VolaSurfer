from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, List, Set


@dataclass
class MarketState:
    last_update: datetime
    last_price: float
    active_instruments: Set[str]
    last_regime: Optional[str] = None
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
    volume: Optional[float] = None


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
    vega: Optional[float] = None
    theta: Optional[float] = None
    snapshot_id: Optional[str] = None
    asset_id: Optional[str] = None

    def __init__(
        self,
        timestamp: datetime,
        strike: float,
        moneyness: float,
        expiry_date: datetime,
        days_to_expiry: int,
        implied_vol: float,
        option_type: str,
        delta: Optional[float] = None,
        gamma: Optional[float] = None,
        vega: Optional[float] = None,
        theta: Optional[float] = None,
        snapshot_id: Optional[str] = None,
        asset_id: Optional[str] = None,
    ):
        self.timestamp = timestamp
        self.strike = strike
        self.moneyness = moneyness
        self.expiry_date = expiry_date
        self.days_to_expiry = days_to_expiry
        self.implied_vol = implied_vol
        self.option_type = option_type
        self.delta = delta
        self.gamma = gamma
        self.vega = vega
        self.theta = theta
        self.snapshot_id = snapshot_id
        self.asset_id = asset_id


@dataclass
class VolSurface:
    timestamp: datetime
    method: str
    strikes: List[float]
    moneyness: List[float]
    maturities: List[datetime]
    days_to_expiry: List[int]
    implied_vols: List[List[float]]
    option_type: List[str]
    snapshot_id: Optional[str] = None
    asset_id: Optional[str] = None
    spot_price: Optional[float] = None

    def __init__(self, timestamp, method, strikes, moneyness, maturities, days_to_expiry, implied_vols, option_type, snapshot_id, asset_id=None, spot_price=None):
        self.timestamp = timestamp
        self.method = method
        self.strikes = strikes
        self.moneyness = moneyness
        self.maturities = maturities
        self.days_to_expiry = days_to_expiry
        self.implied_vols = implied_vols
        self.option_type = option_type
        self.snapshot_id = snapshot_id
        self.asset_id = asset_id
        self.spot_price = spot_price

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
            "asset_id": self.asset_id,
            "spot_price": self.spot_price,
        }


@dataclass
class VolMetrics:
    timestamp: datetime
    avg_vol: float
    term_structure_slope: float
    put_call_skew: float
    wing_risk_metrics: Dict[str, float]
    historical_percentiles: Dict[str, float]