import pytest
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from core.VolatilityEngine import VolPoints, VolatilityEngine
from data.utils.data_schemas import VolatilityPoint

@pytest.fixture
def sample_timestamp():
    return datetime(2024, 1, 1, 12, 0)

@pytest.fixture
def sample_vol_point(sample_timestamp):
    return VolatilityPoint(
        timestamp=sample_timestamp,
        strike=100.0,
        moneyness=1.0,
        expiry_date=sample_timestamp + timedelta(days=30),
        days_to_expiry=30,
        implied_vol=0.2,
        option_type="call",
        delta=0.5,
        gamma=0.02,
        vega=0.3,
        theta=-0.1,
        snapshot_id="test_snapshot"
    )

class TestVolPoints:
    def test_init(self, sample_timestamp):
        vol_points = VolPoints(sample_timestamp)
        assert vol_points.timestamp == sample_timestamp
        assert len(vol_points.vol_points) == 0

    def test_add_point(self, sample_timestamp, sample_vol_point):
        vol_points = VolPoints(sample_timestamp)
        vol_points.add_point(sample_vol_point)
        assert len(vol_points.vol_points) == 1
        assert vol_points.vol_points[0] == sample_vol_point

class TestVolatilityEngine:
    def test_init(self):
        engine = VolatilityEngine(min_points=10)
        assert engine.min_points == 10
        assert len(engine.surfaces_data) == 0

    def test_add_market_data(self, sample_timestamp, sample_vol_point):
        engine = VolatilityEngine()
        
        engine.add_market_data(
            timestamp=sample_timestamp,
            strike=sample_vol_point.strike,
            moneyness=sample_vol_point.moneyness,
            option_type=sample_vol_point.option_type,
            expiry_date=sample_vol_point.expiry_date,
            days_to_expiry=sample_vol_point.days_to_expiry,
            implied_vol=sample_vol_point.implied_vol,
            delta=sample_vol_point.delta,
            gamma=sample_vol_point.gamma,
            vega=sample_vol_point.vega,
            theta=sample_vol_point.theta,
            snapshot_id=sample_vol_point.snapshot_id
        )

        assert len(engine.surfaces_data) == 1
        assert sample_timestamp in engine.surfaces_data
        assert len(engine.surfaces_data[sample_timestamp].vol_points) == 1

    def test_get_volatility_surface(self, sample_timestamp):
        engine = VolatilityEngine()
        snapshot_id = "test_snapshot"
        asset_id = "TEST"

        # Add multiple data points
        for i, strike in enumerate([95.0, 100.0, 105.0]):
            engine.add_market_data(
                timestamp=sample_timestamp,
                strike=strike,
                moneyness=strike/100,
                option_type="call",
                expiry_date=sample_timestamp + timedelta(days=30+i),
                days_to_expiry=30+i,
                implied_vol=0.2 + i*0.02,
                snapshot_id=snapshot_id,
                asset_id=asset_id
            )

        vol_surface = engine.get_volatility_surface(snapshot_id, asset_id)
        assert vol_surface is not None
        assert len(vol_surface.strikes) == 3
        assert len(vol_surface.implied_vols) == 3
        assert all(isinstance(vol, float) for vol in vol_surface.implied_vols)

    def test_empty_surface_data(self):
        engine = VolatilityEngine()
        vol_surface = engine.get_volatility_surface("nonexistent_snapshot", "TEST")
        assert vol_surface is None 

    def test_get_skews(self, sample_timestamp):
        engine = VolatilityEngine(min_points=3)
        snapshot_id = "test_snapshot"

        # Add multiple points with different strikes for the same expiry
        strikes = [90.0, 100.0, 110.0]
        vols = [0.25, 0.20, 0.22]  # Typical smile pattern
        
        for strike, vol in zip(strikes, vols):
            engine.add_market_data(
                timestamp=sample_timestamp,
                strike=strike,
                moneyness=strike/100,
                option_type="call",
                expiry_date=sample_timestamp + timedelta(days=30),
                days_to_expiry=30,
                implied_vol=vol,
                snapshot_id=snapshot_id,
                asset_id="TEST"
            )

        surface = engine.get_volatility_surface(snapshot_id, "TEST")
        skew_data = engine.get_skews(surface)

        assert skew_data is not None
        assert len(skew_data) == 3
        assert all(skew_data.columns.isin(['days_to_expiry', 'strike', 'implied_vol', 'option_type', 'moneyness']))
        assert list(skew_data['strike'].values) == strikes
        assert list(skew_data['implied_vol'].values) == vols

    def test_calculate_term_structure(self, sample_timestamp):
        engine = VolatilityEngine()
        snapshot_id = "test_snapshot"

        # Add points with different expiries but similar moneyness (ATM)
        expiries = [30, 60, 90]
        vols = [0.20, 0.22, 0.23]  # Increasing term structure
        
        for days, vol in zip(expiries, vols):
            engine.add_market_data(
                timestamp=sample_timestamp,
                strike=100.0,
                moneyness=1.0,  # ATM
                option_type="call",
                expiry_date=sample_timestamp + timedelta(days=days),
                days_to_expiry=days,
                implied_vol=vol,
                snapshot_id=snapshot_id,
                asset_id="TEST"
            )

        surface = engine.get_volatility_surface(snapshot_id, "TEST")
        term_data = engine._get_term_structure(surface)

        assert term_data is not None
        assert len(term_data) == 3
        assert all(term_data.columns.isin(['days_to_expiry', 'atm_vol']))
        assert list(term_data['days_to_expiry'].values) == expiries
        assert list(term_data['atm_vol'].values) == vols

    def test_skews_insufficient_points(self, sample_timestamp):
        engine = VolatilityEngine(min_points=5)  # Require 5 points minimum
        snapshot_id = "test_snapshot"

        # Add only 2 points
        for strike in [95.0, 105.0]:
            engine.add_market_data(
                timestamp=sample_timestamp,
                strike=strike,
                moneyness=strike/100,
                option_type="call",
                expiry_date=sample_timestamp + timedelta(days=30),
                days_to_expiry=30,
                implied_vol=0.2,
                snapshot_id=snapshot_id,
                asset_id="TEST"
            )

        surface = engine.get_volatility_surface(snapshot_id, "TEST")
        skew_data = engine.get_skews(surface)
        assert skew_data is None  # Should return None due to insufficient points