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

    def test_get_interpolated_vol_inverse_distance(self, sample_timestamp):
        vol_points = VolPoints(sample_timestamp)
        
        # Add multiple points for interpolation
        points = [
            VolatilityPoint(
                timestamp=sample_timestamp,
                strike=strike,
                moneyness=strike/100,
                expiry_date=sample_timestamp + timedelta(days=days),
                days_to_expiry=days,
                implied_vol=vol,
                option_type="call"
            )
            for strike, days, vol in [
                (95.0, 25, 0.18),
                (100.0, 30, 0.20),
                (105.0, 35, 0.22)
            ]
        ]
        
        for point in points:
            vol_points.add_point(point)

        # Test interpolation
        interpolated_vol = vol_points.get_interpolated_vol(
            strike=100.0,
            expiry_date=sample_timestamp + timedelta(days=30),
            method="inverse_distance"
        )
        assert isinstance(interpolated_vol, float)
        assert 0.18 <= interpolated_vol <= 0.22

    def test_get_interpolated_vol_linear(self, sample_timestamp):
        vol_points = VolPoints(sample_timestamp)
        
        # Add multiple points for interpolation in a grid pattern
        test_points = []
        for strike in [95.0, 100.0, 105.0]:
            for days in [25, 30, 35]:
                # Create a more varied volatility surface
                base_vol = 0.20
                strike_adj = (strike - 100.0) * 0.001  # Small adjustment based on strike
                time_adj = (days - 30) * 0.001  # Small adjustment based on time
                vol = base_vol + strike_adj + time_adj
                
                test_points.append((strike, days, vol))
        
        # Create and add the points
        points = [
            VolatilityPoint(
                timestamp=sample_timestamp,
                strike=strike,
                moneyness=strike/100,
                expiry_date=sample_timestamp + timedelta(days=days),
                days_to_expiry=days,
                implied_vol=vol,
                option_type="call"
            )
            for strike, days, vol in test_points
        ]
        
        for point in points:
            vol_points.add_point(point)

        # Test interpolation at a point within the grid
        interpolated_vol = vol_points.get_interpolated_vol(
            strike=100.0,
            expiry_date=sample_timestamp + timedelta(days=30),
            method="linear"
        )
        
        assert interpolated_vol is not None, "Interpolation should not return None"
        assert isinstance(interpolated_vol, float), f"Expected float, got {type(interpolated_vol)}"
        # The interpolated value should be close to the base volatility
        assert abs(interpolated_vol - 0.20) < 0.01, f"Expected ~0.20, got {interpolated_vol}"

        # Test interpolation outside the grid should return None
        outside_grid_vol = vol_points.get_interpolated_vol(
            strike=200.0,  # Way outside the grid
            expiry_date=sample_timestamp + timedelta(days=30),
            method="linear"
        )
        assert outside_grid_vol is None, "Interpolation outside grid should return None"

    def test_get_interpolated_vol_cubic(self, sample_timestamp):
        vol_points = VolPoints(sample_timestamp)
        
        # Add multiple points in a grid pattern (need more points for cubic interpolation)
        test_points = []
        for strike in [90.0, 95.0, 100.0, 105.0, 110.0]:  # Need more points for cubic
            for days in [20, 25, 30, 35, 40]:  # Need more points for cubic
                # Create a more complex volatility surface suitable for cubic interpolation
                base_vol = 0.20
                strike_adj = (strike - 100.0) * 0.001  # Strike adjustment
                time_adj = (days - 30) * 0.001  # Time adjustment
                # Add some curvature to the surface
                curvature = 0.0001 * (strike - 100.0)**2 + 0.0001 * (days - 30)**2
                vol = base_vol + strike_adj + time_adj + curvature
                
                test_points.append((strike, days, vol))
        
        # Create and add the points
        points = [
            VolatilityPoint(
                timestamp=sample_timestamp,
                strike=strike,
                moneyness=strike/100,
                expiry_date=sample_timestamp + timedelta(days=days),
                days_to_expiry=days,
                implied_vol=vol,
                option_type="call"
            )
            for strike, days, vol in test_points
        ]
        
        for point in points:
            vol_points.add_point(point)

        # Test interpolation at a point within the grid
        interpolated_vol = vol_points.get_interpolated_vol(
            strike=100.0,
            expiry_date=sample_timestamp + timedelta(days=30),
            method="cubic"
        )
        
        assert interpolated_vol is not None, "Cubic interpolation should not return None"
        assert isinstance(interpolated_vol, float), f"Expected float, got {type(interpolated_vol)}"
        # The interpolated value should be close to the base volatility at the center
        assert abs(interpolated_vol - 0.20) < 0.01, f"Expected ~0.20, got {interpolated_vol}"

        # Test interpolation outside the grid should return None
        outside_grid_vol = vol_points.get_interpolated_vol(
            strike=200.0,  # Way outside the grid
            expiry_date=sample_timestamp + timedelta(days=30),
            method="cubic"
        )
        assert outside_grid_vol is None, "Interpolation outside grid should return None"

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
                snapshot_id=snapshot_id
            )

        vol_surface = engine.get_volatility_surface(snapshot_id)
        assert vol_surface is not None
        assert len(vol_surface.strikes) == 3
        assert len(vol_surface.implied_vols) == 3
        assert all(isinstance(vol, float) for vol in vol_surface.implied_vols)

    def test_empty_surface_data(self):
        engine = VolatilityEngine()
        vol_surface = engine.get_volatility_surface("nonexistent_snapshot")
        assert isinstance(vol_surface.strikes, list)
        assert isinstance(vol_surface.implied_vols, list)
        assert len(vol_surface.strikes) == 0
        assert len(vol_surface.implied_vols) == 0
