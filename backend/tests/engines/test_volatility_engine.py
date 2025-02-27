import pytest
from datetime import datetime, timedelta
from core.VolatilityEngine import VolatilityEngine
from data.utils.data_schemas import VolatilityPoint


@pytest.fixture
def timestamp():
    """Fixture providing a consistent timestamp for tests."""
    return datetime.now()


@pytest.fixture
def volatility_engine():
    """Fixture to create a VolatilityEngine instance for testing."""
    return VolatilityEngine()


@pytest.fixture
def basic_surface(volatility_engine, timestamp):
    """Fixture providing a surface with basic market data."""
    # Add single point for basic tests
    volatility_engine.add_market_data(
        strike=100,
        expiry_date=timestamp + timedelta(days=30),
        implied_vol=0.2,
        option_type="call",
        timestamp=timestamp,
    )
    return volatility_engine


@pytest.fixture
def surface_with_skew(volatility_engine, timestamp):
    """Fixture providing a surface with skew data."""
    for i in range(10):
        if i < 3:  # Wing points (25 delta)
            delta = 0.25
            implied_vol = 0.25  # Higher vol for wings
        elif i < 6:  # ATM points
            delta = 0.5
            implied_vol = 0.2  # Lower vol for ATM
        else:  # Other points
            delta = 0.8 - i * 0.1
            implied_vol = 0.2 + i * 0.01

        volatility_engine.add_market_data(
            strike=90 + i * 10,
            expiry_date=timestamp + timedelta(days=30),
            implied_vol=implied_vol,
            option_type="call",
            timestamp=timestamp,
            greeks={"delta": delta},
        )
    return volatility_engine


@pytest.fixture
def surface_with_term_structure(volatility_engine, timestamp):
    """Fixture providing a surface with term structure data."""
    for i in range(10):
        volatility_engine.add_market_data(
            strike=100,  # ATM strike
            expiry_date=timestamp + timedelta(days=30 * (i + 1)),
            implied_vol=0.15 + i * 0.01,
            option_type="call",
            timestamp=timestamp,
            greeks={"delta": 0.5},  # ATM delta
        )
    return volatility_engine


@pytest.fixture
def full_surface(volatility_engine, timestamp):
    """Fixture providing a surface with both skew and term structure data."""
    # Add skew points
    for i in range(10):
        if i < 3:
            delta = 0.25
            implied_vol = 0.25
        elif i < 6:
            delta = 0.5
            implied_vol = 0.2
        else:
            delta = 0.8 - i * 0.1
            implied_vol = 0.2 + i * 0.01

        volatility_engine.add_market_data(
            strike=90 + i * 10,
            expiry_date=timestamp + timedelta(days=30),
            implied_vol=implied_vol,
            option_type="call",
            timestamp=timestamp,
            greeks={"delta": delta},
        )

        # Add term structure points
        volatility_engine.add_market_data(
            strike=100,
            expiry_date=timestamp + timedelta(days=30 * (i + 1)),
            implied_vol=0.15 + i * 0.01,
            option_type="call",
            timestamp=timestamp,
            greeks={"delta": 0.5},
        )
    return volatility_engine


def test_add_market_data(volatility_engine):
    """Test adding market data to the VolatilityEngine."""
    timestamp = datetime.now()
    volatility_engine.add_market_data(
        timestamp=timestamp,
        strike=100,
        moneyness=1.0,
        option_type="call",
        expiry_date=timestamp + timedelta(days=30),
        days_to_expiry=30,
        implied_vol=0.2,
        delta=0.5,
        gamma=0.1,
        vega=0.2,
        theta=0.01,
        snapshot_id="test_snapshot"
    )
    assert len(volatility_engine.surfaces_data) == 1
    assert volatility_engine.surfaces_data[timestamp].vol_points[0].strike == 100
    assert volatility_engine.surfaces_data[timestamp].vol_points[0].implied_vol == 0.2


def test_get_latest_volatility_surface(volatility_engine):
    """Test retrieving the latest volatility surface."""
    timestamp = datetime.now()
    volatility_engine.add_market_data(
        timestamp=timestamp,
        strike=100,
        moneyness=1.0,
        option_type="call",
        expiry_date=timestamp + timedelta(days=30),
        days_to_expiry=30,
        implied_vol=0.2,
        delta=0.5,
        gamma=0.1,
        vega=0.2,
        theta=0.01,
        snapshot_id="test_snapshot"
    )
    surface = volatility_engine.get_latest_volatility_surface(snapshot_id="test_snapshot")
    assert surface is not None
    assert surface.strikes[0] == 100
    assert surface.implied_vols[0][0] == 0.2  # Check the implied vol in the surface


def test_calculate_surface_metrics(volatility_engine):
    """Test calculating surface metrics."""
    timestamp = datetime.now()
    for i in range(5):
        volatility_engine.add_market_data(
            timestamp=timestamp,
            strike=100 + i * 10,
            moneyness=1.0 + i * 0.1,
            option_type="call",
            expiry_date=timestamp + timedelta(days=30),
            days_to_expiry=30,
            implied_vol=0.2 + i * 0.01,
            delta=0.5,
            gamma=0.1,
            vega=0.2,
            theta=0.01,
            snapshot_id="test_snapshot"
        )
    metrics = volatility_engine.get_surface_metrics()
    assert metrics is not None
    assert "avg_skew" in metrics
    assert "term_structure_slope" in metrics
    assert metrics["num_points"] == 5  # Check the number of points added
    assert metrics["avg_vol"] == pytest.approx(0.205, rel=1e-2)  # Check average vol


def test_get_volatility_surface(surface_with_skew):
    """Test retrieving the volatility surface."""
    strikes, expiries, vols = surface_with_skew.get_volatility_surface()
    assert strikes is not None
    assert expiries is not None
    assert vols is not None
    assert len(strikes) == 10
    assert len(expiries) == 1  # Only one expiry in skew data


def test_get_current_skew(surface_with_skew):
    """Test calculating the current skew."""
    skew_data = surface_with_skew.get_current_skew()
    assert skew_data is not None
    assert len(skew_data) >= 10
    assert "strike" in skew_data.columns
    assert "implied_vol" in skew_data.columns


def test_calculate_term_structure(surface_with_term_structure):
    """Test calculating the term structure."""
    term_structure = surface_with_term_structure.calculate_term_structure()
    assert term_structure is not None
    assert len(term_structure) == 10
    assert "days_to_expiry" in term_structure.columns
    assert "atm_vol" in term_structure.columns
    assert list(term_structure["days_to_expiry"]) == sorted(
        term_structure["days_to_expiry"]
    )


def test_get_surface_metrics(full_surface):
    """Test calculating surface metrics."""
    metrics = full_surface.get_surface_metrics()

    assert metrics is not None
    assert "timestamp" in metrics
    assert metrics["num_points"] == 20
    assert 0.15 <= metrics["min_vol"] <= 0.3
    assert 0.15 <= metrics["max_vol"] <= 0.3
    assert 0.15 <= metrics["avg_vol"] <= 0.3

    assert metrics["avg_skew"] is not None
    assert metrics["avg_skew"] > 0

    assert metrics["term_structure_slope"] is not None
    assert metrics["term_structure_slope"] > 0


class TestVolatilityEngine:
    """Test suite for VolatilityEngine class."""

    class TestGetVolatilitySurface:
        """Tests for get_volatility_surface method."""

        def test_with_basic_surface(self, basic_surface):
            """Test with minimal data."""
            strikes, expiries, vols = basic_surface.get_volatility_surface()
            assert strikes is None  # Should return None with < min_points
            assert expiries is None
            assert vols is None

        def test_with_skew_surface(self, surface_with_skew):
            """Test with skew data."""
            strikes, expiries, vols = surface_with_skew.get_volatility_surface()
            assert strikes is not None
            assert len(strikes) == 10  # 10 different strikes
            assert len(expiries) == 1  # Single expiry
            assert vols.shape == (10, 1)

        def test_with_term_structure(self, surface_with_term_structure):
            """Test with term structure data."""
            strikes, expiries, vols = (
                surface_with_term_structure.get_volatility_surface()
            )
            assert strikes is not None
            assert len(strikes) == 1  # Single strike
            assert len(expiries) == 10  # 10 different expiries
            assert vols.shape == (1, 10)

        def test_with_full_surface(self, full_surface):
            """Test with complete surface data."""
            strikes, expiries, vols = full_surface.get_volatility_surface()
            assert strikes is not None
            assert len(strikes) == 10
            assert len(expiries) == 10
            assert vols.shape == (10, 10)

    class TestGetCurrentSkew:
        """Tests for get_current_skew method."""

        def test_with_basic_surface(self, basic_surface):
            """Test with insufficient data."""
            skew = basic_surface.get_current_skew()
            assert skew is None

        def test_with_term_structure(self, surface_with_term_structure):
            """Test with only ATM data."""
            skew = surface_with_term_structure.get_current_skew()
            # Term structure data has points spread across different maturities,
            # so we expect None since there aren't enough points near any single expiry
            assert skew is None

        def test_with_skew_surface(self, surface_with_skew):
            """Test with proper skew data."""
            skew = surface_with_skew.get_current_skew()
            assert skew is not None
            assert len(skew) >= 10
            # Verify we have both wing and ATM points
            deltas = skew["delta"].abs()
            assert any(deltas.between(0.2, 0.3))  # Wing points
            assert any(deltas.between(0.45, 0.55))  # ATM points

    class TestCalculateTermStructure:
        """Tests for calculate_term_structure method."""

        def test_with_basic_surface(self, basic_surface):
            """Test with single point."""
            term = basic_surface.calculate_term_structure()
            assert term is not None
            assert len(term) == 1

        def test_with_skew_surface(self, surface_with_skew):
            """Test with single expiry."""
            term = surface_with_skew.calculate_term_structure()
            assert term is not None
            assert len(term) == 1

        def test_with_term_structure(self, surface_with_term_structure):
            """Test with proper term structure data."""
            term = surface_with_term_structure.calculate_term_structure()
            assert term is not None
            assert len(term) == 10
            assert term["days_to_expiry"].is_monotonic_increasing

    class TestGetSurfaceMetrics:
        """Tests for get_surface_metrics method."""

        def test_with_basic_surface(self, basic_surface):
            """Test with minimal data."""
            metrics = basic_surface.get_surface_metrics()
            assert metrics != {}
            assert metrics["num_points"] == 1
            assert metrics["avg_skew"] is None
            assert metrics["term_structure_slope"] is None

        def test_with_skew_surface(self, surface_with_skew):
            """Test with skew data only."""
            metrics = surface_with_skew.get_surface_metrics()
            assert metrics["avg_skew"] is not None
            assert metrics["term_structure_slope"] is None

        def test_with_term_structure(self, surface_with_term_structure):
            """Test with term structure only."""
            metrics = surface_with_term_structure.get_surface_metrics()
            assert metrics["avg_skew"] is None
            assert metrics["term_structure_slope"] is not None

        def test_with_full_surface(self, full_surface):
            """Test with complete surface data."""
            metrics = full_surface.get_surface_metrics()
            assert metrics["avg_skew"] is not None
            assert metrics["term_structure_slope"] is not None
            assert metrics["num_points"] == 20

        def test_edge_cases(self, volatility_engine):
            """Test edge cases with empty surface."""
            metrics = volatility_engine.get_surface_metrics()
            assert metrics == {}
