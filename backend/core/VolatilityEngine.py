from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from collections import defaultdict
from data.utils.data_schemas import VolatilityPoint, VolSurface


class VolPoints:
    def __init__(self, timestamp: datetime):
        """Initialize a VolatilitySurface instance.

        :param timestamp: The timestamp for the volatility surface.
        """
        # print(f"Initializing VolatilitySurface with timestamp: {timestamp}")
        self.timestamp = timestamp
        self.vol_points: List[VolatilityPoint] = []

    def add_point(self, point: VolatilityPoint):
        """Add a volatility point to the surface.

        :param point: The VolatilityPoint to add.
        """
        # print(f"Adding point: {point}")
        self.vol_points.append(point)

    def get_interpolated_vol(
        self, strike: float, expiry_date: datetime, method: str = "inverse_distance"
    ) -> float:
        """Get the interpolated volatility with multiple interpolation methods.

        :param strike: The strike price for which to interpolate volatility.
        :param expiry_date: The expiry_date date for which to interpolate volatility.
        :param method: Interpolation method ('inverse_distance', 'linear', 'cubic')
        :return: The interpolated implied volatility or None if not available.
        """
        if not self.vol_points:
            return None

        if method == "inverse_distance":
            # Existing inverse distance method
            distances = []
            for point in self.vol_points:
                strike_dist = abs(point.strike - strike)
                time_dist = abs((point.expiry_date - expiry_date).days)
                total_dist = np.sqrt(strike_dist**2 + time_dist**2)
                distances.append((total_dist, point.implied_vol))

            # Weight by inverse distance
            total_weight = 0
            weighted_vol = 0
            for dist, vol in sorted(distances)[:4]:
                if dist == 0:
                    return vol
                weight = 1 / dist
                total_weight += weight
                weighted_vol += weight * vol

            return weighted_vol / total_weight if total_weight > 0 else None

        elif method == "linear":
            # Prepare data
            strikes = [p.strike for p in self.vol_points]
            maturities = [
                (p.expiry_date - datetime.now()).days for p in self.vol_points
            ]
            vols = [p.implied_vol for p in self.vol_points]

            # Target point
            target_expiry_date = (expiry_date - datetime.now()).days

            # Interpolate
            return griddata(
                (strikes, maturities),
                vols,
                (strike, target_expiry_date),
                method="linear",
            )

        elif method == "cubic":
            # Prepare data
            strikes = [p.strike for p in self.vol_points]
            maturities = [
                (p.expiry_date - datetime.now()).days for p in self.vol_points
            ]
            vols = [p.implied_vol for p in self.vol_points]

            # Target point
            target_expiry_date = (expiry_date - datetime.now()).days

            # Interpolate
            return griddata(
                (strikes, maturities),
                vols,
                (strike, target_expiry_date),
                method="cubic",
            )

        else:
            raise ValueError(f"Unsupported interpolation method: {method}")

    def __str__(self):
        """Return a string representation of the VolatilitySurface contents."""
        return f"VolatilitySurface(timestamp={self.timestamp}, vol_points={self.vol_points})"


class VolatilityEngine:
    def __init__(
        self,
        min_points: int = 10,
        # window_size: int = 100,
        # expiry_ranges: List[int] = [7, 14, 30, 60, 90],
        # moneyness_ranges: Tuple[float, float] = (0.8, 1.2),
    ):
        self.min_points = min_points
        self.surfaces_data: Dict[datetime, VolPoints] = {}

        # self.latest_surface: Optional[VolSurface] = None

    def add_market_data(
        self,
        timestamp: datetime,
        strike: float,
        moneyness: float,
        option_type: str,
        expiry_date: datetime,
        days_to_expiry: int,
        implied_vol: float,
        delta: Optional[float] = None,
        gamma: Optional[float] = None,
        vega: Optional[float] = None,
        theta: Optional[float] = None,
        snapshot_id: Optional[str] = None,
    ):
        # Check if the timestamp already exists
        if timestamp not in self.surfaces_data:
            self.surfaces_data[timestamp] = VolPoints(timestamp)
            # print(f"New timestamp added: {timestamp}")

        vol_point = VolatilityPoint(
            timestamp=timestamp,
            strike=strike,
            moneyness=moneyness,
            expiry_date=expiry_date,
            days_to_expiry=days_to_expiry,
            implied_vol=implied_vol,
            option_type=option_type,
            delta=delta,
            gamma=gamma,
            vega=vega,
            theta=theta,
            snapshot_id=snapshot_id,
        )

        # Add the point to the corresponding VolPoints instance
        self.surfaces_data[timestamp].add_point(vol_point)

    # VolSurface TODO: add option type parameter
    def get_latest_volatility_surface(self, snapshot_id: str) -> Optional[VolSurface]:
        """
        Create a VolSurface object containing implied volatilities with strikes and days to expiry.

        :param snapshot_id: The snapshot_id to create the VolSurface from.
        :return: A VolSurface object representing the volatility surface
        """
        print(f"{datetime.now()}: surfaces_data length: {len(self.surfaces_data)}.")

        filtered_vol_points = []
        for (
            vol_points
        ) in self.surfaces_data.values():  # Iterate through VolPoints instances
            filtered_vol_points.extend(
                point
                for point in vol_points.vol_points
                if point.snapshot_id == snapshot_id
            )
        # print(f"{datetime.now()}: Filtered vol points length: {len(filtered_vol_points)}.")

        # Sort filtered_vol_points by days_to_expiry and then by moneyness
        filtered_vol_points.sort(
            key=lambda point: (point.days_to_expiry, point.moneyness)
        )

        # Create a list of strikes and days to expiry
        strikes = [point.strike for point in filtered_vol_points]
        moneyness = [point.moneyness for point in filtered_vol_points]
        maturities = [point.expiry_date for point in filtered_vol_points]
        days_to_expiry = [(point.days_to_expiry) for point in filtered_vol_points]
        implied_vols = [point.implied_vol for point in filtered_vol_points]
        option_types = [point.option_type for point in filtered_vol_points]


        vol_surface = VolSurface(
            timestamp=datetime.now(),  # or use surface.timestamp
            method="computed", 
            strikes=strikes,
            moneyness=moneyness,
            maturities=maturities,
            days_to_expiry=days_to_expiry,
            implied_vols=implied_vols,
            option_type=option_types,
            snapshot_id=snapshot_id,
        )
        return vol_surface

    def get_current_skew(self, expiry_days: int = 30) -> pd.DataFrame:
        """Get the current volatility skew for a specific expiry.

        :param expiry_days: The number of days to expiry for the skew calculation.
        :return: A DataFrame containing the skew data or None if insufficient data.
        """
        print(f"Calculating current skew for expiry_days={expiry_days}.")
        if not self.latest_surface:
            return None

        target_date = datetime.now() + timedelta(days=expiry_days)

        # Filter points near target expiry
        relevant_points = [
            p
            for p in self.latest_surface.vol_points
            if abs((p.expiry_date - target_date).days) <= 5  # 5-day tolerance
        ]

        if len(relevant_points) < self.min_points:
            return None

        # maybe add timestamp for indexing?
        skew_data = pd.DataFrame(
            [
                {
                    "strike": p.strike,
                    "implied_vol": p.implied_vol,
                    "option_type": p.option_type,
                    "delta": p.delta,
                    "gamma": p.gamma,
                    "vega": p.vega,
                }
                for p in relevant_points
            ]
        )
        print(f"{datetime.now()}: Skew data: {skew_data}.")
        return skew_data.sort_values("strike")

    def calculate_term_structure(self) -> pd.DataFrame:
        """Calculate ATM volatility term structure.

        :return: A DataFrame containing the term structure data or None if insufficient data.
        """
        print("Calculating term structure.")
        if not self.latest_surface:
            return None

        # Group by expiry and get ATM vols
        term_structure = defaultdict(list)

        for point in self.latest_surface.vol_points:
            days_to_expiry = (point.expiry_date - datetime.now()).days
            term_structure[days_to_expiry].append(point.implied_vol)

        # Average vols for each expiry
        term_data = {"days_to_expiry": [], "atm_vol": []}

        for dte, vols in sorted(term_structure.items()):
            term_data["days_to_expiry"].append(dte)
            term_data["atm_vol"].append(np.mean(vols))
        print(f"{datetime.now()}: Term data: {term_data}.")

        return pd.DataFrame(term_data)

    def get_surface_metrics(self) -> Dict:
        """Calculate key surface metrics.

        :return: A dictionary containing various metrics of the volatility surface.
        """
        print("Calculating surface metrics.")
        if not self.latest_surface:
            return {}
        print(f"{datetime.now()}: Latest surface: {self.latest_surface}.")
        print(f"{datetime.now()}: Vol points: {self.latest_surface.vol_points}.")
        print(
            f"{datetime.now()}: Vol points length: {len(self.latest_surface.vol_points)}."
        )
        print(
            f"{datetime.now()}: Vol points implied vols: {[p.implied_vol for p in self.latest_surface.vol_points]}."
        )
        metrics = {
            "timestamp": self.latest_surface.timestamp,
            "num_points": len(self.latest_surface.vol_points),
            "avg_vol": np.mean([p.implied_vol for p in self.latest_surface.vol_points]),
            "min_vol": min(p.implied_vol for p in self.latest_surface.vol_points),
            "max_vol": max(p.implied_vol for p in self.latest_surface.vol_points),
            "avg_skew": None,  # Calculated below
            "term_structure_slope": None,  # Calculated below
        }

        # Calculate average skew (ATM to 25D)
        skew_data = self.get_current_skew()
        if skew_data is not None:
            atm_vol = skew_data[skew_data["delta"].abs().between(0.45, 0.55)][
                "implied_vol"
            ].mean()
            wing_vol = skew_data[skew_data["delta"].abs().between(0.2, 0.3)][
                "implied_vol"
            ].mean()
            metrics["avg_skew"] = wing_vol - atm_vol

        # Calculate term structure slope
        term_data = self.calculate_term_structure()
        if term_data is not None and len(term_data) > 1:
            slope, _ = np.polyfit(term_data["days_to_expiry"], term_data["atm_vol"], 1)
            metrics["term_structure_slope"] = slope

        return metrics
