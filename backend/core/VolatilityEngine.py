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

    def __str__(self):
        """Return a string representation of the VolatilitySurface contents."""
        return f"VolatilitySurface(timestamp={self.timestamp}, vol_points={self.vol_points})"


class VolatilityEngine:
    def __init__(
        self,
        min_points: int = 10,
        length: int = 100,
    ):
        self.min_points = min_points
        self.length = length
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
        asset_id: Optional[str] = None,
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
            asset_id=asset_id,
        )

        # Add the point to the corresponding VolPoints instance
        self.surfaces_data[timestamp].add_point(vol_point)

    # VolSurface TODO: add option type parameter
    def get_volatility_surface(self, snapshot_id: str, asset_id: str) -> Optional[VolSurface]:
        """
        Create a VolSurface object containing implied volatilities with strikes and days to expiry.

        :param snapshot_id: The snapshot_id to create the VolSurface from.
        :param asset_id: The asset_id to create the VolSurface from.
        :return: A VolSurface object representing the volatility surface
        """
        print(f"Getting volatility surface for snapshot_id: {snapshot_id}")
        print(f"Number of surfaces in data: {len(self.surfaces_data)}")
        
        filtered_vol_points = []
        for timestamp, vol_points in self.surfaces_data.items():
            print(f"Processing timestamp {timestamp} with {len(vol_points.vol_points)} points")
            matching_points = [
                point
                for point in vol_points.vol_points
                if point.snapshot_id == snapshot_id
            ]
            print(f"Found {len(matching_points)} matching points for this timestamp")
            filtered_vol_points.extend(matching_points)

        print(f"Total filtered points: {len(filtered_vol_points)}")
        
        if not filtered_vol_points:
            print("WARNING: No points found for this snapshot_id")
            return None

        # Sort filtered_vol_points by days_to_expiry and then by moneyness
        filtered_vol_points.sort(
            key=lambda point: (point.days_to_expiry, point.moneyness)
        )

        # Create lists of data points
        strikes = [point.strike for point in filtered_vol_points]
        moneyness = [point.moneyness for point in filtered_vol_points]
        maturities = [point.expiry_date for point in filtered_vol_points]
        days_to_expiry = [point.days_to_expiry for point in filtered_vol_points]
        implied_vols = [point.implied_vol for point in filtered_vol_points]
        option_types = [point.option_type for point in filtered_vol_points]

        print(f"Data ranges:")
        print(f"Strikes: {min(strikes)} to {max(strikes)}")
        print(f"Days to expiry: {min(days_to_expiry)} to {max(days_to_expiry)}")
        print(f"Implied vols: {min(implied_vols)} to {max(implied_vols)}")

        # Add after filtering points
        asset_id = filtered_vol_points[0].asset_id if filtered_vol_points else None

        vol_surface = VolSurface(
            timestamp=datetime.now(),
            method="computed", 
            strikes=strikes,
            moneyness=moneyness,
            maturities=maturities,
            days_to_expiry=days_to_expiry,
            implied_vols=implied_vols,
            option_type=option_types,
            snapshot_id=snapshot_id,
            asset_id=asset_id,
        )
        print("Successfully created VolSurface object")
        return vol_surface

    def get_skews(self, surface: VolSurface) -> pd.DataFrame:
        """Get the volatility skews for each expiry.

        :param surface: VolSurface object containing the volatility surface data
        :return: A DataFrame containing the skew data or None if insufficient data
        """
        print("Calculating skews for all expiries.")
        if not surface:
            print("Surface is None, returning None.")
            return None

        # Debug: Print available days to expiry
        unique_expiries = sorted(set(surface.days_to_expiry))
        print(f"Available days to expiry in surface: {unique_expiries}")

        # Group data by expiry
        all_data = []
        for expiry in unique_expiries:
            # Get indices for this expiry
            indices = [i for i, dte in enumerate(surface.days_to_expiry) if dte == expiry]
            
            if len(indices) < self.min_points:
                print(f"Skipping expiry {expiry} - insufficient points: {len(indices)} < {self.min_points}")
                continue

            # Create DataFrame for this expiry
            expiry_data = pd.DataFrame({
                "days_to_expiry": expiry,
                "strike": [surface.strikes[i] for i in indices],
                "implied_vol": [surface.implied_vols[i] for i in indices],
                "option_type": [surface.option_type[i] for i in indices],
                "moneyness": [surface.moneyness[i] for i in indices]
            })
            
            all_data.append(expiry_data)

        if not all_data:
            print("No valid skews found for any expiry.")
            return None

        # Combine all expiry data
        skew_data = pd.concat(all_data, ignore_index=True)
        print(f"{datetime.now()}: Combined skew data shape: {skew_data.shape}")
        
        return skew_data.sort_values(["days_to_expiry", "strike"])

    def _get_term_structure(self, surface: VolSurface) -> pd.DataFrame:
        """Calculate ATM volatility term structure.

        :param surface: VolSurface object containing the volatility surface data
        :return: A DataFrame containing the term structure data or None if insufficient data.
        """
        print("Calculating term structure.")
        if not surface:
            return None

        # Group by expiry and get ATM vols (using moneyness close to 1.0)
        term_structure = defaultdict(list)
        
        # Zip all relevant data together for processing
        for dte, mon, vol in zip(surface.days_to_expiry, surface.moneyness, surface.implied_vols):
            # Consider a point ATM if moneyness is between 0.95 and 1.05
            if 0.95 <= mon <= 1.05:
                term_structure[dte].append(vol)

        # Average vols for each expiry
        term_data = {"days_to_expiry": [], "atm_vol": []}

        for dte, vols in sorted(term_structure.items()):
            term_data["days_to_expiry"].append(dte)
            term_data["atm_vol"].append(np.mean(vols))
            
        print(f"{datetime.now()}: Term data: {term_data}.")

        return pd.DataFrame(term_data)

    def get_implied_volatility_index(self, surface: VolSurface, target_expiry_days: int = 30) -> float:
        """Get the implied volatility index for the surface.

        Args:
            surface: VolSurface object containing the volatility surface data
            target_expiry_days: Target number of days to expiry (default: 30)

        Returns:
            float: The implied volatility index, or None if surface is invalid
        """
        print("Calculating implied volatility index.")
        if not surface: 
            print("Surface is None, returning None.")
            return None

        # Create sorted data structure
        data = list(zip(
            surface.days_to_expiry,
            surface.moneyness,
            surface.implied_vols,
            surface.vegas if hasattr(surface, 'vegas') else [1.0] * len(surface.implied_vols)
        ))
        data.sort(key=lambda x: (x[0], x[1]))  # Sort by days_to_expiry, then moneyness
        
        # Unzip sorted data
        days_to_expiry, moneyness, implied_vols, vegas = zip(*data)
        
        # Find contracts near target expiry
        target_expiry_indices = [
            i for i, days in enumerate(days_to_expiry)
            if abs(days - target_expiry_days) < 5
        ]

        # Find ATM contracts (moneyness close to 1.0)
        atm_indices = [
            i for i, m in enumerate(moneyness)
            if abs(m - 1.0) < 0.01
        ]

        if not target_expiry_indices or not atm_indices:
            print("Insufficient data for index calculation.")
            return None

        # Calculate vega-weighted averages
        atm_volatility = np.average(
            [implied_vols[i] for i in atm_indices],
            weights=[vegas[i] for i in atm_indices]
        )

        target_expiry_volatility = np.average(
            [implied_vols[i] for i in target_expiry_indices],
            weights=[vegas[i] for i in target_expiry_indices]
        )

        # Calculate volatility index
        volatility_index = atm_volatility - target_expiry_volatility

        return volatility_index

    def get_surface_metrics(self, surface: VolSurface) -> Dict:
        """Calculate key surface metrics.

        Args:
            surface: VolSurface object containing the volatility surface data

        Returns:
            Dict: A dictionary containing various metrics of the volatility surface
        """
        print("Calculating surface metrics.")
        if not surface:
            return {}

        metrics = {
            "timestamp": surface.timestamp,
            "num_points": len(surface.strikes),
            "avg_vol": np.mean(surface.implied_vols),
            "min_vol": min(surface.implied_vols),
            "max_vol": max(surface.implied_vols),
            "avg_skew": None,
            "term_structure_slope": None,
        }

        # Calculate average skew (using moneyness instead of delta)
        skew_data = self.get_skews(surface)
        if skew_data is not None:
            # Use moneyness instead of delta for ATM and wing calculations
            atm_vol = skew_data[skew_data["moneyness"].between(0.95, 1.05)]["implied_vol"].mean()
            wing_vol = skew_data[skew_data["moneyness"].between(0.75, 0.85)]["implied_vol"].mean()
            if not (np.isnan(atm_vol) or np.isnan(wing_vol)):
                metrics["avg_skew"] = wing_vol - atm_vol

        # Calculate term structure slope
        term_data = self._get_term_structure(surface)
        if term_data is not None and len(term_data) > 1:
            slope, _ = np.polyfit(term_data["days_to_expiry"], term_data["atm_vol"], 1)
            metrics["term_structure_slope"] = slope

        return metrics

    def get_latest_snapshot_id(self) -> Optional[str]:
        """
        Fetches the latest snapshot_id from the most recent timestamp.

        :return: The latest snapshot_id or None if no data is available.
        """
        if not self.surfaces_data:
            return None
        return self.surfaces_data[list(self.surfaces_data.keys())[-1]].vol_points[0].snapshot_id