import numpy as np
from scipy.interpolate import griddata
from enum import Enum
from typing import Optional
from data.utils.data_schemas import VolSurface

class SurfaceType(str, Enum):
    RAW = "raw"
    LINEAR = "linear"
    NEAREST = "nearest"
    HESTON = "heston"

def interpolate_surface(data: dict, method: SurfaceType) -> dict:
    """
    Helper function to interpolate volatility surface data.
    
    Args:
        data: Dictionary containing raw surface data (timestamp, moneyness, days_to_expiry, implied_vols)
        method: Interpolation method to use
    
    Returns:
        Dictionary with interpolated surface data
    """
    if method == SurfaceType.RAW:
        return data
        
    # Extract points and values
    points = np.array([(m, d) for m, d in zip(data["moneyness"], data["days_to_expiry"])])
    values = np.array(data["implied_vols"])
    
    # Create regular grid based on data density
    moneyness_points = len(np.unique(points[:,0]))
    dte_points = len(np.unique(points[:,1]))
    
    # Grid boundaries
    moneyness_min, moneyness_max = points[:,0].min(), points[:,0].max()
    dte_min, dte_max = points[:,1].min(), points[:,1].max()
    
    # Create mesh grid
    grid_moneyness = np.linspace(moneyness_min, moneyness_max, moneyness_points)
    grid_dte = np.linspace(dte_min, dte_max, dte_points)
    moneyness_mesh, dte_mesh = np.meshgrid(grid_moneyness, grid_dte)
    
    # Interpolate
    grid_implied_vols = griddata(
        points,
        values,
        (moneyness_mesh, dte_mesh),
        method=method.value,
        fill_value=np.nan
    )
    
    # Replace NaN values with None for JSON serialization
    grid_implied_vols = np.where(
        np.isfinite(grid_implied_vols),
        grid_implied_vols,
        None
    )
    
    return {
        "timestamp": data["timestamp"],
        "moneyness": grid_moneyness.tolist(),
        "days_to_expiry": grid_dte.tolist(),
        "implied_vols": grid_implied_vols.tolist(),
        "interpolation_method": method.value
    }

def filter_surface(
    surface: VolSurface,
    option_type: Optional[str] = None,
    min_dte: Optional[float] = None,
    max_dte: Optional[float] = None,
    min_moneyness: Optional[float] = None,
    max_moneyness: Optional[float] = None,
    ) -> VolSurface:
    """
    Filter a volatility surface by option type, DTE, and moneyness range.
    Prints statistics before and after filtering.

    Args:
        surface: Original VolSurface object
        option_type: 'c' for calls or 'p' for puts
        min_dte: Minimum days to expiry
        max_dte: Maximum days to expiry
        min_moneyness: Minimum moneyness
        max_moneyness: Maximum moneyness

    Returns:
        VolSurface: Filtered surface
    """
    if surface is None:
        return None

    # Print original statistics
    print("Original Surface Statistics:")
    print(f"Number of points: {len(surface.strikes)}")
    print(f"Option types: {set(surface.option_type)}")
    print(f"DTE range: [{min(surface.days_to_expiry)}, {max(surface.days_to_expiry)}]")
    print(f"Moneyness range: [{min(surface.moneyness)}, {max(surface.moneyness)}]")
    print("-" * 50)

    # Create mask for filtering
    mask = [True] * len(surface.strikes)

    if option_type:
        option_mask = [
            opt.lower() == option_type.lower() for opt in surface.option_type
        ]
        mask = [m and om for m, om in zip(mask, option_mask)]

    if min_dte is not None:
        dte_mask = [dte >= min_dte for dte in surface.days_to_expiry]
        mask = [m and dm for m, dm in zip(mask, dte_mask)]

    if max_dte is not None:
        dte_mask = [dte <= max_dte for dte in surface.days_to_expiry]
        mask = [m and dm for m, dm in zip(mask, dte_mask)]

    if min_moneyness is not None:
        mon_mask = [m >= min_moneyness for m in surface.moneyness]
        mask = [m and mm for m, mm in zip(mask, mon_mask)]

    if max_moneyness is not None:
        mon_mask = [m <= max_moneyness for m in surface.moneyness]
        mask = [m and mm for m, mm in zip(mask, mon_mask)]

    # Apply filtering
    filtered_surface = VolSurface(
        timestamp=surface.timestamp,
        method=surface.method,
        strikes=[s for s, m in zip(surface.strikes, mask) if m],
        moneyness=[s for s, m in zip(surface.moneyness, mask) if m],
        maturities=[s for s, m in zip(surface.maturities, mask) if m],
        days_to_expiry=[s for s, m in zip(surface.days_to_expiry, mask) if m],
        implied_vols=[s for s, m in zip(surface.implied_vols, mask) if m],
        option_type=[s for s, m in zip(surface.option_type, mask) if m],
        snapshot_id=surface.snapshot_id,
    )

    # Print filtered statistics
    print("Filtered Surface Statistics:")
    print(f"Number of points: {len(filtered_surface.strikes)}")
    print(f"Option types: {set(filtered_surface.option_type)}")
    print(
        f"DTE range: [{min(filtered_surface.days_to_expiry)}, {max(filtered_surface.days_to_expiry)}]"
    )
    print(
        f"Moneyness range: [{min(filtered_surface.moneyness)}, {max(filtered_surface.moneyness)}]"
    )

    return filtered_surface
