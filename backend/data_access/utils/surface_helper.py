import numpy as np
from scipy.interpolate import griddata
from enum import Enum

class InterpolationMethod(str, Enum):
    RAW = "raw"
    CUBIC = "cubic"
    NEAREST = "nearest"

def interpolate_surface(data: dict, method: InterpolationMethod) -> dict:
    """
    Helper function to interpolate volatility surface data.
    
    Args:
        data: Dictionary containing raw surface data (timestamp, moneyness, days_to_expiry, implied_vols)
        method: Interpolation method to use
    
    Returns:
        Dictionary with interpolated surface data
    """
    if method == InterpolationMethod.RAW:
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