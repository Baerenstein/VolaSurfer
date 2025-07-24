from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import griddata, RBFInterpolator
from scipy.stats import norm
import logging
from dataclasses import dataclass

from data.utils.data_schemas import VolSurface, VolatilityPoint
from data.storage.base_store import BaseStore
from infrastructure.utils.logging import setup_logger


@dataclass
class CalibrationMetrics:
    """Metrics for evaluating calibration performance"""
    rmse: float
    mae: float
    max_error: float
    r_squared: float
    calibration_time: float
    num_points: int
    fit_quality: str
    residuals: List[float]
    parameter_values: Dict[str, float]
    timestamp: datetime


@dataclass
class SurfaceCalibrationResult:
    """Result of surface calibration including fitted parameters and metrics"""
    surface: VolSurface
    metrics: CalibrationMetrics
    fitted_surface: np.ndarray
    model_parameters: Dict[str, float]
    interpolation_grid: Tuple[np.ndarray, np.ndarray]
    success: bool
    message: str


class CalibrationEngine:
    """
    Enhanced calibration engine for volatility surfaces with database integration,
    performance analysis, and comprehensive metrics calculation.
    """
    
    def __init__(self, store: BaseStore, min_points: int = 20):
        self.store = store
        self.min_points = min_points
        self.logger = setup_logger("calibration_engine")
        
        # Calibration parameters
        self.calibration_methods = {
            'svi': self._calibrate_svi,
            'ssvi': self._calibrate_ssvi,
            'sabr': self._calibrate_sabr,
            'spline': self._calibrate_spline,
            'rbf': self._calibrate_rbf
        }
        
    def calibrate_surface_from_db(
        self, 
        asset_id: str, 
        snapshot_id: Optional[str] = None,
        method: str = 'svi',
        min_moneyness: float = 0.7,
        max_moneyness: float = 1.3,
        min_dte: int = 7,
        max_dte: int = 365
    ) -> SurfaceCalibrationResult:
        """
        Calibrate volatility surface directly from database data.
        
        Args:
            asset_id: Asset identifier
            snapshot_id: Specific snapshot ID, if None uses latest
            method: Calibration method ('svi', 'ssvi', 'sabr', 'spline', 'rbf')
            min_moneyness: Minimum moneyness filter
            max_moneyness: Maximum moneyness filter
            min_dte: Minimum days to expiry filter
            max_dte: Maximum days to expiry filter
            
        Returns:
            SurfaceCalibrationResult with fitted surface and metrics
        """
        start_time = datetime.now()
        self.logger.info(f"Starting surface calibration for asset {asset_id} using {method}")
        
        try:
            # Load data from database
            vol_data = self._load_volatility_data(
                asset_id=asset_id,
                snapshot_id=snapshot_id,
                min_moneyness=min_moneyness,
                max_moneyness=max_moneyness,
                min_dte=min_dte,
                max_dte=max_dte
            )
            
            if len(vol_data) < self.min_points:
                return SurfaceCalibrationResult(
                    surface=None,
                    metrics=None,
                    fitted_surface=None,
                    model_parameters={},
                    interpolation_grid=(None, None),
                    success=False,
                    message=f"Insufficient data points: {len(vol_data)} < {self.min_points}"
                )
            
            # Perform calibration
            calibration_func = self.calibration_methods.get(method, self._calibrate_svi)
            result = calibration_func(vol_data)
            
            # Calculate performance metrics
            calibration_time = (datetime.now() - start_time).total_seconds()
            metrics = self._calculate_calibration_metrics(
                vol_data, 
                result['fitted_vols'], 
                result['parameters'],
                calibration_time
            )
            
            # Create volatility surface
            surface = self._create_vol_surface(vol_data, result, asset_id, snapshot_id)
            
            # Store calibration results
            self._store_calibration_results(asset_id, snapshot_id, result, metrics)
            
            self.logger.info(f"Calibration completed successfully. RMSE: {metrics.rmse:.4f}")
            
            return SurfaceCalibrationResult(
                surface=surface,
                metrics=metrics,
                fitted_surface=result['fitted_surface'],
                model_parameters=result['parameters'],
                interpolation_grid=result['grid'],
                success=True,
                message=f"Calibration successful using {method}"
            )
            
        except Exception as e:
            self.logger.error(f"Calibration failed: {str(e)}")
            return SurfaceCalibrationResult(
                surface=None,
                metrics=None,
                fitted_surface=None,
                model_parameters={},
                interpolation_grid=(None, None),
                success=False,
                message=f"Calibration failed: {str(e)}"
            )
    
    def _load_volatility_data(
        self, 
        asset_id: str, 
        snapshot_id: Optional[str] = None,
        min_moneyness: float = 0.7,
        max_moneyness: float = 1.3,
        min_dte: int = 7,
        max_dte: int = 365
    ) -> pd.DataFrame:
        """Load and filter volatility data from database"""
        
        # If no snapshot_id provided, get the latest one
        if snapshot_id is None:
            snapshot_id = self.store.get_latest_snapshot_id(asset_id)
            
        # Get options data from database
        options_df = self.store.get_options_by_snapshot(snapshot_id, asset_id)
        
        if options_df is None or options_df.empty:
            raise ValueError(f"No options data found for asset {asset_id}, snapshot {snapshot_id}")
        
        # Filter data
        filtered_df = options_df[
            (options_df['moneyness'] >= min_moneyness) &
            (options_df['moneyness'] <= max_moneyness) &
            (options_df['days_to_expiry'] >= min_dte) &
            (options_df['days_to_expiry'] <= max_dte) &
            (options_df['implied_vol'] > 0) &
            (options_df['implied_vol'] < 5.0)  # Remove outliers
        ]
        
        return filtered_df
    
    def _calibrate_svi(self, vol_data: pd.DataFrame) -> Dict[str, Any]:
        """Calibrate SVI (Stochastic Volatility Inspired) model"""
        
        def svi_formula(k, a, b, rho, m, sigma):
            """SVI parametrization: w = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))"""
            return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))
        
        def objective(params, k_values, w_observed):
            a, b, rho, m, sigma = params
            w_fitted = svi_formula(k_values, a, b, rho, m, sigma)
            return np.sum((w_fitted - w_observed)**2)
        
        # Convert to log-moneyness and variance
        k_values = np.log(vol_data['moneyness'])
        w_observed = vol_data['implied_vol']**2 * vol_data['days_to_expiry'] / 365.0
        
        # Initial guess and bounds for SVI parameters
        bounds = [
            (0.01, 1.0),     # a: level
            (0.01, 2.0),     # b: angle
            (-0.99, 0.99),   # rho: correlation
            (-1.0, 1.0),     # m: shift
            (0.01, 2.0)      # sigma: smoothness
        ]
        
        # Optimize
        result = differential_evolution(
            objective, 
            bounds, 
            args=(k_values, w_observed),
            seed=42,
            maxiter=1000
        )
        
        a, b, rho, m, sigma = result.x
        
        # Generate fitted surface
        k_grid = np.linspace(k_values.min(), k_values.max(), 50)
        dte_grid = np.linspace(vol_data['days_to_expiry'].min(), vol_data['days_to_expiry'].max(), 30)
        K_grid, T_grid = np.meshgrid(k_grid, dte_grid)
        
        fitted_variance = svi_formula(K_grid, a, b, rho, m, sigma)
        fitted_surface = np.sqrt(fitted_variance / (T_grid / 365.0))
        
        # Calculate fitted vols for metrics
        fitted_vols = np.sqrt(svi_formula(k_values, a, b, rho, m, sigma) / (vol_data['days_to_expiry'] / 365.0))
        
        return {
            'parameters': {'a': a, 'b': b, 'rho': rho, 'm': m, 'sigma': sigma},
            'fitted_surface': fitted_surface,
            'fitted_vols': fitted_vols,
            'grid': (np.exp(K_grid), T_grid),
            'method': 'svi'
        }
    
    def _calibrate_ssvi(self, vol_data: pd.DataFrame) -> Dict[str, Any]:
        """Calibrate Surface SVI model"""
        # Simplified SSVI implementation
        # Group by expiry and fit SVI per slice, then interpolate
        
        grouped = vol_data.groupby('days_to_expiry')
        slice_results = {}
        
        for dte, group in grouped:
            if len(group) >= 5:  # Minimum points per slice
                svi_result = self._calibrate_svi(group)
                slice_results[dte] = svi_result
        
        if not slice_results:
            raise ValueError("No valid expiry slices for SSVI calibration")
        
        # Simple interpolation across time for now
        dte_values = list(slice_results.keys())
        param_evolution = {param: [] for param in ['a', 'b', 'rho', 'm', 'sigma']}
        
        for dte in dte_values:
            params = slice_results[dte]['parameters']
            for param, value in params.items():
                param_evolution[param].append(value)
        
        # Create interpolated surface
        k_grid = np.linspace(np.log(vol_data['moneyness']).min(), np.log(vol_data['moneyness']).max(), 50)
        dte_grid = np.linspace(vol_data['days_to_expiry'].min(), vol_data['days_to_expiry'].max(), 30)
        K_grid, T_grid = np.meshgrid(k_grid, dte_grid)
        
        # Simplified: use average parameters (should be improved with proper SSVI)
        avg_params = {param: np.mean(values) for param, values in param_evolution.items()}
        
        def svi_formula(k, a, b, rho, m, sigma):
            return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))
        
        fitted_variance = svi_formula(K_grid, **avg_params)
        fitted_surface = np.sqrt(fitted_variance / (T_grid / 365.0))
        
        k_values = np.log(vol_data['moneyness'])
        fitted_vols = np.sqrt(svi_formula(k_values, **avg_params) / (vol_data['days_to_expiry'] / 365.0))
        
        return {
            'parameters': avg_params,
            'fitted_surface': fitted_surface,
            'fitted_vols': fitted_vols,
            'grid': (np.exp(K_grid), T_grid),
            'method': 'ssvi'
        }
    
    def _calibrate_sabr(self, vol_data: pd.DataFrame) -> Dict[str, Any]:
        """Calibrate SABR model"""
        
        def sabr_vol(F, K, T, alpha, beta, rho, nu):
            """SABR volatility formula"""
            if F == K:
                return alpha * (F**(beta-1))
            
            z = (nu / alpha) * ((F * K)**((1-beta)/2)) * np.log(F/K)
            x_z = np.log((np.sqrt(1 - 2*rho*z + z**2) + z - rho) / (1 - rho))
            
            vol = (alpha / ((F * K)**((1-beta)/2))) * (z / x_z)
            vol *= (1 + ((1-beta)**2 / 24) * (np.log(F/K))**2 + 
                   ((1-beta)**4 / 1920) * (np.log(F/K))**4)
            vol *= (1 + (((1-beta)**2 * alpha**2) / (24 * (F*K)**(1-beta)) + 
                        (rho * beta * nu * alpha) / (4 * (F*K)**((1-beta)/2)) + 
                        ((2-3*rho**2) * nu**2) / 24) * T)
            
            return vol
        
        def objective(params, F_values, K_values, T_values, observed_vols):
            alpha, beta, rho, nu = params
            try:
                fitted_vols = np.array([
                    sabr_vol(F, K, T, alpha, beta, rho, nu) 
                    for F, K, T in zip(F_values, K_values, T_values)
                ])
                return np.sum((fitted_vols - observed_vols)**2)
            except:
                return 1e6
        
        # Prepare data
        spot_price = vol_data['strike'].mean() / vol_data['moneyness'].mean()  # Approximate
        F_values = np.full(len(vol_data), spot_price)
        K_values = vol_data['strike'].values
        T_values = vol_data['days_to_expiry'].values / 365.0
        observed_vols = vol_data['implied_vol'].values
        
        # SABR parameter bounds
        bounds = [
            (0.01, 2.0),     # alpha
            (0.1, 1.0),      # beta
            (-0.99, 0.99),   # rho
            (0.01, 2.0)      # nu
        ]
        
        result = differential_evolution(
            objective,
            bounds,
            args=(F_values, K_values, T_values, observed_vols),
            seed=42
        )
        
        alpha, beta, rho, nu = result.x
        
        # Generate fitted surface
        k_grid = np.linspace(K_values.min(), K_values.max(), 50)
        t_grid = np.linspace(T_values.min(), T_values.max(), 30)
        K_grid, T_grid = np.meshgrid(k_grid, t_grid)
        
        fitted_surface = np.array([
            [sabr_vol(spot_price, k, t, alpha, beta, rho, nu) 
             for k in k_grid] for t in t_grid
        ])
        
        fitted_vols = np.array([
            sabr_vol(F, K, T, alpha, beta, rho, nu) 
            for F, K, T in zip(F_values, K_values, T_values)
        ])
        
        return {
            'parameters': {'alpha': alpha, 'beta': beta, 'rho': rho, 'nu': nu},
            'fitted_surface': fitted_surface,
            'fitted_vols': fitted_vols,
            'grid': (K_grid, T_grid * 365),
            'method': 'sabr'
        }
    
    def _calibrate_spline(self, vol_data: pd.DataFrame) -> Dict[str, Any]:
        """Calibrate using spline interpolation"""
        from scipy.interpolate import RectBivariateSpline
        
        # Prepare data
        moneyness = vol_data['moneyness'].values
        dte = vol_data['days_to_expiry'].values
        vols = vol_data['implied_vol'].values
        
        # Create regular grid
        m_unique = np.unique(moneyness)
        t_unique = np.unique(dte)
        
        if len(m_unique) < 4 or len(t_unique) < 4:
            raise ValueError("Insufficient unique points for spline interpolation")
        
        # Interpolate to regular grid first
        M_grid, T_grid = np.meshgrid(m_unique, t_unique)
        vol_grid = griddata(
            (moneyness, dte), vols, 
            (M_grid, T_grid), 
            method='linear',
            fill_value=np.nan
        )
        
        # Remove NaN values
        mask = ~np.isnan(vol_grid)
        if mask.sum() < len(m_unique) * len(t_unique) * 0.5:
            raise ValueError("Too many missing values for spline interpolation")
        
        # Fit spline
        spline = RectBivariateSpline(t_unique, m_unique, vol_grid, kx=3, ky=3)
        
        # Generate fitted surface
        m_grid = np.linspace(moneyness.min(), moneyness.max(), 50)
        t_grid = np.linspace(dte.min(), dte.max(), 30)
        fitted_surface = spline(t_grid, m_grid)
        
        # Calculate fitted values for metrics
        fitted_vols = np.array([spline(t, m)[0, 0] for m, t in zip(moneyness, dte)])
        
        return {
            'parameters': {'smoothing': 0.0, 'kx': 3, 'ky': 3},
            'fitted_surface': fitted_surface,
            'fitted_vols': fitted_vols,
            'grid': (np.meshgrid(m_grid, t_grid)),
            'method': 'spline'
        }
    
    def _calibrate_rbf(self, vol_data: pd.DataFrame) -> Dict[str, Any]:
        """Calibrate using Radial Basis Function interpolation"""
        
        # Prepare data
        points = np.column_stack([vol_data['moneyness'], vol_data['days_to_expiry']])
        values = vol_data['implied_vol'].values
        
        # Fit RBF
        rbf = RBFInterpolator(points, values, kernel='thin_plate_spline')
        
        # Generate fitted surface
        m_grid = np.linspace(vol_data['moneyness'].min(), vol_data['moneyness'].max(), 50)
        t_grid = np.linspace(vol_data['days_to_expiry'].min(), vol_data['days_to_expiry'].max(), 30)
        M_grid, T_grid = np.meshgrid(m_grid, t_grid)
        
        grid_points = np.column_stack([M_grid.ravel(), T_grid.ravel()])
        fitted_surface = rbf(grid_points).reshape(M_grid.shape)
        
        # Calculate fitted values for metrics
        fitted_vols = rbf(points)
        
        return {
            'parameters': {'kernel': 'thin_plate_spline'},
            'fitted_surface': fitted_surface,
            'fitted_vols': fitted_vols,
            'grid': (M_grid, T_grid),
            'method': 'rbf'
        }
    
    def _calculate_calibration_metrics(
        self, 
        vol_data: pd.DataFrame, 
        fitted_vols: np.ndarray,
        parameters: Dict[str, float],
        calibration_time: float
    ) -> CalibrationMetrics:
        """Calculate comprehensive calibration performance metrics"""
        
        observed_vols = vol_data['implied_vol'].values
        residuals = fitted_vols - observed_vols
        
        # Basic error metrics
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))
        max_error = np.max(np.abs(residuals))
        
        # R-squared
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((observed_vols - np.mean(observed_vols))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Fit quality assessment
        if rmse < 0.01:
            fit_quality = "Excellent"
        elif rmse < 0.02:
            fit_quality = "Good"
        elif rmse < 0.05:
            fit_quality = "Fair"
        else:
            fit_quality = "Poor"
        
        return CalibrationMetrics(
            rmse=rmse,
            mae=mae,
            max_error=max_error,
            r_squared=r_squared,
            calibration_time=calibration_time,
            num_points=len(observed_vols),
            fit_quality=fit_quality,
            residuals=residuals.tolist(),
            parameter_values=parameters,
            timestamp=datetime.now()
        )
    
    def _create_vol_surface(
        self, 
        vol_data: pd.DataFrame, 
        result: Dict[str, Any], 
        asset_id: str, 
        snapshot_id: str
    ) -> VolSurface:
        """Create VolSurface object from calibration results"""
        
        # Use original data points for the surface
        return VolSurface(
            timestamp=datetime.now(),
            method=result['method'],
            strikes=vol_data['strike'].tolist(),
            moneyness=vol_data['moneyness'].tolist(),
            maturities=vol_data['expiry_date'].tolist(),
            days_to_expiry=vol_data['days_to_expiry'].tolist(),
            implied_vols=result['fitted_vols'].tolist(),
            option_type=vol_data['option_type'].tolist(),
            snapshot_id=snapshot_id,
            asset_id=asset_id,
            spot_price=vol_data['strike'].mean() / vol_data['moneyness'].mean()
        )
    
    def _store_calibration_results(
        self, 
        asset_id: str, 
        snapshot_id: str, 
        result: Dict[str, Any], 
        metrics: CalibrationMetrics
    ):
        """Store calibration results in database"""
        try:
            # Store in model_parameters table
            calibration_data = {
                'method': result['method'],
                'parameters': result['parameters'],
                'metrics': {
                    'rmse': metrics.rmse,
                    'mae': metrics.mae,
                    'r_squared': metrics.r_squared,
                    'fit_quality': metrics.fit_quality,
                    'calibration_time': metrics.calibration_time
                },
                'timestamp': metrics.timestamp.isoformat()
            }
            
            # Store calibration results using the store interface
            self.store.store_calibration_results(asset_id, snapshot_id, calibration_data)
            self.logger.info(f"Stored calibration results for asset {asset_id}")
            
        except Exception as e:
            self.logger.warning(f"Failed to store calibration results: {str(e)}")
    
    def analyze_calibration_performance(
        self, 
        asset_id: str, 
        lookback_days: int = 30
    ) -> Dict[str, Any]:
        """
        Analyze calibration performance over time for an asset.
        
        Returns comprehensive performance analysis including trends,
        stability metrics, and recommendations.
        """
        
        try:
            # Get historical calibration data from database
            calibration_history = self.store.get_calibration_history(asset_id, lookback_days)
            
            if not calibration_history:
                return {
                    'asset_id': asset_id,
                    'analysis_period': lookback_days,
                    'error': 'No calibration history found',
                    'last_updated': datetime.now().isoformat()
                }
            
            # Extract metrics from history
            rmse_values = []
            calibration_times = []
            methods_used = []
            
            for entry in calibration_history:
                if 'calibration_data' in entry and entry['calibration_data']:
                    calib_data = entry['calibration_data']
                    if 'metrics' in calib_data:
                        metrics = calib_data['metrics']
                        if 'rmse' in metrics:
                            rmse_values.append(metrics['rmse'])
                        if 'calibration_time' in metrics:
                            calibration_times.append(metrics['calibration_time'])
                        
                if 'method' in entry:
                    methods_used.append(entry['method'])
            
            # Calculate performance metrics
            avg_rmse = np.mean(rmse_values) if rmse_values else 0.0
            rmse_std = np.std(rmse_values) if rmse_values else 0.0
            avg_calibration_time = np.mean(calibration_times) if calibration_times else 0.0
            
            # Trend analysis (simplified)
            if len(rmse_values) >= 2:
                rmse_trend = "improving" if rmse_values[-1] < rmse_values[0] else "deteriorating"
            else:
                rmse_trend = "stable"
            
            # Stability assessment
            if rmse_std < 0.01:
                stability = "high"
            elif rmse_std < 0.02:
                stability = "medium"
            else:
                stability = "low"
            
            # Parameter drift analysis (simplified)
            parameter_drift = {}
            if len(calibration_history) >= 2:
                latest_params = calibration_history[0].get('parameters', {})
                oldest_params = calibration_history[-1].get('parameters', {})
                
                for param in latest_params:
                    if param in oldest_params:
                        drift = abs(latest_params[param] - oldest_params[param])
                        parameter_drift[param] = drift
            
            # Generate recommendations
            recommendations = []
            if avg_rmse > 0.05:
                recommendations.append("High RMSE detected - consider data quality checks")
            if stability == "low":
                recommendations.append("Low stability - review calibration frequency")
            if avg_calibration_time > 30:
                recommendations.append("Long calibration times - consider method optimization")
            if len(set(methods_used)) > 1:
                recommendations.append("Multiple methods used - consider standardizing approach")
            
            if not recommendations:
                recommendations.append("Calibration performance is satisfactory")
            
            return {
                'asset_id': asset_id,
                'analysis_period': lookback_days,
                'average_rmse': avg_rmse,
                'rmse_std': rmse_std,
                'rmse_trend': rmse_trend,
                'calibration_stability': stability,
                'average_calibration_time': avg_calibration_time,
                'methods_used': list(set(methods_used)),
                'parameter_drift': parameter_drift,
                'total_calibrations': len(calibration_history),
                'recommendations': recommendations,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing calibration performance: {str(e)}")
            return {
                'asset_id': asset_id,
                'analysis_period': lookback_days,
                'error': f'Analysis failed: {str(e)}',
                'last_updated': datetime.now().isoformat()
            }