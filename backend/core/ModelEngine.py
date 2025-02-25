from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import numpy as np
from scipy.optimize import minimize
from datetime import datetime

from data.storage.base_store import BaseStore
from data.storage.storage_factory import StorageFactory
from infrastructure.settings import Settings
from data.utils.data_schemas import OptionContract, VolSurface

@dataclass
class HestonParameters:
    """Heston model parameters"""
    kappa: float = 2.0       # Mean reversion speed
    theta: float = 0.04      # Long-run variance
    sigma: float = 0.3       # Volatility of variance
    rho: float = -0.7        # Correlation
    v0: float = 0.04        # Initial variance
    
    def to_array(self) -> np.ndarray:
        return np.array([self.kappa, self.theta, self.sigma, self.rho, self.v0])
    
    @classmethod
    def from_array(cls, params: np.ndarray) -> 'HestonParameters':
        return cls(
            kappa=params[0],
            theta=params[1],
            sigma=params[2],
            rho=params[3],
            v0=params[4]
        )

class HestonEngine:
    """Main engine for Heston model operations"""
    
    def __init__(self, store: BaseStore, params: Optional[HestonParameters] = None):
        self.store = store
        self.params = params or HestonParameters()
        self.calibrated = False
        
    def calibrate(self, 
                 symbol: str,
                 asset_type: str,
                 min_expiry: int = 7,
                 max_expiry: int = 90) -> Tuple[bool, Dict]:
        """
        Calibrate model parameters to market data
        
        Args:
            symbol: Asset symbol
            asset_type: Type of asset (e.g., 'crypto', 'equity')
            min_expiry: Minimum days to expiry to consider
            max_expiry: Maximum days to expiry to consider
            
        Returns:
            Tuple[bool, Dict]: Success flag and calibration results
        """
        # Get market data
        options_chain = self.store.get_options_chain(symbol)
        if not options_chain:
            return False, {"error": "No options data available"}
            
        # Filter relevant options
        filtered_options = self._filter_options(options_chain, min_expiry, max_expiry)
        if not filtered_options:
            return False, {"error": "No valid options after filtering"}
            
        # Prepare market data for calibration
        market_data = self._prepare_calibration_data(filtered_options)
        
        # Define optimization bounds
        bounds = [
            (0.1, 10.0),    # kappa
            (0.001, 0.5),   # theta
            (0.05, 0.8),    # sigma
            (-0.95, 0.95),  # rho
            (0.001, 0.5)    # v0
        ]
        
        # Run calibration
        try:
            result = minimize(
                self._objective_function,
                self.params.to_array(),
                args=(market_data,),
                bounds=bounds,
                method='L-BFGS-B'
            )
            
            if result.success:
                self.params = HestonParameters.from_array(result.x)
                self.calibrated = True
                return True, {
                    "parameters": self.params.__dict__,
                    "optimization_result": {
                        "success": result.success,
                        "message": result.message,
                        "iterations": result.nit,
                        "objective_value": result.fun
                    }
                }
            else:
                return False, {"error": f"Calibration failed: {result.message}"}
                
        except Exception as e:
            return False, {"error": f"Calibration error: {str(e)}"}
            
    def _filter_options(self, 
                       options: List[OptionContract], 
                       min_expiry: int,
                       max_expiry: int) -> List[OptionContract]:
        """Filter options based on criteria"""
        return [
            opt for opt in options
            if min_expiry <= opt.days_to_expiry <= max_expiry
            and opt.last_price > 0
            and opt.implied_vol > 0
        ]
        
    def _prepare_calibration_data(self, 
                                options: List[OptionContract]) -> Dict:
        """Prepare market data for calibration"""
        return {
            "strikes": np.array([opt.strike for opt in options]),
            "expiries": np.array([opt.days_to_expiry/365 for opt in options]),
            "prices": np.array([opt.last_price for opt in options]),
            "types": np.array([opt.option_type for opt in options]),
            "spot": self.store.get_underlying_data(options[0].symbol)['price'].iloc[-1]
        }
        
    def _objective_function(self, 
                          params: np.ndarray, 
                          market_data: Dict) -> float:
        """
        Objective function for calibration
        
        Computes sum of squared errors between model and market prices
        """
        self.params = HestonParameters.from_array(params)
        
        total_error = 0
        for i in range(len(market_data['strikes'])):
            model_price = self._price_option(
                S=market_data['spot'],
                K=market_data['strikes'][i],
                T=market_data['expiries'][i],
                option_type=market_data['types'][i]
            )
            market_price = market_data['prices'][i]
            total_error += (model_price - market_price) ** 2
            
        return total_error
        
    def _price_option(self, 
                     S: float,
                     K: float, 
                     T: float,
                     option_type: str = 'c') -> float:
        """
        Price an option using current model parameters
        
        TODO: Implement actual Heston pricing logic
        Currently returns Black-Scholes as placeholder
        """
        # Placeholder for actual Heston implementation
        r = 0.0  # Risk-free rate
        vol = np.sqrt(self.params.v0)
        
        d1 = (np.log(S/K) + (r + vol**2/2)*T) / (vol*np.sqrt(T))
        d2 = d1 - vol*np.sqrt(T)
        
        if option_type.lower() == 'c':
            price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        else:
            price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
            
        return price
        
    def generate_surface(self, 
                        spot: float,
                        strikes: np.ndarray,
                        expiries: np.ndarray) -> VolSurface:
        """
        Generate volatility surface using calibrated parameters
        
        Args:
            spot: Current spot price
            strikes: Array of strike prices
            expiries: Array of expiries in years
            
        Returns:
            VolSurface object containing the generated surface
        """
        if not self.calibrated:
            raise ValueError("Model must be calibrated before generating surface")
            
        # Create mesh grid of strikes and expiries
        strike_grid, expiry_grid = np.meshgrid(strikes, expiries)
        
        # Calculate implied vols for each point
        implied_vols = np.zeros_like(strike_grid)
        for i in range(len(expiries)):
            for j in range(len(strikes)):
                implied_vols[i,j] = self._implied_vol(
                    spot, strikes[j], expiries[i]
                )
                
        # Create VolSurface object
        return VolSurface(
            timestamp=datetime.now(),
            method="heston",
            strikes=strikes.tolist(),
            days_to_expiry=(expiries * 365).tolist(),
            implied_vols=implied_vols.flatten().tolist(),
            snapshot_id=datetime.now().isoformat()
        )
        
    def _implied_vol(self, S: float, K: float, T: float) -> float:
        """
        Calculate implied volatility for given parameters
        
        TODO: Implement proper Heston implied vol calculation
        Currently returns simple approximation
        """
        return np.sqrt(self.params.theta + 
                      (self.params.v0 - self.params.theta)*
                      np.exp(-self.params.kappa*T))

def create_heston_engine(settings: Settings) -> HestonEngine:
    """Factory function to create HestonEngine instance"""
    store = StorageFactory.create_storage(settings)
    return HestonEngine(store)