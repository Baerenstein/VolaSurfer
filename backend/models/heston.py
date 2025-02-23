from dataclasses import dataclass
import numpy as np
from scipy.integrate import quad
from typing import Optional, Tuple, Dict
import warnings


@dataclass
class HestonParameters:
    """Parameters for the Heston model"""
    kappa: float  # Mean reversion speed of variance
    theta: float  # Long-term variance
    sigma: float  # Volatility of variance
    rho: float   # Correlation between spot and variance
    v0: float    # Initial variance
    
    def validate(self):
        """Validate Feller condition and other parameter constraints"""
        if 2 * self.kappa * self.theta <= self.sigma ** 2:
            warnings.warn("Feller condition not satisfied: 2κθ <= σ²")
        if not (0 <= self.rho <= 1):
            raise ValueError("Correlation ρ must be between 0 and 1")
        if self.v0 < 0:
            raise ValueError("Initial variance v0 must be non-negative")
        return True


class HestonModel:
    def __init__(self, params: HestonParameters):
        """
        Initialize Heston model with parameters.
        
        Args:
            params: HestonParameters object containing model parameters
        """
        self.params = params
        self.params.validate()
        
    def characteristic_function(self, u: float, t: float, S: float, r: float) -> complex:
        """
        Compute the characteristic function for the Heston model.
        
        Args:
            u: Integration variable
            t: Time to maturity
            S: Current spot price
            r: Risk-free rate
            
        Returns:
            Complex value of the characteristic function
        """
        kappa = self.params.kappa
        theta = self.params.theta
        sigma = self.params.sigma
        rho = self.params.rho
        v0 = self.params.v0
        
        # Complex intermediate terms
        alpha = -0.5 * (u ** 2 + u * 1j)
        beta = kappa - rho * sigma * u * 1j
        gamma = 0.5 * sigma ** 2
        
        # Compute d
        d = np.sqrt(beta ** 2 - 4 * alpha * gamma)
        
        # Compute g
        g = (beta - d) / (beta + d)
        
        # Compute first exponential term
        exp1 = np.exp(1j * u * (np.log(S) + r * t))
        
        # Compute A term
        A = 1j * u * r * t
        
        # Compute B term
        B = theta * kappa / (sigma ** 2) * (
            (beta - d) * t - 2 * np.log((1 - g * np.exp(-d * t)) / (1 - g))
        )
        
        # Compute C term
        C = v0 / (sigma ** 2) * (beta - d) * (1 - np.exp(-d * t)) / (1 - g * np.exp(-d * t))
        
        return exp1 * np.exp(A + B + C)
    
    def price_option(
        self, 
        S: float, 
        K: float, 
        T: float, 
        r: float, 
        option_type: str = 'call',
        integration_params: Optional[Dict] = None
    ) -> float:
        """
        Price a European option using Heston model.
        
        Args:
            S: Current spot price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            option_type: 'call' or 'put'
            integration_params: Optional parameters for numerical integration
            
        Returns:
            Option price
        """
        if integration_params is None:
            integration_params = {
                'limit': 100,
                'points': 1000
            }
            
        def integrand(u: float, j: int) -> float:
            """Integration function for option pricing"""
            denominator = 1j * u
            if j == 2:
                denominator = denominator - 1
                
            cf = self.characteristic_function(u - (1j * (j-1)), T, S, r)
            return (np.exp(-1j * u * np.log(K)) * cf / denominator).real
            
        # Compute probabilities P1 and P2
        P = [0, 0]
        for j in range(1, 3):
            integral, _ = quad(
                integrand,
                0,
                integration_params['limit'],
                args=(j,),
                limit=integration_params['points']
            )
            P[j-1] = 0.5 + (1/np.pi) * integral
            
        # Calculate option price
        if option_type.lower() == 'call':
            price = S * P[0] - K * np.exp(-r * T) * P[1]
        else:  # put
            price = K * np.exp(-r * T) * (1 - P[1]) - S * (1 - P[0])
            
        return max(0, price)
    
    def implied_volatility_surface(
        self,
        spot: float,
        strikes: np.ndarray,
        maturities: np.ndarray,
        r: float,
        option_type: str = 'call'
    ) -> np.ndarray:
        """
        Generate implied volatility surface using the Heston model.
        
        Args:
            spot: Current spot price
            strikes: Array of strike prices
            maturities: Array of maturities
            r: Risk-free rate
            option_type: 'call' or 'put'
            
        Returns:
            2D array of implied volatilities
        """
        vols = np.zeros((len(maturities), len(strikes)))
        
        for i, T in enumerate(maturities):
            for j, K in enumerate(strikes):
                price = self.price_option(spot, K, T, r, option_type)
                vols[i,j] = self._newton_implied_vol(price, spot, K, T, r, option_type)
                
        return vols
    
    def _newton_implied_vol(
        self,
        price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: str,
        tol: float = 1e-5,
        max_iter: int = 100
    ) -> float:
        """
        Calculate implied volatility using Newton-Raphson method.
        
        Args:
            price: Option price
            S: Spot price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            option_type: 'call' or 'put'
            tol: Tolerance for convergence
            max_iter: Maximum iterations
            
        Returns:
            Implied volatility
        """
        def bs_price(sigma):
            """Black-Scholes price for Newton iteration"""
            d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            if option_type.lower() == 'call':
                return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            
        def bs_vega(sigma):
            """Black-Scholes vega for Newton iteration"""
            d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            return S * np.sqrt(T) * norm.pdf(d1)
        
        # Initial guess using simplified Black-Scholes approximation
        sigma = np.sqrt(2 * abs(np.log(S/K) + r * T) / T)
        
        for _ in range(max_iter):
            price_diff = bs_price(sigma) - price
            if abs(price_diff) < tol:
                return sigma
            sigma = sigma - price_diff / bs_vega(sigma)
            
        warnings.warn("Implied volatility calculation did not converge")
        return sigma