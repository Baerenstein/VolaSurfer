"""
Model Engine for VolaSurfer

Contains implementations of various volatility models including:
- Rough Bergomi (rBergomi) model
- Future models can be added here
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from scipy import special
from infrastructure.utils.logging import setup_logger


def g(x: float, a: float) -> float:
    """
    TBSS kernel applicable to the rBergomi variance process.
    """
    return x**a


def b(k: int, a: float) -> float:
    """
    Riemann sum weights for approximation of the rBergomi variance process.
    """
    return ((k**(a+1) - (k-1)**(a+1)) / (a+1))**1


def cov(a: float, n: int) -> np.ndarray:
    """
    Covariance matrix for the rBergomi model.
    """
    # For kappa = 1, the covariance structure
    cov_matrix = np.array([[1.0/n, 0], [0, 1.0/n]])
    return cov_matrix


class rBergomi:
    """
    Class for generating paths of the rBergomi model.
    
    The rough Bergomi model is a stochastic volatility model where:
    - The variance follows a fractional Brownian motion with Hurst parameter H = a + 1/2
    - The price process is correlated with the variance through parameter rho
    - The volatility-of-volatility is controlled by parameter eta (nu)
    - The initial variance level is xi
    """
    
    def __init__(self, n: int = 100, N: int = 1000, T: float = 1.00, a: float = -0.4):
        """
        Constructor for rBergomi class.
        
        Parameters:
        -----------
        n : int
            Granularity (steps per year)
        N : int  
            Number of Monte Carlo paths
        T : float
            Maturity in years
        a : float
            Alpha parameter (related to Hurst: H = a + 0.5)
        """
        # Basic assignments
        self.T = T  # Maturity
        self.n = n  # Granularity (steps per year)
        self.dt = 1.0/self.n  # Step size
        self.s = int(self.n * self.T)  # Steps
        self.t = np.linspace(0, self.T, 1 + self.s)[np.newaxis, :]  # Time grid
        self.a = a  # Alpha parameter
        self.N = N  # Number of paths

        # Construct hybrid scheme correlation structure for kappa = 1
        self.e = np.array([0, 0])
        self.c = cov(self.a, self.n)
        
        # Store parameters for external access
        self.xi = None
        self.eta = None 
        self.rho = None
        self.S0 = None

    def dW1(self) -> np.ndarray:
        """
        Produces random numbers for variance process with required
        covariance structure.
        """
        rng = np.random.multivariate_normal
        return rng(self.e, self.c, (self.N, self.s))

    def Y(self, dW: np.ndarray) -> np.ndarray:
        """
        Constructs Volterra process from appropriately
        correlated 2d Brownian increments.
        """
        Y1 = np.zeros((self.N, 1 + self.s))  # Exact integrals
        Y2 = np.zeros((self.N, 1 + self.s))  # Riemann sums

        # Construct Y1 through exact integral
        for i in np.arange(1, 1 + self.s, 1):
            Y1[:, i] = dW[:, i-1, 1]  # Assumes kappa = 1

        # Construct arrays for convolution
        G = np.zeros(1 + self.s)  # Gamma
        for k in np.arange(2, 1 + self.s, 1):
            G[k] = g(b(k, self.a)/self.n, self.a)

        X = dW[:, :, 0]  # Xi

        # Initialise convolution result, GX
        GX = np.zeros((self.N, len(X[0, :]) + len(G) - 1))

        # Compute convolution, FFT not used for small n
        for i in range(self.N):
            GX[i, :] = np.convolve(G, X[i, :])

        # Extract appropriate part of convolution
        Y2 = GX[:, :1 + self.s]

        # Finally construct and return full process
        Y = np.sqrt(2 * self.a + 1) * (Y1 + Y2)
        return Y

    def dW2(self) -> np.ndarray:
        """
        Obtain orthogonal increments.
        """
        return np.random.randn(self.N, self.s) * np.sqrt(self.dt)

    def dB(self, dW1: np.ndarray, dW2: np.ndarray, rho: float = 0.0) -> np.ndarray:
        """
        Constructs correlated price Brownian increments, dB.
        """
        self.rho = rho
        dB = rho * dW1[:, :, 0] + np.sqrt(1 - rho**2) * dW2
        return dB

    def V(self, Y: np.ndarray, xi: float = 1.0, eta: float = 1.0) -> np.ndarray:
        """
        rBergomi variance process.
        """
        self.xi = xi
        self.eta = eta
        a = self.a
        t = self.t
        V = xi * np.exp(eta * Y - 0.5 * eta**2 * t**(2 * a + 1))
        return V

    def S(self, V: np.ndarray, dB: np.ndarray, S0: float = 1) -> np.ndarray:
        """
        rBergomi price process.
        """
        self.S0 = S0
        dt = self.dt
        rho = self.rho

        # Construct non-anticipative Riemann increments
        increments = np.sqrt(V[:, :-1]) * dB - 0.5 * V[:, :-1] * dt

        # Cumsum is a little slower than Python loop.
        integral = np.cumsum(increments, axis=1)

        S = np.zeros_like(V)
        S[:, 0] = S0
        S[:, 1:] = S0 * np.exp(integral)
        return S

    def S1(self, V: np.ndarray, dW1: np.ndarray, rho: float, S0: float = 1) -> np.ndarray:
        """
        rBergomi parallel price process.
        """
        dt = self.dt

        # Construct non-anticipative Riemann increments
        increments = rho * np.sqrt(V[:, :-1]) * dW1[:, :, 0] - 0.5 * rho**2 * V[:, :-1] * dt

        # Cumsum is a little slower than Python loop.
        integral = np.cumsum(increments, axis=1)

        S = np.zeros_like(V)
        S[:, 0] = S0
        S[:, 1:] = S0 * np.exp(integral)
        return S

    def generate_paths(self, xi: float = 0.04, eta: float = 1.9, rho: float = -0.9, 
                      S0: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate complete price and variance paths for the rBergomi model.
        
        Parameters:
        -----------
        xi : float
            Initial variance level
        eta : float
            Volatility of volatility (nu parameter)
        rho : float
            Correlation between price and variance processes
        S0 : float
            Initial stock price
            
        Returns:
        --------
        S : np.ndarray
            Price paths (N x (s+1))
        V : np.ndarray  
            Variance paths (N x (s+1))
        """
        # Generate correlated Brownian motions
        dW1 = self.dW1()
        dW2 = self.dW2()
        
        # Construct Volterra process
        Y = self.Y(dW1)
        
        # Generate variance process
        V = self.V(Y, xi=xi, eta=eta)
        
        # Generate correlated price Brownian motion
        dB = self.dB(dW1, dW2, rho=rho)
        
        # Generate price process
        S = self.S(V, dB, S0=S0)
        
        return S, V

    def price_european_option(self, K: float, T: float, xi: float = 0.04, 
                            eta: float = 1.9, rho: float = -0.9, 
                            S0: float = 1.0, r: float = 0.0, 
                            option_type: str = 'call') -> float:
        """
        Price European option using Monte Carlo simulation.
        
        Parameters:
        -----------
        K : float
            Strike price
        T : float  
            Time to maturity
        xi : float
            Initial variance level
        eta : float
            Volatility of volatility
        rho : float
            Correlation parameter
        S0 : float
            Initial stock price
        r : float
            Risk-free rate
        option_type : str
            'call' or 'put'
            
        Returns:
        --------
        float
            Option price
        """
        # Temporarily update maturity if different
        original_T = self.T
        original_s = self.s
        original_t = self.t
        
        if T != self.T:
            self.T = T
            self.s = int(self.n * self.T)
            self.t = np.linspace(0, self.T, 1 + self.s)[np.newaxis, :]
        
        try:
            # Generate paths
            S, V = self.generate_paths(xi=xi, eta=eta, rho=rho, S0=S0)
            
            # Calculate payoffs
            ST = S[:, -1]  # Terminal stock prices
            
            if option_type.lower() == 'call':
                payoffs = np.maximum(ST - K, 0)
            elif option_type.lower() == 'put':
                payoffs = np.maximum(K - ST, 0)
            else:
                raise ValueError("option_type must be 'call' or 'put'")
            
            # Discount back to present value
            price = np.exp(-r * T) * np.mean(payoffs)
            
            return price
            
        finally:
            # Restore original parameters
            self.T = original_T
            self.s = original_s
            self.t = original_t

    def implied_volatility_surface(self, strikes: np.ndarray, maturities: np.ndarray,
                                 xi: float = 0.04, eta: float = 1.9, rho: float = -0.9,
                                 S0: float = 1.0, r: float = 0.0) -> np.ndarray:
        """
        Generate implied volatility surface for given strikes and maturities.
        
        Parameters:
        -----------
        strikes : np.ndarray
            Array of strike prices
        maturities : np.ndarray
            Array of maturities in years
        xi : float
            Initial variance level
        eta : float
            Volatility of volatility
        rho : float
            Correlation parameter
        S0 : float
            Initial stock price
        r : float
            Risk-free rate
            
        Returns:
        --------
        np.ndarray
            Implied volatility surface (len(maturities) x len(strikes))
        """
        from scipy.optimize import brentq
        from scipy.stats import norm
        
        def black_scholes_price(S, K, T, r, sigma, option_type='call'):
            """Black-Scholes formula for European options"""
            if T <= 0:
                if option_type == 'call':
                    return max(S - K, 0)
                else:
                    return max(K - S, 0)
            
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            
            if option_type == 'call':
                return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
            else:
                return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
        
        def implied_vol_from_price(market_price, S, K, T, r, option_type='call'):
            """Extract implied volatility from market price"""
            if market_price <= 0:
                return 0.01
                
            try:
                def objective(sigma):
                    return black_scholes_price(S, K, T, r, sigma, option_type) - market_price
                
                # Use reasonable bounds for implied volatility
                return brentq(objective, 0.001, 5.0)
            except:
                return 0.01  # Fallback
        
        iv_surface = np.zeros((len(maturities), len(strikes)))
        
        for i, T in enumerate(maturities):
            for j, K in enumerate(strikes):
                # Price option with rBergomi model
                market_price = self.price_european_option(
                    K=K, T=T, xi=xi, eta=eta, rho=rho, S0=S0, r=r, option_type='call'
                )
                
                # Extract implied volatility
                iv = implied_vol_from_price(market_price, S0, K, T, r, 'call')
                iv_surface[i, j] = iv
        
        return iv_surface


class ModelEngine:
    """
    Model Engine for managing different volatility models.
    """
    
    def __init__(self):
        self.logger = setup_logger("model_engine")
        self.models = {}
        
    def create_rbergomi_model(self, n: int = 100, N: int = 1000, T: float = 1.0, 
                            H: float = 0.1) -> rBergomi:
        """
        Create a rBergomi model instance.
        
        Parameters:
        -----------
        n : int
            Time steps per year
        N : int
            Number of Monte Carlo paths
        T : float
            Default maturity
        H : float
            Hurst parameter (H = a + 0.5, so a = H - 0.5)
            
        Returns:
        --------
        rBergomi
            Configured rBergomi model instance
        """
        a = H - 0.5  # Convert Hurst to alpha parameter
        model = rBergomi(n=n, N=N, T=T, a=a)
        
        model_key = f"rBergomi_H{H}_n{n}_N{N}"
        self.models[model_key] = model
        
        self.logger.info(f"Created rBergomi model: H={H}, n={n}, N={N}")
        return model
        
    def get_model(self, model_key: str) -> Optional[rBergomi]:
        """Get a previously created model by key."""
        return self.models.get(model_key)
        
    def list_models(self) -> Dict[str, Any]:
        """List all created models."""
        return {key: type(model).__name__ for key, model in self.models.items()}
