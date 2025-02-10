import numpy as np
from scipy.stats import norm


class BlackScholesIV:
    def __init__(self, rate=0.05, dividend=0.0, max_iterations=100, precision=1e-5):
        self.rate = rate
        self.dividend = dividend
        self.max_iterations = max_iterations
        self.precision = precision

    def _black_scholes_price(self, S, K, T, r, q, sigma, option_type="call"):
        """Calculate Black-Scholes option price"""
        if T <= 0:
            # Handle expired options
            if option_type == "call":
                return max(0, S - K)
            else:
                return max(0, K - S)

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == "call":
            price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(
                d2
            )
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(
                -d1
            )

        return price

    def _vega(self, S, K, T, r, q, sigma):
        """Calculate option vega for Newton-Raphson iteration"""
        if T <= 0:
            return 0

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return S * np.exp(-q * T) * np.sqrt(T) * norm.pdf(d1)

    def implied_volatility(self, price, S, K, T, option_type="call"):
        """
        Calculate implied volatility using Newton-Raphson method

        Parameters:
        price: Market price of option
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        option_type: 'call' or 'put'

        Returns:
        float: Implied volatility
        """
        if T <= 0:
            return None

        # Initial guess (use simple rule of thumb)
        sigma = np.sqrt(2 * np.pi / T) * price / S

        # Ensure reasonable starting point
        sigma = max(0.01, min(sigma, 5.0))

        for i in range(self.max_iterations):
            # Calculate price and vega at current sigma
            price_calc = self._black_scholes_price(
                S, K, T, self.rate, self.dividend, sigma, option_type
            )
            vega = self._vega(S, K, T, self.rate, self.dividend, sigma)

            # Handle zero vega case
            if abs(vega) < 1e-10:
                return None

            # Newton-Raphson update
            diff = price - price_calc
            if abs(diff) < self.precision:
                return sigma

            sigma = sigma + diff / vega

            # Add bounds to prevent unrealistic values
            if sigma < 0.001 or sigma > 5:
                return None

        # If we didn't converge, return None
        return None


# Example usage:
# bs = BlackScholesIV(rate=0.05, dividend=0.0)
# iv = bs.implied_volatility(price=2.5, S=100, K=100, T=0.5, option_type='call')
