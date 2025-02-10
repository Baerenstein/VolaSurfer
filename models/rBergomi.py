import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


class rBergomi(object):
    """
    Class for generating paths of the rBergomi model.

    Attributes:
        T (float): Maturity time.
        n (int): Number of steps per year.
        dt (float): Step size.
        s (int): Total number of steps.
        t (ndarray): Time grid.
        a (float): Alpha parameter.
        N (int): Number of paths.
        e (ndarray): Mean vector for multivariate normal distribution.
        c (ndarray): Covariance matrix for the process.
    """

    def __init__(self, n=100, N=1000, T=1.00, a=-0.4):
        """
        Initializes the rBergomi model with specified parameters.

        Parameters:
            n (int): Number of steps per year.
            N (int): Number of paths to generate.
            T (float): Maturity time.
            a (float): Alpha parameter.
        """
        # Basic assignments
        self.T = T  # Maturity
        self.n = n  # Granularity (steps per year)
        self.dt = 1.0 / self.n  # Step size
        self.s = int(self.n * self.T)  # Steps
        self.t = np.linspace(0, self.T, 1 + self.s)[np.newaxis, :]  # Time grid
        self.a = a  # Alpha
        self.N = N  # Paths

        # Construct hybrid scheme correlation structure for kappa = 1
        self.e = np.array([0, 0])
        self.c = cov(self.a, self.n)

    def dW1(self):
        """
        Produces random numbers for variance process with required
        covariance structure.

        Returns:
            ndarray: Random numbers with shape (N, s, 2) for the variance process.
        """
        rng = np.random.multivariate_normal
        return rng(self.e, self.c, (self.N, self.s))

    def Y(self, dW):
        """
        Constructs Volterra process from appropriately
        correlated 2D Brownian increments.

        Parameters:
            dW (ndarray): 2D Brownian increments.

        Returns:
            ndarray: The constructed Volterra process.
        """
        Y1 = np.zeros((self.N, 1 + self.s))  # Exact integrals
        Y2 = np.zeros((self.N, 1 + self.s))  # Riemann sums

        # Construct Y1 through exact integral
        for i in np.arange(1, 1 + self.s, 1):
            Y1[:, i] = dW[:, i - 1, 1]  # Assumes kappa = 1

        # Construct arrays for convolution
        G = np.zeros(1 + self.s)  # Gamma
        for k in np.arange(2, 1 + self.s, 1):
            G[k] = g(b(k, self.a) / self.n, self.a)

        X = dW[:, :, 0]  # Xi

        # Initialise convolution result, GX
        GX = np.zeros((self.N, len(X[0, :]) + len(G) - 1))

        # Compute convolution, FFT not used for small n
        # Possible to compute for all paths in C-layer?
        for i in range(self.N):
            GX[i, :] = np.convolve(G, X[i, :])

        # Extract appropriate part of convolution
        Y2 = GX[:, : 1 + self.s]

        # Finally contruct and return full process
        Y = np.sqrt(2 * self.a + 1) * (Y1 + Y2)
        return Y

    def dW2(self):
        """
        Obtain orthogonal increments.

        Returns:
            ndarray: Orthogonal increments with shape (N, s).
        """
        return np.random.randn(self.N, self.s) * np.sqrt(self.dt)

    def dB(self, dW1, dW2, rho=0.0):
        """
        Constructs correlated price Brownian increments, dB.

        Parameters:
            dW1 (ndarray): First set of Brownian increments.
            dW2 (ndarray): Second set of Brownian increments.
            rho (float): Correlation coefficient.

        Returns:
            ndarray: Correlated Brownian increments.
        """
        self.rho = rho
        dB = rho * dW1[:, :, 0] + np.sqrt(1 - rho**2) * dW2
        return dB

    def V(self, Y, xi=1.0, eta=1.0):
        """
        rBergomi variance process.

        Parameters:
            Y (ndarray): Volterra process.
            xi (float): Parameter for variance process.
            eta (float): Parameter for variance process.

        Returns:
            ndarray: Variance process values.
        """
        self.xi = xi
        self.eta = eta
        a = self.a
        t = self.t
        V = xi * np.exp(eta * Y - 0.5 * eta**2 * t ** (2 * a + 1))
        return V

    def S(self, V, dB, S0=1):
        """
        rBergomi price process.

        Parameters:
            V (ndarray): Variance process.
            dB (ndarray): Brownian increments.
            S0 (float): Initial price.

        Returns:
            ndarray: Simulated price process.
        """
        self.S0 = S0
        dt = self.dt

        # Construct non-anticipative Riemann increments
        increments = np.sqrt(V[:, :-1]) * dB - 0.5 * V[:, :-1] * dt

        # Cumsum is a little slower than Python loop.
        integral = np.cumsum(increments, axis=1)

        S = np.zeros_like(V)
        S[:, 0] = S0
        S[:, 1:] = S0 * np.exp(integral)
        return S

    def S1(self, V, dW1, rho, S0=1):
        """
        rBergomi parallel price process.

        Parameters:
            V (ndarray): Variance process.
            dW1 (ndarray): First set of Brownian increments.
            rho (float): Correlation coefficient.
            S0 (float): Initial price.

        Returns:
            ndarray: Simulated parallel price process.
        """
        dt = self.dt

        # Construct non-anticipative Riemann increments
        increments = (
            rho * np.sqrt(V[:, :-1]) * dW1[:, :, 0] - 0.5 * rho**2 * V[:, :-1] * dt
        )

        # Cumsum is a little slower than Python loop.
        integral = np.cumsum(increments, axis=1)

        S = np.zeros_like(V)
        S[:, 0] = S0
        S[:, 1:] = S0 * np.exp(integral)
        return S


def g(x, a):
    """
    TBSS kernel applicable to the rBergomi variance process.

    Parameters:
        x (float): Input value.
        a (float): Alpha parameter.

    Returns:
        float: Result of the TBSS kernel function.
    """
    return x**a


def b(k, a):
    """
    Optimal discretization of TBSS process for minimizing hybrid scheme error.

    Parameters:
        k (int): Step index.
        a (float): Alpha parameter.

    Returns:
        float: Optimal discretization value.
    """
    return ((k ** (a + 1) - (k - 1) ** (a + 1)) / (a + 1)) ** (1 / a)


def cov(a, n):
    """
    Covariance matrix for given alpha and n, assuming kappa = 1 for tractability.

    Parameters:
        a (float): Alpha parameter.
        n (int): Number of steps.

    Returns:
        ndarray: Covariance matrix.
    """
    cov = np.array([[0.0, 0.0], [0.0, 0.0]])
    cov[0, 0] = 1.0 / n
    cov[0, 1] = 1.0 / ((1.0 * a + 1) * n ** (1.0 * a + 1))
    cov[1, 1] = 1.0 / ((2.0 * a + 1) * n ** (2.0 * a + 1))
    cov[1, 0] = cov[0, 1]
    return cov


def bs(F, K, V, o="call"):
    """
    Returns the Black call price for given forward, strike and integrated variance.

    Parameters:
        F (float): Forward price.
        K (float): Strike price.
        V (float): Integrated variance.
        o (str): Option type ('call', 'put', or 'otm').

    Returns:
        float: Price of the option.
    """
    # Set appropriate weight for option token o
    w = 1
    if o == "put":
        w = -1
    elif o == "otm":
        w = 2 * (K > 1.0) - 1

    sv = np.sqrt(V)
    d1 = np.log(F / K) / sv + 0.5 * sv
    d2 = d1 - sv
    P = w * F * norm.cdf(w * d1) - w * K * norm.cdf(w * d2)
    return P


def bs_vega(S, K, T, r, sigma):
    """
    Computes the vega of a European call option.

    Parameters:
        S (float): Spot price.
        K (float): Strike price.
        T (float): Time to maturity.
        r (float): Risk-free interest rate.
        sigma (float): Volatility of the underlying asset.

    Returns:
        float: Vega of the option.
    """
    # Spot price
    # K: Strike price
    # T: time to maturity
    # r: interest rate (1=100%)
    # sigma: volatility of underlying asset

    d1 = (np.log(S / K) + (r + 0.5 * np.power(sigma, 2)) * T) / sigma * np.sqrt(T)
    vg = S * norm.pdf(d1, 0.0, 1.0) * np.sqrt(T)
    vega = np.maximum(vg, 1e-19)
    return vega


def call_price(sigma, S, K, r, T):
    """
    Computes the price of a European call option using the Black-Scholes formula.

    Parameters:
        sigma (float): Volatility of the underlying asset.
        S (float): Spot price.
        K (float): Strike price.
        r (float): Risk-free interest rate.
        T (float): Time to maturity.

    Returns:
        float: Price of the call option.
    """
    d1 = 1 / (sigma * np.sqrt(T)) * (np.log(S / K) + (r + np.power(sigma, 2) / 2) * T)
    d2 = d1 - sigma * np.sqrt(T)
    C = norm.cdf(d1) * S - norm.cdf(d2) * K * np.exp(-r * T)
    return C


def find_vol(target_value, S, K, T, r):
    """
    Finds the implied volatility for a given option price using the Newton-Raphson method.

    Parameters:
        target_value (float): Price of the option contract.
        S (float): Spot price.
        K (float): Strike price.
        T (float): Time to maturity.
        r (float): Risk-free interest rate.

    Returns:
        float: Implied volatility.
    """
    # target value: price of the option contract
    MAX_iterations = 10000
    prec = 1.0e-8
    sigma = 0.5
    for i in range(0, MAX_iterations):
        price = call_price(sigma, S, K, r, T)
        diff = target_value - price  # the root
        if abs(diff) < prec:
            return sigma
        vega = bs_vega(S, K, T, r, sigma)
        sigma = sigma + diff / vega  # f(x) / f'(x)
        if sigma > 10 or sigma < 0:
            sigma = 0.5

    return sigma


def bsinv(P, F, K, t, o="call"):
    """
    Returns implied Black volatility from given call price, forward, strike and time to maturity.

    Parameters:
        P (float): Call price.
        F (float): Forward price.
        K (float): Strike price.
        t (float): Time to maturity.
        o (str): Option type ('call', 'put', or 'otm').

    Returns:
        float: Implied volatility.
    """
    # Set appropriate weight for option token o
    w = 1
    if o == "put":
        w = -1
    elif o == "otm":
        w = 2 * (K > 1.0) - 1

    # Ensure at least instrinsic value
    P = np.maximum(P, np.maximum(w * (F - K), 0))

    def error(s):
        return bs(F, K, s**2 * t, o) - P

    s = brentq(error, 1e-19, 1e9)

    return s


def forward_price(spot, div, r, tau):
    """
    Computes the forward price given the spot price, dividends, risk-free rate, and time to expiration.

    Parameters:
        spot (float): Spot price.
        div (float): Sum of dividends to be paid until expiration.
        r (float): Risk-free rate.
        tau (float or ndarray): Time to expiration.

    Returns:
        float: Forward price.
    """

    F = spot * np.exp(r * tau) - div * np.exp(r * tau)
    return F


def rbergomi_iv(forward, strikes, texp, params):
    """
    Computes the rBergomi implied volatilities and call prices for a single expiration.

    Parameters:
        forward (float): Forward price.
        strikes (ndarray): Set of strike prices.
        texp (float): Time to expiration.
        params (tuple): rBergomi parameters (alpha, eta, rho, xi).

    Returns:
        tuple: rBivs (implied volatilities), call_prices (computed call prices), rB (rBergomi model object), FT (forward price realizations).
    """

    alpha, eta, rho, xi = params

    # to ensure computation works, we have to give greater granularity for very short durations.
    if texp < 0.01:
        steps_year = 100000
    else:
        steps_year = 366

    # Defining the Fractional Brownian Process and Resulting Price Process
    np.random.seed(4)
    rB = rBergomi(n=steps_year, N=20000, T=texp, a=alpha)
    dW1 = rB.dW1()
    dW2 = rB.dW2()
    Y = rB.Y(dW1)
    dB = rB.dB(dW1, dW2, rho=rho)
    V = rB.V(Y, xi=xi, eta=eta)
    S = rB.S(V, dB)

    # rBergomi Implied Volatilities and Call Prices
    ST = S[:, -1][:, np.newaxis]
    FT = ST * forward
    K = strikes  # np.exp(strikes) #(np.exp(k)*spot)[np.newaxis,:]
    call_payoffs = np.maximum(FT - K, 0)
    call_prices = np.mean(call_payoffs, axis=0)[:, np.newaxis]
    vec_bsinv = np.vectorize(bsinv)
    rBivs = vec_bsinv(call_prices, forward, np.transpose([K]), rB.T)
    return rBivs, call_prices, rB, FT
