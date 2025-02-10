import streamlit as st
import numpy as np
import pandas as pd
from models.rBergomi import rBergomi, bsinv

# Function to compute implied volatilities
def compute_implied_vols(n=100, N=1000, T=1.0, a=-0.43, xi=0.235**2, eta=1.9, rho=-0.9):
    rB = rBergomi(n=n, N=N, T=T, a=a)
    np.random.seed(0)

    dW1 = rB.dW1()
    dW2 = rB.dW2()
    Y = rB.Y(dW1)
    dB = rB.dB(dW1, dW2, rho=rho)
    V = rB.V(Y, xi=xi, eta=eta)
    S = rB.S(V, dB)

    k = np.arange(-0.5, 0.51, 0.01)
    ST = S[:, -1][:, np.newaxis]
    K = np.exp(k)[np.newaxis, :]
    call_payoffs = np.maximum(ST - K, 0)
    call_prices = np.mean(call_payoffs, axis=0)[:, np.newaxis]
    implied_vols = np.vectorize(bsinv)(call_prices, 1., np.transpose(K), rB.T)

    return k, implied_vols

# Streamlit app
st.title("rBergomi Implied Volatility Visualization")

# Sidebar for parameters
st.sidebar.header("Parameters")
n = st.sidebar.slider("Number of Steps per Year (n)", 10, 500, 100)
N = st.sidebar.slider("Number of Paths (N)", 1000, 50000, 10000)
T = st.sidebar.slider("Maturity Time (T)", 0.1, 2.0, 1.0)
a = st.sidebar.slider("Alpha Parameter (a)", -1.0, 0.0, -0.43)
xi = st.sidebar.number_input("Variance Parameter (xi)", value=0.235**2)
eta = st.sidebar.number_input("Eta Parameter (eta)", value=1.9)
rho = st.sidebar.slider("Correlation Coefficient (rho)", -1.0, 1.0, -0.9)

# Compute implied volatilities
k, implied_vols = compute_implied_vols(n, N, T, a, xi, eta, rho)

# Plotting
st.subheader("Implied Volatilities")
st.line_chart(pd.DataFrame(implied_vols.flatten(), index=k.flatten(), columns=["Implied Volatility"]))