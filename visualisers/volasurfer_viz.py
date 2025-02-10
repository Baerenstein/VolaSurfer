import os
import time
import threading
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from polygon import RESTClient, WebSocketClient
from polygon.websocket.models import Market, WebSocketMessage
from scipy.interpolate import griddata
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx

from dataclasses import dataclass
from models.BlackScholesSolver import BlackScholesIV


@dataclass
class OptionData:
    strike: float
    maturity: datetime
    contract_type: str
    ticker: str
    bid_prices: deque
    ask_prices: deque
    timestamps: deque
    implied_vols: deque


class VolatilitySurfaceStreamer:
    def __init__(
        self,
        underlying: str,
        num_strikes: int = 2,  # strikes above and below ATM
        days_to_expiration: List[int] = [7, 14, 30, 45],  # target maturities
        max_streaming_len: int = 1000,
        risk_free_rate: float = 0.05,
        dividend_yield: float = 0.01,
    ):
        self.underlying = underlying
        self.num_strikes = num_strikes
        self.target_maturities = days_to_expiration
        self.max_streaming_len = max_streaming_len
        self.rf_rate = risk_free_rate
        self.div_yield = dividend_yield
        self.api_key = os.getenv("API_KEY")
        self.rest_client = RESTClient(self.api_key)
        self.contracts: Dict[str, OptionData] = {}
        self.bs_solver = BlackScholesIV(rate=self.rf_rate, dividend=self.div_yield)

    def fetch_last_price(self):
        """Fetch previous day's closing price"""
        try:
            prev_close_url = f"https://api.polygon.io/v2/aggs/ticker/{self.underlying}/prev?adjusted=true&apiKey={self.api_key}"

            response = requests.get(prev_close_url, timeout=10)
            response.raise_for_status()  # Raise exception for bad status codes

            data = response.json()
            if "results" not in data:
                raise Exception("No results found in stock data")

            prev_day_data = data["results"][0]  # Access first element of results array
            self.last_price = prev_day_data["c"]

            st.sidebar.metric("Ticker: ", self.underlying)
            st.sidebar.metric("Last Close", f"${self.last_price:.2f}")
            st.sidebar.write(
                f"Date: {datetime.fromtimestamp(prev_day_data['t']/1000).strftime('%Y-%m-%d')}"
            )

            return self.last_price
        except Exception as e:
            st.error(f"Error fetching last price: {str(e)}")
            return None

    def select_contracts(self):
        """Select option contracts around ATM for each target maturity"""
        try:
            selected_contracts = []
            now = datetime.now()

            for target_days in self.target_maturities:
                expiry_date = (now + timedelta(days=target_days)).strftime("%Y-%m-%d")

                # Fetch contracts for this expiry
                contracts = list(
                    self.rest_client.list_options_contracts(
                        self.underlying,
                        expiration_date_gte=now.strftime("%Y-%m-%d"),
                        expiration_date_lte=expiry_date,
                    )
                )

                if not contracts:
                    continue

                # Group by strike
                strikes = sorted(set(c.strike_price for c in contracts))
                atm_strike = min(strikes, key=lambda x: abs(x - self.last_price))
                atm_index = strikes.index(atm_strike)

                # Select strikes around ATM
                selected_indices = range(
                    max(0, atm_index - self.num_strikes),
                    min(len(strikes), atm_index + self.num_strikes + 1),
                )
                selected_strikes = [strikes[i] for i in selected_indices]

                # Add both calls and puts
                for c in contracts:
                    if c.strike_price in selected_strikes:
                        selected_contracts.append(c)

                        self.contracts[c.ticker] = OptionData(
                            strike=c.strike_price,
                            maturity=datetime.strptime(c.expiration_date, "%Y-%m-%d"),
                            contract_type=c.contract_type,
                            ticker=c.ticker,
                            bid_prices=deque(maxlen=self.max_streaming_len),
                            ask_prices=deque(maxlen=self.max_streaming_len),
                            timestamps=deque(maxlen=self.max_streaming_len),
                            implied_vols=deque(maxlen=self.max_streaming_len),
                        )

            return selected_contracts

        except Exception as e:
            st.error(f"Error selecting contracts: {str(e)}")
            return []

    def calculate_implied_vol(
        self, mid_price: float, strike: float, maturity: datetime, contract_type: str
    ) -> float:
        """Calculate implied volatility using custom Black-Scholes solver"""
        try:
            days_to_expiry = (maturity - datetime.now()).days
            T = days_to_expiry / 365.0

            if T <= 0:
                return None

            impl_vol = self.bs_solver.implied_volatility(
                price=mid_price,
                S=self.last_price,
                K=strike,
                T=T,
                option_type=contract_type,
            )

            return impl_vol

        except Exception as e:
            print(e)
            return None

    def handle_msg(self, msgs: List[WebSocketMessage]):
        """Handle incoming websocket messages"""
        for m in msgs:
            contract_symbol = str(m.symbol)
            if contract_symbol in self.contracts:
                current_time = datetime.now()
                contract = self.contracts[contract_symbol]

                # Store prices
                contract.bid_prices.append(m.bid_price)
                contract.ask_prices.append(m.ask_price)
                contract.timestamps.append(current_time)

                # Calculate mid price and IV
                mid_price = (m.bid_price + m.ask_price) / 2
                impl_vol = self.calculate_implied_vol(
                    mid_price,
                    contract.strike,
                    contract.maturity,
                    contract.contract_type,
                )
                contract.implied_vols.append(impl_vol)

    def create_vol_surface(self, interpolation_method: str) -> go.Figure:
        """Create volatility surface plot"""
        # Collect latest IVs
        data = []
        for contract in self.contracts.values():
            if contract.implied_vols and contract.implied_vols[-1] is not None:
                days_to_expiry = (contract.maturity - datetime.now()).days
                data.append(
                    {
                        "strike": contract.strike,
                        "maturity": days_to_expiry,
                        "implied_vol": contract.implied_vols[-1],
                    }
                )

        if not data:
            return None

        df = pd.DataFrame(data)

        # Create surface grid
        strikes = sorted(df["strike"].unique(), reverse=True)
        maturities = sorted(df["maturity"].unique(), reverse=True)

        # Swap K and T in meshgrid to swap axes
        T, K = np.meshgrid(maturities, strikes)
        vol_matrix = np.zeros_like(K)

        # Update matrix filling to match swapped axes
        for i, k in enumerate(strikes):
            for j, t in enumerate(maturities):
                matching = df[(df["strike"] == k) & (df["maturity"] == t)]
                if not matching.empty:
                    vol_matrix[i, j] = matching["implied_vol"].iloc[0]

        # Interpolate to create a smoother surface using the selected method
        grid_x, grid_y = np.meshgrid(maturities, strikes)
        vol_matrix = griddata((df["maturity"], df["strike"]), df["implied_vol"], (grid_x, grid_y), method=interpolation_method)

        # Create surface plot with swapped axes and custom colorscale
        fig = go.Figure(data=[go.Surface(x=T, y=K, z=vol_matrix, showscale=False, colorscale="sunset")])

        # Update layout with bird's eye view and swapped axis labels
        fig.update_layout(
            title="Real-time Implied Volatility Surface",
            scene=dict(
                xaxis_title="Time to Maturity (Days)",
                yaxis_title="Strike Price",
                zaxis_title="Implied Volatility",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=0.8),
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=0, z=1),
                ),
                xaxis=dict(autorange="reversed"),
                yaxis=dict(autorange="reversed"),
            ),
            width=800,  # Adjust width
            height=500,  # Adjust height
        )

        return fig


def main():
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select a page:", ["Volatility Surface", "Bid Ask Prices", "Volatility Series"])

    if page == "Volatility Surface":
        show_volatility_surface()
    elif page == "Bid Ask Prices":
        show_bid_ask_prices()
    elif page == "Volatility Series":
        show_volatility_series()

def show_volatility_surface():
    st.title("Volatility Surface")
    # Your existing volatility surface code here
    if "vss" not in st.session_state:
        st.session_state.vss = VolatilitySurfaceStreamer(
            underlying="AAPL",  # You can change this to a dynamic input if needed
            num_strikes=6,
            days_to_expiration=[7, 14, 30, 60],
        )

        # Fetch initial data
        if st.session_state.vss.fetch_last_price():
            selected_contracts = st.session_state.vss.select_contracts()
            st.sidebar.write(f"Selected {len(selected_contracts)} contracts")

            # Display unique strikes and maturities
            unique_strikes = len(set(c.strike_price for c in selected_contracts))
            unique_maturities = len(set((datetime.strptime(c.expiration_date, "%Y-%m-%d") - datetime.now()).days for c in selected_contracts))
            st.sidebar.write(f"Unique Strikes: {unique_strikes}")
            st.sidebar.write(f"Unique Maturities: {unique_maturities}")

    # Add an option to choose interpolation style
    interpolation_method = st.sidebar.selectbox(
        "Select Interpolation Method:",
        options=["linear", "cubic", "nearest"],
        index=0
    )

    # Create plot placeholders
    surface_plot = st.empty()

    def update_display():
        # Update surface plot
        fig = st.session_state.vss.create_vol_surface(interpolation_method)
        if fig:
            surface_plot.plotly_chart(fig)

    if "websocket_running" not in st.session_state:
        st.session_state.websocket_running = False

    if st.sidebar.button("Start Streaming") and not st.session_state.websocket_running:
        st.session_state.websocket_running = True

        # Initialize WebSocket
        client = WebSocketClient(
            api_key=st.session_state.vss.api_key, market=Market.Options
        )

        # Subscribe to contracts
        subscription_string = ",".join(
            f"Q.{ticker}" for ticker in st.session_state.vss.contracts.keys()
        )
        client.subscribe(subscription_string)

        # Start WebSocket in a thread
        def run_websocket():
            try:
                client.run(st.session_state.vss.handle_msg)
            except Exception as e:
                st.error(f"WebSocket error: {str(e)}")
                st.session_state.websocket_running = False

        websocket_thread = threading.Thread(target=run_websocket)
        websocket_thread.daemon = True
        add_script_run_ctx(websocket_thread)
        websocket_thread.start()

    # Add a delay between updates
    if st.session_state.websocket_running:
        update_display()
        time.sleep(5)  # Add a x-second delay between updates
        st.rerun()

def show_bid_ask_prices():
    st.title("Bid Ask Prices")

    # Move the contract selection to the sidebar
    if "vss" in st.session_state:
        contracts = list(st.session_state.vss.contracts.keys())
        selected_contracts = st.sidebar.multiselect("Select contracts:", contracts)

        # Initialize contract data if not already done
        if selected_contracts:
            if "contract_data" not in st.session_state:
                st.session_state.contract_data = {}

            # Initialize data for each selected contract
            for contract in selected_contracts:
                if contract not in st.session_state.contract_data:
                    st.session_state.contract_data[contract] = {
                        "bid_prices": deque(maxlen=100000),
                        "ask_prices": deque(maxlen=100000),
                        "timestamps": deque(maxlen=100000),
                    }

            chart_placeholder = st.empty()

            # Define a color map for contracts
            colors = {contract: f'rgba({np.random.randint(0, 255)}, {np.random.randint(0, 255)}, {np.random.randint(0, 255)}, 1)' for contract in selected_contracts}

            def update_chart():
                fig = go.Figure()

                for contract in selected_contracts:
                    data = st.session_state.contract_data[contract]
                    if len(data["bid_prices"]) > 0:
                        # Use the same color for bid and ask prices
                        color = colors[contract]
                        fig.add_trace(
                            go.Scatter(
                                x=list(data["timestamps"]),
                                y=list(data["bid_prices"]),
                                name=f"{contract} Bid",
                                line=dict(color=color),
                            )
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=list(data["timestamps"]),
                                y=list(data["ask_prices"]),
                                name=f"{contract} Ask",
                                line=dict(color=color, dash='dash'),  # Optional: make ask line dashed
                            )
                        )

                fig.update_layout(
                    title="Live Bid and Ask Prices", 
                    xaxis_title="Time", 
                    yaxis_title="Price", 
                    width=1500,  # Adjusted width to fit the screen
                    height=600
                )

                chart_placeholder.plotly_chart(fig, use_container_width=True)

            def handle_msg(msgs):
                for m in msgs:
                    contract_symbol = str(m.symbol)
                    if contract_symbol in selected_contracts:
                        current_time = datetime.now()

                        data = st.session_state.contract_data[contract_symbol]
                        data["bid_prices"].append(m.bid_price)
                        data["ask_prices"].append(m.ask_price)
                        data["timestamps"].append(current_time)

                        update_chart()

            def start_websocket():
                api_key = os.getenv("API_KEY")
                client = WebSocketClient(api_key=api_key, market=Market.Options)

                subscription_string = ",".join(f"Q.{contract}" for contract in selected_contracts)
                client.subscribe(subscription_string)

                try:
                    client.run(handle_msg)
                except Exception as e:
                    st.error(f"WebSocket error: {str(e)}")

            if st.sidebar.button("Start Streaming"):
                websocket_thread = threading.Thread(target=start_websocket)
                websocket_thread.daemon = True
                add_script_run_ctx(websocket_thread)
                websocket_thread.start()

            update_chart()
        else:
            st.warning("No contracts available to display.")
    else:
        st.warning("Volatility Surface data not loaded yet.")

def show_volatility_series():
    st.title("Volatility Series")
    st.write("This page will provide information on the volatility series for the selected options.")

if __name__ == "__main__":
    main()
