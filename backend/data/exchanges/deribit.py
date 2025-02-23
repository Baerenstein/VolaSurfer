from .base import ExchangeAPI
from typing import Dict
import requests
from datetime import datetime


class DeribitAPI(ExchangeAPI):
    BASE_URL = "https://www.deribit.com/api/v2"

    def __init__(self):
        self.session = requests.Session()  # Reuse connection for better performance

    def test_connection(self) -> bool:
        endpoint = f"{self.BASE_URL}/public/test"
        try:
            response = self.session.get(endpoint)
            response.raise_for_status()
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            print(f"Deribit connection error: {e}")
            return False

    def get_options(self, currency: str) -> Dict:
        endpoint = f"{self.BASE_URL}/public/get_instruments"
        params = {
            "currency": currency,
            "kind": "option",  # Only fetch options
        }
        try:
            # print(f"Making request to {endpoint} with params {params}")
            response = requests.get(endpoint, params=params)
            # print(f"Got response status code: {response.status_code}")
            response.raise_for_status()
            result = response.json()
            # print(f"Response JSON status: {result.get('success', False)}")
            options = result.get("result", [])
            print(f"{datetime.now()}: Found {len(options)} options")
            return {"options": options}
        except requests.exceptions.RequestException as e:
            print(f"Error fetching options for {currency}: {e}")
            return None

    def get_option_data(self, instrument_name: str) -> Dict:
        """Get detailed option data including mark price and implied vol"""
        endpoint = f"{self.BASE_URL}/public/ticker"
        params = {"instrument_name": instrument_name}

        try:
            # print(f"\nFetching data for {instrument_name}")
            response = requests.get(endpoint, params=params)
            # print(f"Ticker response status: {response.status_code}")
            response.raise_for_status()
            result = response.json().get("result", {})
            # print(f"Result: {result}")
            data = {
                "last_price": result.get("mark_price", 0),
                "implied_vol": result.get("mark_iv", 0),
                "delta": result.get("greeks", {}).get("delta"),
                "gamma": result.get("greeks", {}).get("gamma"),
                "vega": result.get("greeks", {}).get("vega"),
                "theta": result.get("greeks", {}).get("theta"),
            }
            # print(f"Got option data: {data}")
            return data
        except requests.exceptions.RequestException as e:
            print(f"Error fetching option data for {instrument_name}: {e}")
            return None

    def get_last_price(self, currency: str) -> float:
        endpoint = f"{self.BASE_URL}/public/ticker"
        params = {"instrument_name": f"{currency}-PERPETUAL"}
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            ticker = response.json().get("result", {})
            return ticker.get("mark_price")  # Using mark_price instead of last_price
        except requests.exceptions.RequestException as e:
            print(f"Error fetching last price for {currency}: {e}")
            return None
