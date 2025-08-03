from .base import ExchangeAPI
from typing import Dict
import requests
from datetime import datetime
from infrastructure.currency_config import get_currency_config, get_deribit_currency_code, get_perpetual_instrument


def decode_xrp_strike(strike_str: str) -> float:
    """
    Decode XRP strike price from hexadecimal-like format.
    Examples: '2d5' -> 2.5, '2d55' -> 2.55, '3d1' -> 3.1
    """
    if 'd' in strike_str:
        parts = strike_str.split('d')
        if len(parts) == 2:
            integer_part = parts[0]
            decimal_part = parts[1]
            return float(f"{integer_part}.{decimal_part}")
    # If no 'd' found, try to parse as regular float
    return float(strike_str)


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
        """
        Get options for a currency, handling different quote currencies and naming conventions
        """
        # Get the proper currency code for Deribit API
        try:
            deribit_currency = get_deribit_currency_code(currency)
        except ValueError as e:
            print(f"Currency error: {e}")
            return {"options": []}
            
        endpoint = f"{self.BASE_URL}/public/get_instruments"
        
        # Special handling for SOL and XRP - the API doesn't return their options with currency filter
        if currency.upper() in ["SOL", "XRP"]:
            try:
                print(f"Fetching options for {currency} (special handling for {currency})")
                response = requests.get(endpoint)
                response.raise_for_status()
                result = response.json()
                all_instruments = result.get("result", [])
                
                # Filter for currency-specific options
                if currency.upper() == "SOL":
                    filtered_options = [
                        inst for inst in all_instruments 
                        if inst.get("instrument_name", "").startswith("SOL_USDC-") 
                        and inst.get("kind") == "option"
                    ]
                elif currency.upper() == "XRP":
                    filtered_options = [
                        inst for inst in all_instruments 
                        if inst.get("instrument_name", "").startswith("XRP_USDC-") 
                        and inst.get("kind") == "option"
                    ]
                
                print(f"{datetime.now()}: Found {len(filtered_options)} options for {currency}")
                return {"options": filtered_options}
            except requests.exceptions.RequestException as e:
                print(f"Error fetching options for {currency}: {e}")
                return {"options": []}
        
        # Standard handling for other currencies
        params = {
            "currency": deribit_currency,
            "kind": "option",  # Only fetch options
        }
        try:
            print(f"Fetching options for {currency} (Deribit currency: {deribit_currency})")
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            result = response.json()
            options = result.get("result", [])
            print(f"{datetime.now()}: Found {len(options)} options for {currency}")
            return {"options": options}
        except requests.exceptions.RequestException as e:
            print(f"Error fetching options for {currency} ({deribit_currency}): {e}")
            return {"options": []}

    def get_option_data(self, instrument_name: str) -> Dict:
        """Get detailed option data including mark price and implied vol"""
        endpoint = f"{self.BASE_URL}/public/ticker"
        params = {"instrument_name": instrument_name}

        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            result = response.json().get("result", {})
            data = {
                "last_price": result.get("mark_price", 0),
                "implied_vol": result.get("mark_iv", 0),
                "delta": result.get("greeks", {}).get("delta"),
                "gamma": result.get("greeks", {}).get("gamma"),
                "vega": result.get("greeks", {}).get("vega"),
                "theta": result.get("greeks", {}).get("theta"),
            }
            return data
        except requests.exceptions.RequestException as e:
            print(f"Error fetching option data for {instrument_name}: {e}")
            return None

    def get_last_price(self, currency: str) -> float:
        """
        Get the last price for a currency, handling different perpetual naming conventions
        """
        try:
            perpetual_instrument = get_perpetual_instrument(currency)
        except ValueError as e:
            print(f"Currency error: {e}")
            return None
            
        endpoint = f"{self.BASE_URL}/public/ticker"
        params = {"instrument_name": perpetual_instrument}
        try:
            print(f"Fetching price for {currency} (instrument: {perpetual_instrument})")
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            ticker = response.json().get("result", {})
            price = ticker.get("mark_price")
            print(f"Price for {currency}: ${price:,.2f}")
            return price
        except requests.exceptions.RequestException as e:
            print(f"Error fetching last price for {currency} ({perpetual_instrument}): {e}")
            return None
