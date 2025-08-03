"""
Currency configuration for VolaSurfer
Handles different quote currencies and naming conventions
"""
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class CurrencyConfig:
    """Configuration for a supported currency"""
    symbol: str
    name: str
    quote_currency: str  # USDT or USDC
    instrument_format: str  # "UPPERCASE-DASH" or "UPPERCASE_UNDERSCORE" or "lowercase_underscore"
    perpetual_suffix: str  # "-PERPETUAL" or "_perpetual"
    enabled: bool = True

# Currency configurations
CURRENCY_CONFIGS = {
    "BTC": CurrencyConfig("BTC", "Bitcoin", "USDT", "UPPERCASE-DASH", "-PERPETUAL"),
    "ETH": CurrencyConfig("ETH", "Ethereum", "USDT", "UPPERCASE-DASH", "-PERPETUAL"),
    "SOL": CurrencyConfig("SOL", "Solana", "USDC", "UPPERCASE_UNDERSCORE", "-PERPETUAL"),
    "XRP": CurrencyConfig("XRP", "Ripple", "USDC", "UPPERCASE_UNDERSCORE", "-PERPETUAL"),
    "ADA": CurrencyConfig("ADA", "Cardano", "USDC", "lowercase_underscore", "_perpetual"),
}

def get_currency_config(symbol: str) -> Optional[CurrencyConfig]:
    """Get configuration for a specific currency"""
    return CURRENCY_CONFIGS.get(symbol.upper())

def is_currency_supported(symbol: str) -> bool:
    """Check if a currency is supported"""
    config = get_currency_config(symbol)
    return config is not None and config.enabled

def get_deribit_currency_code(symbol: str) -> str:
    """Get the currency code as expected by Deribit API"""
    config = get_currency_config(symbol)
    if not config:
        raise ValueError(f"Currency '{symbol}' not supported")
    
    # Special handling for currencies that use different API formats
    api_format_overrides = {
        "SOL": "sol_usdc",
        "XRP": "xrp_usdc",
        "BTC": "BTC",
        "ETH": "ETH"
    }
    
    if symbol.upper() in api_format_overrides:
        return api_format_overrides[symbol.upper()]
    
    # Use configured format for other currencies
    if config.instrument_format == "lowercase_underscore":
        return f"{symbol.lower()}_{config.quote_currency.lower()}"
    else:
        return f"{symbol}-{config.quote_currency}"

def get_perpetual_instrument(symbol: str) -> str:
    """Get the perpetual instrument name for a currency"""
    config = get_currency_config(symbol)
    if not config:
        raise ValueError(f"Currency '{symbol}' not supported")
    
    # Special handling for currencies that use different perpetual formats
    perpetual_format_overrides = {
        "SOL": "SOL_USDC",
        "XRP": "XRP_USDC"
    }
    
    if symbol.upper() in perpetual_format_overrides:
        return perpetual_format_overrides[symbol.upper()]
    
    # Use configured format for other currencies
    if config.instrument_format == "lowercase_underscore":
        return f"{symbol.lower()}{config.perpetual_suffix}"
    else:
        return f"{symbol}{config.perpetual_suffix}"

def get_supported_currencies() -> list:
    """Get list of supported currency symbols"""
    return [symbol for symbol, config in CURRENCY_CONFIGS.items() if config.enabled]

def validate_currency(symbol: str) -> str:
    """Validate and normalize currency symbol"""
    normalized = symbol.upper()
    if not is_currency_supported(normalized):
        supported = get_supported_currencies()
        raise ValueError(f"Currency '{symbol}' not supported. Supported currencies: {', '.join(supported)}")
    return normalized 