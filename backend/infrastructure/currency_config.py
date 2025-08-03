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
    instrument_format: str  # "UPPERCASE-DASH" or "lowercase_underscore"
    perpetual_suffix: str  # "-PERPETUAL" or "_perpetual"
    enabled: bool = True

# Currency configurations
CURRENCY_CONFIGS = {
    "BTC": CurrencyConfig("BTC", "Bitcoin", "USDT", "UPPERCASE-DASH", "-PERPETUAL"),
    "ETH": CurrencyConfig("ETH", "Ethereum", "USDT", "UPPERCASE-DASH", "-PERPETUAL"),
    "SOL": CurrencyConfig("SOL", "Solana", "USDC", "UPPERCASE-DASH", "-PERPETUAL"),
    "XRP": CurrencyConfig("XRP", "Ripple", "USDC", "UPPERCASE-DASH", "-PERPETUAL"),
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
    
    # For SOL, the options use "sol_usdc" as the price_index, not "SOL"
    if symbol.upper() == "SOL":
        return "sol_usdc"
    
    # For XRP, the options use "xrp_usdc" as the price_index, not "XRP"
    if symbol.upper() == "XRP":
        return "xrp_usdc"
    
    # For BTC and ETH, the options use just the symbol as the currency
    if symbol.upper() in ["BTC", "ETH"]:
        return symbol.upper()
    
    if config.instrument_format == "lowercase_underscore":
        return f"{symbol.lower()}_{config.quote_currency.lower()}"
    else:
        return f"{symbol}-{config.quote_currency}"

def get_perpetual_instrument(symbol: str) -> str:
    """Get the perpetual instrument name for a currency"""
    config = get_currency_config(symbol)
    if not config:
        raise ValueError(f"Currency '{symbol}' not supported")
    
    # For SOL, the perpetual is SOL_USDC (not SOL-PERPETUAL)
    if symbol.upper() == "SOL":
        return "SOL_USDC"
    
    # For XRP, the perpetual is XRP_USDC (not XRP-PERPETUAL)
    if symbol.upper() == "XRP":
        return "XRP_USDC"
    
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