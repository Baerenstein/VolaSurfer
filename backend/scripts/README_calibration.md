# Surface Calibration for VolaSurfer

This module implements surface calibration using the rough Bergomi (rBergomi) volatility model with a CNN-based approach.

## Overview

The surface calibration system provides two main approaches:

1. **Market Data Approach** (Recommended): Uses your existing volatility surfaces from the database
2. **Synthetic Data Approach**: Generates synthetic rBergomi surfaces for training

## Installation

Install the required ML dependencies:

```bash
pip install tensorflow scikit-learn
```

## Quick Start

### 1. Train using Market Data (Recommended)

```bash
cd backend/scripts
python calibrate_surface.py --mode train --use-market-data --instrument BTC --n-synthetic 500 --epochs 50
```

This approach:
- Uses your database volatility surfaces as training data
- Runs optimization to get "ground truth" parameters for each surface
- Augments with 500 synthetic surfaces for better generalization
- Trains faster with fewer epochs since real data is more informative

### 2. Calibrate Recent Surfaces

```bash
python calibrate_surface.py --mode calibrate --instrument BTC --surface-limit 20 --output-file btc_calibration.json
```

### 3. Evaluate Model Performance

```bash
python calibrate_surface.py --mode evaluate --model-path ./models/rbergomi_cnn
```

## Why Market Data + Synthetic Approach Works Better

You asked an excellent question about why we need synthetic data when we have market data. Here's the hybrid approach I implemented:

### The Problem with Pure Market Data
- **No Ground Truth**: We don't know the "true" rBergomi parameters that generated each market surface
- **Limited Coverage**: Market only explores certain parameter combinations
- **Noise**: Market data has noise, bid-ask spreads, liquidity issues

### The Problem with Pure Synthetic Data  
- **Model Risk**: Synthetic data may not match real market behavior
- **Parameter Assumptions**: We're assuming rBergomi is the right model

### The Hybrid Solution
1. **Use market surfaces as training targets** (your real data)
2. **Use optimization to estimate parameters** for each market surface (creates "ground truth")
3. **Augment with some synthetic data** to explore parameter space more fully
4. **Train CNN on combined dataset**

This gives you:
- Real market behavior from your database
- Robust parameter estimation
- Fast calibration once trained

## Usage Examples

### Instrument-Specific Calibration

```bash
# Train model for Bitcoin
python calibrate_surface.py --mode train --use-market-data --instrument BTC

# Train model for Ethereum  
python calibrate_surface.py --mode train --use-market-data --instrument ETH

# Calibrate recent BTC surfaces
python calibrate_surface.py --mode calibrate --instrument BTC --surface-limit 50

# Train for all instruments (if no --instrument specified)
python calibrate_surface.py --mode train --use-market-data
```

### Custom Grid Parameters

```bash
python calibrate_surface.py --mode train \
  --use-market-data \
  --instrument BTC \
  --maturities 0.08 0.25 0.5 1.0 2.0 \
  --moneyness-min 0.7 \
  --moneyness-max 1.3 \
  --moneyness-points 13
```

### Optimization-Only Calibration (No ML)

```bash
python calibrate_surface.py --mode calibrate \
  --instrument BTC \
  --calibration-method optimization \
  --surface-limit 5
```

### Large Scale Training

```bash
python calibrate_surface.py --mode train \
  --use-market-data \
  --instrument BTC \
  --n-synthetic 2000 \
  --epochs 100 \
  --batch-size 64 \
  --model-path ./models/btc_rbergomi_production
```

## Model Parameters

The rBergomi model has three key parameters:

- **H** (Hurst exponent): Controls volatility clustering [0.01, 0.5]
- **nu** (Vol-of-vol): Controls volatility magnitude [0.1, 3.0] 
- **rho** (Correlation): Price-volatility correlation [-0.99, 0.99]

## Database Integration

Calibrated parameters are automatically stored in your existing database schema:

- `models` table: Model definitions and schemas
- `model_parameters` table: Calibrated parameter sets
- Links to `surfaces` table for traceability

## Output Format

Calibration results include:

```json
{
  "H": 0.123,
  "nu": 1.456,
  "rho": -0.789,
  "timestamp": "2025-01-XX...",
  "calibration_method": "cnn",
  "parameter_id": 42,
  "surface_id": 123,
  "asset_id": 1
}
```

## Performance Tips

1. **Start with market data**: `--use-market-data` for most realistic training
2. **Use fewer synthetic samples**: 500-1000 is often enough for augmentation
3. **Early stopping**: Model will stop training when validation loss plateaus
4. **Batch calibration**: Process multiple surfaces at once for efficiency

## Instrument Support

The calibration system supports per-instrument models:

- **Available instruments**: Check with `SELECT ticker FROM assets;` in your database
- **Common tickers**: BTC, ETH, SPX, etc. (depends on your data)
- **Model isolation**: Each instrument gets its own parameter distributions
- **Mixed training**: Omit `--instrument` to train on all instruments combined

## Troubleshooting

- **TensorFlow not found**: Install with `pip install tensorflow`
- **No market surfaces**: Check your database has surfaces with `option_type='c'`
- **Instrument not found**: Check `SELECT ticker FROM assets;` for available instruments
- **Model training slow**: Reduce `--n-synthetic` or use fewer `--epochs`
- **Calibration fails**: Try `--calibration-method optimization` as fallback 