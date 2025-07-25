#!/usr/bin/env python3
"""
Surface Calibration Script for VolaSurfer

This script provides a command-line interface for training and running
surface calibration using the rBergomi model with CNN.

Usage:
    python calibrate_surface.py --mode train --use-market-data --n-synthetic 1000
    python calibrate_surface.py --mode calibrate --surface-limit 10
    python calibrate_surface.py --mode evaluate --model-path ./models/rbergomi_cnn
"""

import argparse
import sys
import os
import json
from datetime import datetime
from pathlib import Path

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.SurfaceCalibration import SurfaceCalibrationEngine
from core.ModelEngine import ModelEngine
from core.plotting import (
    generate_surface_from_params, 
    calculate_fit_statistics, 
    plot_surface_comparison,
    plot_training_history,
    plot_parameter_distribution,
    plot_surface_evolution,
    PLOTTING_AVAILABLE
)
from data.storage import StorageFactory
from infrastructure.settings import Settings
from infrastructure.utils.logging import setup_logger


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Surface calibration for volatility models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Main operation mode
    parser.add_argument(
        '--mode', 
        choices=['train', 'calibrate', 'evaluate'],
        required=True,
        help='Operation mode: train model, calibrate surfaces, or evaluate performance'
    )
    
    # Database settings
    parser.add_argument(
        '--db-url',
        help='Database connection URL (uses settings.py default if not provided)'
    )
    
    parser.add_argument(
        '--instrument',
        help='Instrument ticker to calibrate (e.g., BTC, ETH, SPX). If not specified, uses all instruments'
    )
    
    # Training options
    parser.add_argument(
        '--use-market-data',
        action='store_true',
        help='Use market data from database for training'
    )
    
    parser.add_argument(
        '--n-synthetic',
        type=int,
        default=1000,
        help='Number of synthetic surfaces to generate for training'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Training batch size'
    )
    
    # Calibration options
    parser.add_argument(
        '--surface-limit',
        type=int,
        default=50,
        help='Number of latest surfaces to calibrate'
    )
    
    parser.add_argument(
        '--calibration-method',
        choices=['cnn', 'optimization'],
        default='cnn',
        help='Calibration method to use'
    )
    
    # Model options
    parser.add_argument(
        '--model-name',
        default='rBergomi_CNN',
        help='Model name for database storage'
    )
    
    parser.add_argument(
        '--model-path',
        default='./models/rbergomi_cnn',
        help='Path to save/load model files'
    )
    
    # Grid parameters
    parser.add_argument(
        '--maturities',
        nargs='+',
        type=float,
        default=[0.1, 0.3, 0.6, 1.0],
        help='Maturity points in years'
    )
    
    parser.add_argument(
        '--moneyness-min',
        type=float,
        default=0.8,
        help='Minimum moneyness value'
    )
    
    parser.add_argument(
        '--moneyness-max',
        type=float,
        default=1.2,
        help='Maximum moneyness value'
    )
    
    parser.add_argument(
        '--moneyness-points',
        type=int,
        default=11,
        help='Number of moneyness points'
    )
    
    # Output options
    parser.add_argument(
        '--output-file',
        help='Output file for calibration results (JSON format)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    # Plotting options
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate surface plots and visualizations'
    )
    
    parser.add_argument(
        '--plot-limit',
        type=int,
        default=3,
        help='Number of surfaces to plot (for performance)'
    )
    
    parser.add_argument(
        '--plot-dir',
        default='./plots',
        help='Directory to save plot files'
    )
    
    return parser.parse_args()


def setup_calibration_engine(args):
    """Set up the calibration engine with specified parameters"""
    # Get database connection
    settings = Settings()
    if args.db_url:
        settings.STORAGE.POSTGRES_URI = args.db_url
    store = StorageFactory.create_storage(settings)
    
    # Initialize calibration engine with instrument filter
    engine = SurfaceCalibrationEngine(
        store, 
        model_name=args.model_name,
        instrument=args.instrument
    )
    
    # Set custom grid parameters
    import numpy as np
    engine.maturities = np.array(args.maturities)
    engine.moneyness_range = np.linspace(
        args.moneyness_min, 
        args.moneyness_max, 
        args.moneyness_points
    )
    
    return engine


def train_mode(args):
    """Train the CNN model"""
    logger = setup_logger("calibration_train")
    logger.info("Starting model training...")
    
    engine = setup_calibration_engine(args)
    
    # Create model directory
    Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Train model
        if args.use_market_data:
            logger.info("Training with market data + synthetic augmentation")
            history = engine.train_model_with_market_data(
                use_market_data=True,
                n_synthetic=args.n_synthetic,
                epochs=args.epochs
            )
        else:
            logger.info("Training with synthetic data only")
            X, Y = engine.generate_synthetic_data(args.n_synthetic)
            history = engine.train_model(
                X, Y, 
                epochs=args.epochs,
                batch_size=args.batch_size
            )
        
        # Save model
        engine.save_model(args.model_path)
        logger.info(f"Model saved to {args.model_path}")
        
        # Save training history
        history_file = f"{args.model_path}_history.json"
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
        logger.info(f"Training history saved to {history_file}")
        
        print(f"Training completed successfully!")
        print(f"Final validation loss: {history['val_loss'][-1]:.6f}")
        print(f"Final validation MAE: {history['val_mae'][-1]:.6f}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


def calibrate_mode(args):
    """Calibrate recent volatility surfaces"""
    logger = setup_logger("calibration_calibrate")
    if args.instrument:
        logger.info(f"Starting surface calibration for {args.instrument}...")
    else:
        logger.info("Starting surface calibration for all instruments...")
    
    # Create plot directory if plotting is enabled
    if args.plot and PLOTTING_AVAILABLE:
        Path(args.plot_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Plots will be saved to {args.plot_dir}")
    
    engine = setup_calibration_engine(args)
    
    # Load trained model
    try:
        engine.load_model(args.model_path)
        logger.info(f"Model loaded from {args.model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        if args.calibration_method == 'cnn':
            logger.info("Falling back to optimization method")
            args.calibration_method = 'optimization'
    
    # Get recent surfaces (filtered by instrument if specified)
    surfaces = engine.store.get_last_n_surfaces(limit=args.surface_limit, asset_id=engine.asset_id)
    if not surfaces:
        if args.instrument:
            logger.error(f"No surfaces found for instrument {args.instrument}")
        else:
            logger.error("No surfaces found in database")
        return
    
    if args.instrument:
        logger.info(f"Calibrating {len(surfaces)} {args.instrument} surfaces using {args.calibration_method} method")
    else:
        logger.info(f"Calibrating {len(surfaces)} surfaces using {args.calibration_method} method")
    
    results = []
    for i, surface in enumerate(surfaces):
        try:
            # Convert surface to grid
            grid_surface = engine._surface_to_grid(surface)
            if grid_surface is None:
                logger.warning(f"Skipping surface {i+1}: invalid data")
                continue
            
            # Calibrate parameters
            params = engine.calibrate_surface(grid_surface, method=args.calibration_method)
            
            # Store in database if we have surface ID
            if hasattr(surface, 'asset_id') and surface.asset_id:
                param_id = engine.store_calibrated_parameters(
                    params, 
                    surface_id=getattr(surface, 'id', i),
                    asset_id=surface.asset_id
                )
                params['parameter_id'] = param_id
            elif engine.asset_id:
                param_id = engine.store_calibrated_parameters(
                    params, 
                    surface_id=getattr(surface, 'id', i),
                    asset_id=engine.asset_id
                )
                params['parameter_id'] = param_id
            
            # Add metadata
            params.update({
                'timestamp': surface.timestamp.isoformat(),
                'surface_id': getattr(surface, 'id', i),
                'asset_id': getattr(surface, 'asset_id', None),
                'calibration_method': args.calibration_method
            })
            
            results.append(params)
            
            logger.info(f"Surface {i+1}/{len(surfaces)}: H={params['H']:.4f}, "
                       f"nu={params['nu']:.4f}, rho={params['rho']:.4f}")
            
            # Generate plots if requested
            if args.plot and PLOTTING_AVAILABLE and i < args.plot_limit:
                try:
                    logger.info(f"Generating plot for surface {i+1}...")
                    
                    # Import numpy for calculations
                    import numpy as np
                    
                    # Get surface size for dimension handling
                    surface_size = len(grid_surface)
                    
                    # Reshape original surface to 2D for comparison
                    if surface_size == 44:  # Fixed grid (4 x 11)
                        original_2d = grid_surface.reshape(len(engine.maturities), len(engine.moneyness_range))
                    else:
                        # For adaptive grids, use reasonable dimensions
                        n_maturities = 7  # Most common in the logs
                        n_moneyness = surface_size // n_maturities
                        if surface_size % n_maturities != 0:
                            n_maturities = int(np.sqrt(surface_size))
                            n_moneyness = surface_size // n_maturities
                        original_2d = grid_surface.reshape(n_maturities, n_moneyness)
                    
                    # Generate surface from calibrated parameters
                    # Always use the same dimensions as the original surface
                    n_maturities, n_moneyness = original_2d.shape
                    # Create maturity and moneyness ranges to match the original surface
                    maturities_adaptive = np.linspace(0.1, 1.0, n_maturities)
                    moneyness_adaptive = np.linspace(0.8, 1.2, n_moneyness)
                    generated_surface = generate_surface_from_params(
                        H=params['H'],
                        nu=params['nu'], 
                        rho=params['rho'],
                        maturities=maturities_adaptive,
                        moneyness=moneyness_adaptive
                    )
                    
                    # Ensure both surfaces have the same dimensions for plotting
                    if original_2d.shape != generated_surface.shape:
                        logger.info(f"   Reshaping generated surface from {generated_surface.shape} to {original_2d.shape}")
                        # Resize generated surface to match original
                        zoom_factors = (original_2d.shape[0] / generated_surface.shape[0], 
                                      original_2d.shape[1] / generated_surface.shape[1])
                        generated_surface = zoom(generated_surface, zoom_factors, order=1)
                    
                    # Calculate fit statistics
                    stats = calculate_fit_statistics(original_2d, generated_surface)
                    
                    # Create plot
                    surface_id = getattr(surface, 'id', i+1)
                    instrument_name = args.instrument or getattr(surface, 'asset_id', 'UNKNOWN')
                    plot_surface_comparison(
                        original_2d, generated_surface,
                        maturities_adaptive, moneyness_adaptive,
                        params, stats, instrument_name, surface_id, args.plot_dir
                    )
                    
                except Exception as e:
                    logger.warning(f"Failed to generate plot for surface {i+1}: {e}")
            
        except Exception as e:
            logger.error(f"Failed to calibrate surface {i+1}: {e}")
            continue
    
    # Save results
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {args.output_file}")
    
    if args.instrument:
        print(f"Calibration completed for {args.instrument}: {len(results)}/{len(surfaces)} surfaces processed")
    else:
        print(f"Calibration completed: {len(results)}/{len(surfaces)} surfaces processed")
    
    # Print summary statistics
    if results:
        import numpy as np
        H_values = [r['H'] for r in results]
        nu_values = [r['nu'] for r in results]
        rho_values = [r['rho'] for r in results]
        
        print(f"\nParameter Summary:")
        print(f"H:   mean={np.mean(H_values):.4f}, std={np.std(H_values):.4f}")
        print(f"nu:  mean={np.mean(nu_values):.4f}, std={np.std(nu_values):.4f}")
        print(f"rho: mean={np.mean(rho_values):.4f}, std={np.std(rho_values):.4f}")


def evaluate_mode(args):
    """Evaluate model performance"""
    logger = setup_logger("calibration_evaluate")
    logger.info("Starting model evaluation...")
    
    engine = setup_calibration_engine(args)
    
    # Load model
    try:
        engine.load_model(args.model_path)
        logger.info(f"Model loaded from {args.model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # Generate test data
    logger.info("Generating test dataset...")
    X_test, Y_test = engine.generate_synthetic_data(500)  # Test on synthetic data
    
    # Evaluate on test set
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import numpy as np
    
    predictions = []
    for i in range(len(Y_test)):
        try:
            pred_params = engine.calibrate_surface(Y_test[i], method='cnn')
            predictions.append([pred_params['H'], pred_params['nu'], pred_params['rho']])
        except Exception as e:
            logger.warning(f"Failed to predict for test sample {i}: {e}")
            continue
    
    predictions = np.array(predictions)
    
    if len(predictions) > 0:
        # Calculate metrics
        mse = mean_squared_error(X_test[:len(predictions)], predictions)
        mae = mean_absolute_error(X_test[:len(predictions)], predictions)
        
        # Parameter-wise metrics
        param_names = ['H', 'nu', 'rho']
        for i, name in enumerate(param_names):
            true_vals = X_test[:len(predictions), i]
            pred_vals = predictions[:, i]
            param_mse = mean_squared_error(true_vals, pred_vals)
            param_mae = mean_absolute_error(true_vals, pred_vals)
            
            print(f"{name}: MSE={param_mse:.6f}, MAE={param_mae:.6f}")
        
        print(f"\nOverall: MSE={mse:.6f}, MAE={mae:.6f}")
        print(f"Evaluated {len(predictions)}/{len(Y_test)} test samples")
        
    else:
        logger.error("No valid predictions generated")


def main():
    """Main entry point"""
    args = parse_args()
    
    # Set up logging level
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        if args.mode == 'train':
            train_mode(args)
        elif args.mode == 'calibrate':
            calibrate_mode(args)
        elif args.mode == 'evaluate':
            evaluate_mode(args)
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 