#!/usr/bin/env python3
"""
Simple example of using the surface calibration system with market data.

This example shows the hybrid approach:
1. Train CNN using market data from your database + synthetic augmentation
2. Calibrate recent surfaces quickly using the trained model
3. Visualize generated surfaces and analyze model fit statistics
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.SurfaceCalibration import SurfaceCalibrationEngine
from core.ModelEngine import rBergomi
from data.storage import StorageFactory
from infrastructure.settings import Settings
from infrastructure.utils.logging import setup_logger


def generate_surface_from_params(H: float, nu: float, rho: float, 
                                maturities: np.ndarray, moneyness: np.ndarray,
                                xi: float = 0.04, S0: float = 1.0) -> np.ndarray:
    """
    Generate a volatility surface using rBergomi model with given parameters.
    
    Args:
        H: Hurst parameter
        nu: Volatility of volatility
        rho: Correlation parameter
        maturities: Array of maturities in years
        moneyness: Array of moneyness values (K/S0)
        xi: Initial variance level
        S0: Initial stock price
        
    Returns:
        Implied volatility surface as 2D array
    """
    # Convert H to alpha parameter for rBergomi
    a = H - 0.5
    
    # Create rBergomi model
    T_max = max(maturities)
    model = rBergomi(n=100, N=5000, T=T_max, a=a)
    
    # Generate surface
    surface = np.zeros((len(maturities), len(moneyness)))
    
    for i, T in enumerate(maturities):
        for j, m in enumerate(moneyness):
            K = m * S0  # Strike price
            try:
                # Price option with rBergomi
                price = model.price_european_option(
                    K=K, T=T, xi=xi, eta=nu, rho=rho, S0=S0, r=0.0
                )
                
                # Extract implied volatility
                iv = extract_implied_vol(price, S0, K, T, r=0.0)
                surface[i, j] = iv
            except:
                # Fallback if pricing fails
                surface[i, j] = 0.2
    
    return surface


def extract_implied_vol(price: float, S0: float, K: float, T: float, r: float = 0.0) -> float:
    """
    Extract implied volatility from option price using Newton-Raphson method.
    
    Args:
        price: Option price
        S0: Current stock price
        K: Strike price
        T: Time to maturity
        r: Risk-free rate
        
    Returns:
        Implied volatility
    """
    from scipy.stats import norm
    
    def black_scholes(S, K, T, r, sigma, option_type='call'):
        """Black-Scholes formula"""
        if T <= 0:
            return max(S - K, 0) if option_type == 'call' else max(K - S, 0)
        
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if option_type == 'call':
            return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        else:
            return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    
    def vega(S, K, T, r, sigma):
        """Vega of Black-Scholes option"""
        if T <= 0:
            return 0
        
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        return S * np.sqrt(T) * norm.pdf(d1)
    
    # Newton-Raphson to find implied volatility
    sigma = 0.3  # Initial guess
    max_iter = 100
    tolerance = 1e-6
    
    for _ in range(max_iter):
        bs_price = black_scholes(S0, K, T, r, sigma)
        v = vega(S0, K, T, r, sigma)
        
        diff = price - bs_price
        if abs(diff) < tolerance:
            break
            
        sigma = sigma + diff / v
        sigma = max(0.001, min(5.0, sigma))  # Clamp to reasonable range
    
    return sigma


def calculate_fit_statistics(original_surface: np.ndarray, generated_surface: np.ndarray) -> dict:
    """
    Calculate statistics to measure how well the generated surface fits the original.
    
    Args:
        original_surface: Original market surface
        generated_surface: Surface generated from calibrated parameters
        
    Returns:
        Dictionary with fit statistics
    """
    # Flatten surfaces for comparison
    orig_flat = original_surface.flatten()
    gen_flat = generated_surface.flatten()
    
    # Remove NaN values
    valid_mask = ~(np.isnan(orig_flat) | np.isnan(gen_flat))
    orig_valid = orig_flat[valid_mask]
    gen_valid = gen_flat[valid_mask]
    
    if len(orig_valid) == 0:
        return {
            'mse': np.nan,
            'mae': np.nan,
            'rmse': np.nan,
            'r_squared': np.nan,
            'correlation': np.nan,
            'max_error': np.nan,
            'mean_abs_percent_error': np.nan
        }
    
    # Calculate statistics
    mse = np.mean((orig_valid - gen_valid) ** 2)
    mae = np.mean(np.abs(orig_valid - gen_valid))
    rmse = np.sqrt(mse)
    
    # R-squared
    ss_res = np.sum((orig_valid - gen_valid) ** 2)
    ss_tot = np.sum((orig_valid - np.mean(orig_valid)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Correlation
    correlation = np.corrcoef(orig_valid, gen_valid)[0, 1] if len(orig_valid) > 1 else 0
    
    # Max error
    max_error = np.max(np.abs(orig_valid - gen_valid))
    
    # Mean absolute percentage error
    mean_abs_percent_error = np.mean(np.abs((orig_valid - gen_valid) / orig_valid)) * 100
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r_squared': r_squared,
        'correlation': correlation,
        'max_error': max_error,
        'mean_abs_percent_error': mean_abs_percent_error
    }


def plot_surface_comparison(original_surface: np.ndarray, generated_surface: np.ndarray,
                           maturities: np.ndarray, moneyness: np.ndarray,
                           params: dict, stats: dict, instrument: str):
    """
    Create a comparison plot showing original vs generated surfaces.
    
    Args:
        original_surface: Original market surface
        generated_surface: Surface generated from calibrated parameters
        maturities: Maturity values
        moneyness: Moneyness values
        params: Calibrated parameters
        stats: Fit statistics
        instrument: Instrument name
    """
    fig = plt.figure(figsize=(15, 10))
    
    # Create meshgrid for plotting
    M, T = np.meshgrid(moneyness, maturities)
    
    # Original surface
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    surf1 = ax1.plot_surface(M, T, original_surface, cmap=cm.viridis, alpha=0.8)
    ax1.set_title(f'Original {instrument} Surface')
    ax1.set_xlabel('Moneyness')
    ax1.set_ylabel('Maturity (years)')
    ax1.set_zlabel('Implied Volatility')
    
    # Generated surface
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    surf2 = ax2.plot_surface(M, T, generated_surface, cmap=cm.viridis, alpha=0.8)
    ax2.set_title(f'Generated Surface (H={params["H"]:.3f}, ν={params["nu"]:.3f}, ρ={params["rho"]:.3f})')
    ax2.set_xlabel('Moneyness')
    ax2.set_ylabel('Maturity (years)')
    ax2.set_zlabel('Implied Volatility')
    
    # Difference surface
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    diff_surface = original_surface - generated_surface
    surf3 = ax3.plot_surface(M, T, diff_surface, cmap=cm.RdBu_r, alpha=0.8)
    ax3.set_title('Difference (Original - Generated)')
    ax3.set_xlabel('Moneyness')
    ax3.set_ylabel('Maturity (years)')
    ax3.set_zlabel('Volatility Difference')
    
    # Heatmap comparison
    ax4 = fig.add_subplot(2, 3, 4)
    im4 = ax4.imshow(original_surface, cmap=cm.viridis, aspect='auto', 
                     extent=[moneyness.min(), moneyness.max(), maturities.min(), maturities.max()])
    ax4.set_title('Original Surface (Heatmap)')
    ax4.set_xlabel('Moneyness')
    ax4.set_ylabel('Maturity (years)')
    plt.colorbar(im4, ax=ax4)
    
    ax5 = fig.add_subplot(2, 3, 5)
    im5 = ax5.imshow(generated_surface, cmap=cm.viridis, aspect='auto',
                     extent=[moneyness.min(), moneyness.max(), maturities.min(), maturities.max()])
    ax5.set_title('Generated Surface (Heatmap)')
    ax5.set_xlabel('Moneyness')
    ax5.set_ylabel('Maturity (years)')
    plt.colorbar(im5, ax=ax5)
    
    # Statistics text
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    stats_text = f"""Fit Statistics:
    
R² = {stats['r_squared']:.4f}
RMSE = {stats['rmse']:.4f}
MAE = {stats['mae']:.4f}
Correlation = {stats['correlation']:.4f}
Max Error = {stats['max_error']:.4f}
MAPE = {stats['mean_abs_percent_error']:.2f}%

Calibrated Parameters:
H (Hurst) = {params['H']:.4f}
ν (Vol-of-vol) = {params['nu']:.4f}
ρ (Correlation) = {params['rho']:.4f}"""
    
    ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{instrument}_surface_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Simple calibration example with visualization"""
    logger = setup_logger("calibration_example")
    
    # Connect to your database
    settings = Settings()
    store = StorageFactory.create_storage(settings)
    
    # Example: Calibrate for ETH (change to your instrument)
    instrument = "ETH"  # Change this to your instrument ticker
    
    # Initialize calibration engine
    engine = SurfaceCalibrationEngine(
        store, 
        model_name="rBergomi_Example",
        instrument=instrument
    )
    
    # Ensure maturities are set (needed for market data processing)
    if not hasattr(engine, 'maturities'):
        engine.maturities = np.array([0.1, 0.3, 0.6, 1.0])
        engine.moneyness_range = np.linspace(0.8, 1.2, 11)
    
    print(f"🔍 Checking available data for {instrument}...")
    
    # Check what data we have
    surfaces = store.get_last_n_surfaces(limit=10, asset_id=engine.asset_id)
    if not surfaces:
        print(f"❌ No volatility surfaces found for {instrument}")
        print("💡 Try a different instrument or make sure you have collected market data")
        print("💡 Available instruments can be checked with: SELECT ticker FROM assets;")
        return
    
    print(f"✅ Found {len(surfaces)} recent volatility surfaces for {instrument}")
    
    # Example 1: Market data approach (recommended)
    print(f"\n🤖 Training model with {instrument} market data approach...")
    
    try:
        # Train using market data + synthetic augmentation
        history = engine.train_model_with_market_data(
            use_market_data=True,
            n_synthetic=200,  # Small number for quick demo
            epochs=20         # Fewer epochs for demo
        )
        
        print(f"✅ Training completed for {instrument}!")
        print(f"   Final validation loss: {history['val_loss'][-1]:.6f}")
        
        # Save the model
        model_path = f"./models/{instrument}_rbergomi"
        engine.save_model(model_path)
        print(f"💾 Model saved to {model_path}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        print(f"   Debug: {traceback.format_exc()}")
        
        print("💡 Falling back to optimization-only approach")
        
        # Example 2: Fallback to optimization-only
        print("\n🔧 Using optimization-only calibration...")
        
        # Calibrate just one surface as example
        surface = surfaces[0]
        grid_surface = engine._surface_to_grid(surface)
        
        if grid_surface is not None:
            params = engine.calibrate_surface(grid_surface, method='optimization')
            print(f"✅ Calibrated parameters for latest {instrument} surface:")
            print(f"   H (Hurst): {params['H']:.4f}")
            print(f"   nu (Vol-of-vol): {params['nu']:.4f}")
            print(f"   rho (Correlation): {params['rho']:.4f}")
            
            # Store results
            if hasattr(surface, 'asset_id'):
                param_id = engine.store_calibrated_parameters(
                    params, 
                    surface_id=getattr(surface, 'id', 1),
                    asset_id=surface.asset_id
                )
                print(f"💾 Parameters stored with ID: {param_id}")
        
        return
    
    # Example 3: Fast calibration using trained CNN with visualization
    print("\n⚡ Fast calibration using trained CNN...")
    
    # Calibrate recent surfaces and analyze fit
    results = []
    fit_statistics = []
    
    for i, surface in enumerate(surfaces[:3]):  # Just first 3 for demo
        try:
            grid_surface = engine._surface_to_grid(surface)
            if grid_surface is None:
                continue
                
            # Fast CNN calibration
            params = engine.calibrate_surface(grid_surface, method='cnn')
            
            print(f"\n📊 Surface {i+1} Analysis:")
            print(f"   Calibrated: H={params['H']:.3f}, nu={params['nu']:.3f}, rho={params['rho']:.3f}")
            
            # Get surface size for dimension handling
            surface_size = len(grid_surface)
            
            # Generate surface from calibrated parameters
            # Use the same dimensions as the original surface
            if surface_size == 44:  # Fixed grid
                generated_surface = generate_surface_from_params(
                    H=params['H'],
                    nu=params['nu'], 
                    rho=params['rho'],
                    maturities=engine.maturities,
                    moneyness=engine.moneyness_range
                )
            else:
                # For adaptive grids, use the actual dimensions
                n_maturities = 7  # Most common in the logs
                n_moneyness = surface_size // n_maturities
                if surface_size % n_maturities != 0:
                    padded_size = n_maturities * n_moneyness
                    if padded_size < surface_size:
                        padded_size = n_maturities * (n_moneyness + 1)
                    n_moneyness = padded_size // n_maturities
                
                # Create adaptive maturities and moneyness ranges
                adaptive_maturities = np.linspace(0.1, 1.0, n_maturities)
                adaptive_moneyness = np.linspace(0.8, 1.2, n_moneyness)
                
                generated_surface = generate_surface_from_params(
                    H=params['H'],
                    nu=params['nu'], 
                    rho=params['rho'],
                    maturities=adaptive_maturities,
                    moneyness=adaptive_moneyness
                )
            
            # Reshape original surface to 2D for comparison
            # Use the actual grid dimensions from the surface
            surface_size = len(grid_surface)
            if surface_size == 44:  # Fixed grid (4 x 11)
                original_2d = grid_surface.reshape(len(engine.maturities), len(engine.moneyness_range))
            else:
                # For adaptive grids, we need to get the actual dimensions
                # This is a simplified approach - in practice you'd want to store the dimensions
                # For now, let's use a reasonable approximation
                n_maturities = 7  # Most common in the logs
                n_moneyness = surface_size // n_maturities
                if surface_size % n_maturities != 0:
                    # If not perfectly divisible, pad with the last value
                    padded_size = n_maturities * n_moneyness
                    if padded_size < surface_size:
                        padded_size = n_maturities * (n_moneyness + 1)
                    padded_surface = np.full(padded_size, grid_surface[-1])
                    padded_surface[:surface_size] = grid_surface
                    grid_surface = padded_surface
                    n_moneyness = padded_size // n_maturities
                
                original_2d = grid_surface.reshape(n_maturities, n_moneyness)
            
            # Calculate fit statistics
            stats = calculate_fit_statistics(original_2d, generated_surface)
            fit_statistics.append(stats)
            
            print(f"   Fit Quality: R²={stats['r_squared']:.4f}, RMSE={stats['rmse']:.4f}")
            print(f"   Correlation: {stats['correlation']:.4f}")
            print(f"   Mean Abs % Error: {stats['mean_abs_percent_error']:.2f}%")
            
            results.append(params)
            
            # Plot comparison for the first surface
            if i == 0:
                print(f"\n🎨 Generating visualization for Surface {i+1}...")
                # Ensure both surfaces have the same dimensions for plotting
                if original_2d.shape != generated_surface.shape:
                    print(f"   Reshaping generated surface from {generated_surface.shape} to {original_2d.shape}")
                    # Resize generated surface to match original
                    from scipy.ndimage import zoom
                    zoom_factors = (original_2d.shape[0] / generated_surface.shape[0], 
                                   original_2d.shape[1] / generated_surface.shape[1])
                    generated_surface = zoom(generated_surface, zoom_factors, order=1)
                
                # Use the actual dimensions for plotting
                if surface_size == 44:  # Fixed grid
                    plot_maturities = engine.maturities
                    plot_moneyness = engine.moneyness_range
                else:
                    # For adaptive grids, use the actual dimensions
                    plot_maturities = np.linspace(0.1, 1.0, original_2d.shape[0])
                    plot_moneyness = np.linspace(0.8, 1.2, original_2d.shape[1])
                
                plot_surface_comparison(
                    original_2d, generated_surface,
                    plot_maturities, plot_moneyness,
                    params, stats, instrument
                )
            
        except Exception as e:
            print(f"   Surface {i+1}: Failed ({e})")
    
    if results and fit_statistics:
        # Summary statistics across all surfaces
        print(f"\n📈 Overall Model Performance for {instrument}:")
        print("=" * 60)
        
        # Parameter statistics
        H_vals = [r['H'] for r in results]
        nu_vals = [r['nu'] for r in results]
        rho_vals = [r['rho'] for r in results]
        
        print(f"Parameter Ranges:")
        print(f"   H (Hurst): {np.mean(H_vals):.4f} ± {np.std(H_vals):.4f} [{np.min(H_vals):.4f}, {np.max(H_vals):.4f}]")
        print(f"   ν (Vol-of-vol): {np.mean(nu_vals):.4f} ± {np.std(nu_vals):.4f} [{np.min(nu_vals):.4f}, {np.max(nu_vals):.4f}]")
        print(f"   ρ (Correlation): {np.mean(rho_vals):.4f} ± {np.std(rho_vals):.4f} [{np.min(rho_vals):.4f}, {np.max(rho_vals):.4f}]")
        
        # Fit quality statistics
        r2_vals = [s['r_squared'] for s in fit_statistics]
        rmse_vals = [s['rmse'] for s in fit_statistics]
        corr_vals = [s['correlation'] for s in fit_statistics]
        mape_vals = [s['mean_abs_percent_error'] for s in fit_statistics]
        
        print(f"\nFit Quality Statistics:")
        print(f"   R²: {np.mean(r2_vals):.4f} ± {np.std(r2_vals):.4f} [{np.min(r2_vals):.4f}, {np.max(r2_vals):.4f}]")
        print(f"   RMSE: {np.mean(rmse_vals):.4f} ± {np.std(rmse_vals):.4f} [{np.min(rmse_vals):.4f}, {np.max(rmse_vals):.4f}]")
        print(f"   Correlation: {np.mean(corr_vals):.4f} ± {np.std(corr_vals):.4f} [{np.min(corr_vals):.4f}, {np.max(corr_vals):.4f}]")
        print(f"   MAPE: {np.mean(mape_vals):.2f}% ± {np.std(mape_vals):.2f}% [{np.min(mape_vals):.2f}%, {np.max(mape_vals):.2f}%]")
        
        # Model interpretation
        print(f"\n🔍 Model Interpretation:")
        avg_H = np.mean(H_vals)
        avg_nu = np.mean(nu_vals)
        avg_rho = np.mean(rho_vals)
        
        print(f"   Hurst Parameter (H={avg_H:.3f}): ", end="")
        if avg_H < 0.1:
            print("Very rough volatility (strong mean reversion)")
        elif avg_H < 0.2:
            print("Rough volatility (moderate mean reversion)")
        else:
            print("Smooth volatility (weak mean reversion)")
            
        print(f"   Vol-of-Vol (ν={avg_nu:.3f}): ", end="")
        if avg_nu < 1.0:
            print("Low volatility clustering")
        elif avg_nu < 2.0:
            print("Moderate volatility clustering")
        else:
            print("High volatility clustering")
            
        print(f"   Correlation (ρ={avg_rho:.3f}): ", end="")
        if avg_rho < -0.5:
            print("Strong leverage effect (volatility increases when price falls)")
        elif avg_rho < -0.2:
            print("Moderate leverage effect")
        else:
            print("Weak or positive leverage effect")
    
    print("\n🎉 Calibration example completed!")
    print("\n💡 Next steps:")
    print(f"   - Run full training: python calibrate_surface.py --mode train --use-market-data --instrument {instrument}")
    print(f"   - Calibrate all surfaces: python calibrate_surface.py --mode calibrate --instrument {instrument}")
    print("   - Try different instruments: --instrument ETH, --instrument SPX, etc.")
    print("   - See README_calibration.md for more options")


if __name__ == "__main__":
    main() 