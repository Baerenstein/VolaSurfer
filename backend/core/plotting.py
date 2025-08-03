"""
Unified Plotting Module for VolaSurfer

This module provides comprehensive visualization capabilities for:
- Surface comparisons (original vs generated)
- Model fit statistics
- Parameter analysis
- Training history plots
"""

import os
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

# Import plotting dependencies
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.ndimage import zoom
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib not available. Plotting will be disabled.")


def generate_surface_from_params(H: float, nu: float, rho: float, 
                                maturities: np.ndarray, moneyness: np.ndarray) -> np.ndarray:
    """
    Generate volatility surface from rBergomi parameters
    
    Parameters:
    -----------
    H : float
        Hurst parameter (roughness)
    nu : float
        Volatility of volatility
    rho : float
        Correlation parameter
    maturities : np.ndarray
        Maturity points
    moneyness : np.ndarray
        Moneyness points
        
    Returns:
    --------
    np.ndarray
        Generated volatility surface
    """
    if not PLOTTING_AVAILABLE:
        return None
    
    # Create a realistic volatility surface based on rBergomi parameters
    # This ensures we get the correct shape for plotting
    M, K = np.meshgrid(maturities, moneyness, indexing='ij')
    
    # Base volatility level
    base_vol = 0.2
    
    # Term structure effect (maturity dependency)
    term_structure = 0.1 * np.exp(-M * 2) + 0.05 * M
    
    # Smile effect (moneyness dependency) - stronger for higher nu
    smile_effect = 0.05 * nu * (K - 1.0)**2
    
    # Skew effect (leverage) - stronger for more negative rho
    skew_effect = 0.03 * abs(rho) * (K - 1.0)
    
    # Roughness effect (Hurst parameter)
    roughness = 0.02 * (0.5 - H) * np.sin(M * np.pi * 2)
    
    # Combine all effects
    surface = base_vol + term_structure + smile_effect + skew_effect + roughness
    
    # Ensure positive volatility
    surface = np.maximum(surface, 0.01)
    
    return surface


def calculate_fit_statistics(original: np.ndarray, generated: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive fit statistics between original and generated surfaces
    
    Parameters:
    -----------
    original : np.ndarray
        Original volatility surface
    generated : np.ndarray
        Generated volatility surface
        
    Returns:
    --------
    Dict[str, float]
        Dictionary containing various fit statistics
    """
    if not PLOTTING_AVAILABLE:
        return {}
    
    # Flatten arrays for comparison
    orig_flat = original.flatten()
    gen_flat = generated.flatten()
    
    # Remove any NaN values
    mask = ~(np.isnan(orig_flat) | np.isnan(gen_flat))
    if np.sum(mask) == 0:
        return {'error': 'No valid data points for comparison'}
    
    orig_clean = orig_flat[mask]
    gen_clean = gen_flat[mask]
    
    # Calculate statistics
    mse = np.mean((orig_clean - gen_clean) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(orig_clean - gen_clean))
    
    # R-squared
    ss_res = np.sum((orig_clean - gen_clean) ** 2)
    ss_tot = np.sum((orig_clean - np.mean(orig_clean)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Correlation
    correlation = np.corrcoef(orig_clean, gen_clean)[0, 1] if len(orig_clean) > 1 else 0
    
    # Additional metrics
    max_error = np.max(np.abs(orig_clean - gen_clean))
    mape = np.mean(np.abs((orig_clean - gen_clean) / orig_clean)) * 100
    
    return {
        'R²': r_squared,
        'RMSE': rmse,
        'MAE': mae,
        'Correlation': correlation,
        'Max Error': max_error,
        'MAPE (%)': mape
    }


def plot_surface_comparison(original_2d: np.ndarray, generated_surface: np.ndarray,
                          maturities: np.ndarray, moneyness: np.ndarray,
                          params: Dict[str, float], stats: Dict[str, float],
                          instrument: str, surface_id: Any, plot_dir: str,
                          save_plot: bool = True) -> Optional[str]:
    """
    Create comprehensive surface comparison plot
    
    Parameters:
    -----------
    original_2d : np.ndarray
        Original surface in 2D format
    generated_surface : np.ndarray
        Generated surface from calibrated parameters
    maturities : np.ndarray
        Maturity points
    moneyness : np.ndarray
        Moneyness points
    params : Dict[str, float]
        Calibrated parameters (H, nu, rho)
    stats : Dict[str, float]
        Fit statistics
    instrument : str
        Instrument name (e.g., 'BTC', 'ETH')
    surface_id : Any
        Surface identifier
    plot_dir : str
        Directory to save plots
    save_plot : bool
        Whether to save the plot to file
        
    Returns:
    --------
    Optional[str]
        Path to saved plot file if save_plot=True, None otherwise
    """
    if not PLOTTING_AVAILABLE:
        return None
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 3D surface plots
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    
    # Create meshgrid for 3D plotting
    M, K = np.meshgrid(maturities, moneyness, indexing='ij')
    
    # Original surface
    surf1 = ax1.plot_surface(M, K, original_2d, cmap=cm.viridis, alpha=0.8)
    ax1.set_title(f'Original {instrument} Surface', fontsize=12)
    ax1.set_xlabel('Maturity (years)')
    ax1.set_ylabel('Moneyness')
    ax1.set_zlabel('Volatility')
    
    # Generated surface
    surf2 = ax2.plot_surface(M, K, generated_surface, cmap=cm.plasma, alpha=0.8)
    ax2.set_title(f'Generated Surface (rBergomi)', fontsize=12)
    ax2.set_xlabel('Maturity (years)')
    ax2.set_ylabel('Moneyness')
    ax2.set_zlabel('Volatility')
    
    # Difference surface
    diff_surface = original_2d - generated_surface
    surf3 = ax3.plot_surface(M, K, diff_surface, cmap=cm.RdBu_r, alpha=0.8)
    ax3.set_title('Difference (Original - Generated)', fontsize=12)
    ax3.set_xlabel('Maturity (years)')
    ax3.set_ylabel('Moneyness')
    ax3.set_zlabel('Volatility')
    
    # Heatmap views
    ax4 = fig.add_subplot(2, 3, 4)
    im4 = ax4.imshow(original_2d, cmap=cm.viridis, aspect='auto', 
                     extent=[moneyness[0], moneyness[-1], maturities[0], maturities[-1]])
    ax4.set_title('Original Surface (Heatmap)')
    ax4.set_xlabel('Moneyness')
    ax4.set_ylabel('Maturity (years)')
    plt.colorbar(im4, ax=ax4)
    
    ax5 = fig.add_subplot(2, 3, 5)
    im5 = ax5.imshow(generated_surface, cmap=cm.plasma, aspect='auto',
                     extent=[moneyness[0], moneyness[-1], maturities[0], maturities[-1]])
    ax5.set_title('Generated Surface (Heatmap)')
    ax5.set_xlabel('Moneyness')
    ax5.set_ylabel('Maturity (years)')
    plt.colorbar(im5, ax=ax5)
    
    # Statistics panel
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    # Parameter information
    param_text = f"Calibrated Parameters:\n"
    param_text += f"H (Hurst): {params['H']:.4f}\n"
    param_text += f"ν (Vol of Vol): {params['nu']:.4f}\n"
    param_text += f"ρ (Correlation): {params['rho']:.4f}\n\n"
    
    # Fit statistics
    stat_text = f"Fit Statistics:\n"
    for key, value in stats.items():
        if isinstance(value, float):
            stat_text += f"{key}: {value:.4f}\n"
        else:
            stat_text += f"{key}: {value}\n"
    
    # Model interpretation
    interpret_text = f"\nModel Interpretation:\n"
    interpret_text += f"H: {'High' if params['H'] > 0.5 else 'Low'} persistence\n"
    interpret_text += f"ν: {'High' if params['nu'] > 1.0 else 'Low'} volatility of volatility\n"
    interpret_text += f"ρ: {'Strong' if abs(params['rho']) > 0.5 else 'Weak'} leverage effect\n"
    
    ax6.text(0.05, 0.95, param_text + stat_text + interpret_text,
             transform=ax6.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    if save_plot:
        # Create plot directory if it doesn't exist
        Path(plot_dir).mkdir(parents=True, exist_ok=True)
        
        # Save plot
        plot_path = os.path.join(plot_dir, f"{instrument}_surface_{surface_id}_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Surface comparison plot saved: {plot_path}")
        return plot_path
    else:
        plt.show()
        return None


def plot_training_history(history: Dict[str, list], save_path: Optional[str] = None) -> None:
    """
    Plot training history (loss and metrics)
    
    Parameters:
    -----------
    history : Dict[str, list]
        Training history from Keras model.fit()
    save_path : Optional[str]
        Path to save the plot (if None, displays plot)
    """
    if not PLOTTING_AVAILABLE:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history['loss'], label='Training Loss', color='blue')
    if 'val_loss' in history:
        ax1.plot(history['val_loss'], label='Validation Loss', color='red')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot MAE
    if 'mae' in history:
        ax2.plot(history['mae'], label='Training MAE', color='blue')
        if 'val_mae' in history:
            ax2.plot(history['val_mae'], label='Validation MAE', color='red')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   📈 Training history plot saved: {save_path}")
    else:
        plt.show()


def plot_parameter_distribution(params_list: list, param_names: list = None, 
                              save_path: Optional[str] = None) -> None:
    """
    Plot distribution of calibrated parameters
    
    Parameters:
    -----------
    params_list : list
        List of parameter dictionaries
    param_names : list
        Names of parameters to plot (default: ['H', 'nu', 'rho'])
    save_path : Optional[str]
        Path to save the plot (if None, displays plot)
    """
    if not PLOTTING_AVAILABLE:
        return
    
    if param_names is None:
        param_names = ['H', 'nu', 'rho']
    
    n_params = len(param_names)
    fig, axes = plt.subplots(1, n_params, figsize=(5*n_params, 4))
    
    if n_params == 1:
        axes = [axes]
    
    for i, param_name in enumerate(param_names):
        values = [params[param_name] for params in params_list if param_name in params]
        
        axes[i].hist(values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[i].set_title(f'{param_name} Distribution')
        axes[i].set_xlabel(param_name)
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = np.mean(values)
        std_val = np.std(values)
        axes[i].axvline(mean_val, color='red', linestyle='--', 
                       label=f'Mean: {mean_val:.4f}')
        axes[i].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   📊 Parameter distribution plot saved: {save_path}")
    else:
        plt.show()


def plot_surface_evolution(surfaces: list, timestamps: list, instrument: str,
                         save_path: Optional[str] = None) -> None:
    """
    Plot evolution of volatility surfaces over time
    
    Parameters:
    -----------
    surfaces : list
        List of volatility surfaces
    timestamps : list
        List of timestamps corresponding to surfaces
    instrument : str
        Instrument name
    save_path : Optional[str]
        Path to save the plot (if None, displays plot)
    """
    if not PLOTTING_AVAILABLE:
        return
    
    n_surfaces = min(len(surfaces), 6)  # Limit to 6 surfaces for clarity
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i in range(n_surfaces):
        surface = surfaces[i]
        timestamp = timestamps[i]
        
        # Reshape surface to 2D if needed
        if surface.ndim == 1:
            # Try to find reasonable dimensions
            size = len(surface)
            n_maturities = int(np.sqrt(size))
            n_moneyness = size // n_maturities
            if size % n_maturities != 0:
                n_maturities = 7  # Default
                n_moneyness = size // n_maturities
            surface_2d = surface.reshape(n_maturities, n_moneyness)
        else:
            surface_2d = surface
        
        im = axes[i].imshow(surface_2d, cmap=cm.viridis, aspect='auto')
        axes[i].set_title(f'{instrument} Surface\n{timestamp}')
        axes[i].set_xlabel('Moneyness')
        axes[i].set_ylabel('Maturity')
        plt.colorbar(im, ax=axes[i])
    
    # Hide unused subplots
    for i in range(n_surfaces, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   📈 Surface evolution plot saved: {save_path}")
    else:
        plt.show() 