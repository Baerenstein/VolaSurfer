#!/usr/bin/env python3
"""
Demonstration script for the enhanced surface calibration pipeline.
This script shows the calibration engine working with synthetic data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.core.CalibrationEngine import CalibrationEngine, CalibrationMetrics

class DemoStore:
    """Demo store for testing calibration functionality"""
    
    def get_options_by_snapshot(self, snapshot_id, asset_id):
        """Generate synthetic options data"""
        print(f"📊 Generating synthetic options data for {asset_id}")
        
        # Generate realistic option chain
        np.random.seed(42)  # For reproducible results
        
        # Market parameters
        S0 = 100  # Current price
        r = 0.05  # Risk-free rate
        
        # Generate strikes and expiries
        strikes = np.linspace(80, 120, 15)  # 15 strikes from 80 to 120
        expiries = [7, 14, 30, 60, 90, 120, 180]  # 7 expiry dates
        
        options_data = []
        
        for dte in expiries:
            for strike in strikes:
                # Calculate theoretical Black-Scholes parameters
                moneyness = strike / S0
                T = dte / 365.0
                
                # Generate realistic implied volatility with smile
                atm_vol = 0.20  # Base ATM vol
                
                # Volatility smile: higher vol for OTM options
                vol_smile = atm_vol + 0.1 * (moneyness - 1.0)**2
                
                # Add some noise
                vol_noise = np.random.normal(0, 0.02)
                implied_vol = max(0.05, vol_smile + vol_noise)
                
                # Calculate Greeks (simplified)
                d1 = (np.log(S0/strike) + (r + 0.5*implied_vol**2)*T) / (implied_vol*np.sqrt(T))
                d2 = d1 - implied_vol*np.sqrt(T)
                
                from scipy.stats import norm
                delta = norm.cdf(d1)
                gamma = norm.pdf(d1) / (S0 * implied_vol * np.sqrt(T))
                vega = S0 * norm.pdf(d1) * np.sqrt(T) / 100
                theta = -(S0 * norm.pdf(d1) * implied_vol) / (2 * np.sqrt(T)) - r * strike * np.exp(-r*T) * norm.cdf(d2)
                theta = theta / 365  # Daily theta
                
                options_data.append({
                    'timestamp': datetime.now().isoformat(),
                    'strike': strike,
                    'moneyness': moneyness,
                    'expiry_date': (datetime.now() + timedelta(days=dte)).date().isoformat(),
                    'days_to_expiry': dte,
                    'implied_vol': implied_vol,
                    'option_type': 'call',
                    'delta': delta,
                    'gamma': gamma,
                    'vega': vega,
                    'theta': theta,
                    'snapshot_id': snapshot_id,
                    'asset_id': asset_id
                })
        
        return pd.DataFrame(options_data)
    
    def store_calibration_results(self, data):
        """Mock store calibration results"""
        print(f"💾 Storing calibration results: {data.get('method', 'unknown')} method")
        return True
    
    def get_calibration_history(self, asset_id, lookback_days=30):
        """Mock calibration history"""
        return []
    
    def get_latest_snapshot_id(self, asset_id):
        """Mock latest snapshot ID"""
        return f"snapshot_{asset_id}_{datetime.now().strftime('%Y%m%d')}"
    
    def get_surface_quality_metrics(self, snapshot_id, asset_id):
        """Mock surface quality metrics"""
        return {
            'data_coverage': 0.85,
            'smoothness_score': 0.92,
            'outlier_count': 2,
            'total_points': 105,
            'unique_expiries': 7,
            'unique_strikes': 15,
            'arbitrage_violations': 0,
            'calendar_violations': 0,
            'bid_ask_spread_avg': 0.02,
            'last_updated': datetime.now().isoformat()
        }

def demonstrate_calibration():
    """Demonstrate the calibration pipeline"""
    
    print("🎯 Surface Calibration Pipeline Demonstration")
    print("=" * 60)
    
    # Initialize components
    store = DemoStore()
    engine = CalibrationEngine(store)
    
    # Test parameters
    asset_id = "SPY"
    snapshot_id = store.get_latest_snapshot_id(asset_id)
    
    print(f"\n📈 Testing with asset: {asset_id}")
    print(f"📸 Snapshot ID: {snapshot_id}")
    
    # Test all calibration methods
    methods = ['svi', 'rbf', 'spline', 'sabr', 'ssvi']
    results = {}
    
    for method in methods:
        print(f"\n🔧 Testing {method.upper()} calibration...")
        
        try:
            result = engine.calibrate_surface_from_db(
                asset_id=asset_id,
                snapshot_id=snapshot_id,
                method=method,
                min_moneyness=0.8,
                max_moneyness=1.2,
                min_dte=7,
                max_dte=180
            )
            
            if result.success:
                metrics = result.metrics
                print(f"   ✅ {method.upper()} completed successfully!")
                print(f"   📊 RMSE: {metrics.rmse:.4f}")
                print(f"   📊 MAE: {metrics.mae:.4f}")
                print(f"   📊 R²: {metrics.r_squared:.4f}")
                print(f"   📊 Quality: {metrics.fit_quality}")
                print(f"   ⏱️  Time: {metrics.calibration_time:.2f}s")
                print(f"   📈 Data points: {metrics.num_points}")
                
                results[method] = result
            else:
                print(f"   ❌ {method.upper()} failed: {result.message}")
                
        except Exception as e:
            print(f"   ❌ {method.upper()} error: {str(e)}")
    
    # Performance analysis demo
    print(f"\n📊 Performance Analysis for {asset_id}")
    print("-" * 40)
    
    try:
        performance = engine.analyze_calibration_performance(asset_id, lookback_days=30)
        print(f"   📈 Analysis period: 30 days")
        print(f"   🎯 Asset: {performance['asset_id']}")
        print(f"   ⏰ Last updated: {performance['last_updated']}")
        
        if 'error' in performance:
            print(f"   ℹ️  Note: {performance['error']}")
    except Exception as e:
        print(f"   ❌ Performance analysis error: {str(e)}")
    
    # Surface quality metrics demo
    print(f"\n🔍 Surface Quality Metrics for {asset_id}")
    print("-" * 40)
    
    try:
        quality = store.get_surface_quality_metrics(snapshot_id, asset_id)
        print(f"   📊 Data coverage: {quality['data_coverage']:.1%}")
        print(f"   📈 Smoothness score: {quality['smoothness_score']:.2f}")
        print(f"   🎯 Total points: {quality['total_points']}")
        print(f"   📐 Unique strikes: {quality['unique_strikes']}")
        print(f"   📅 Unique expiries: {quality['unique_expiries']}")
        print(f"   ⚠️  Outliers: {quality['outlier_count']}")
        print(f"   🚫 Arbitrage violations: {quality['arbitrage_violations']}")
        print(f"   📊 Avg bid-ask spread: {quality['bid_ask_spread_avg']:.1%}")
    except Exception as e:
        print(f"   ❌ Quality metrics error: {str(e)}")
    
    # Summary comparison
    if results:
        print(f"\n📈 Calibration Methods Comparison")
        print("-" * 50)
        print(f"{'Method':<8} {'RMSE':<8} {'MAE':<8} {'R²':<8} {'Quality':<12} {'Time':<6}")
        print("-" * 50)
        
        for method, result in results.items():
            if result.success:
                m = result.metrics
                print(f"{method.upper():<8} {m.rmse:<8.4f} {m.mae:<8.4f} {m.r_squared:<8.4f} {m.fit_quality:<12} {m.calibration_time:<6.2f}")
    
    print(f"\n🎉 Demonstration completed!")
    print(f"💡 The calibration pipeline is ready for production use.")
    
    # Create a simple visualization if matplotlib is available
    try:
        create_demo_visualization(results)
    except Exception as e:
        print(f"📊 Visualization skipped: {str(e)}")

def create_demo_visualization(results):
    """Create a simple comparison visualization"""
    if not results:
        return
    
    methods = []
    rmse_values = []
    time_values = []
    
    for method, result in results.items():
        if result.success:
            methods.append(method.upper())
            rmse_values.append(result.metrics.rmse)
            time_values.append(result.metrics.calibration_time)
    
    if not methods:
        return
    
    # Create comparison plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # RMSE comparison
    bars1 = ax1.bar(methods, rmse_values, color='skyblue', alpha=0.7)
    ax1.set_title('Calibration Accuracy (RMSE)')
    ax1.set_ylabel('RMSE')
    ax1.set_xlabel('Method')
    
    # Add value labels on bars
    for bar, value in zip(bars1, rmse_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.4f}', ha='center', va='bottom')
    
    # Time comparison
    bars2 = ax2.bar(methods, time_values, color='lightcoral', alpha=0.7)
    ax2.set_title('Calibration Speed')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_xlabel('Method')
    
    # Add value labels on bars
    for bar, value in zip(bars2, time_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.2f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('calibration_comparison.png', dpi=150, bbox_inches='tight')
    print(f"📊 Visualization saved as 'calibration_comparison.png'")

if __name__ == "__main__":
    print("🚀 Starting Surface Calibration Pipeline Demo...")
    demonstrate_calibration()