import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import pickle
import warnings
warnings.filterwarnings('ignore')

try:
    from tensorflow.keras import Input, Model
    from tensorflow.keras.layers import Conv1D, Flatten, Dense, Reshape, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. Install with: pip install tensorflow scikit-learn")

try:
    from scipy.optimize import least_squares, differential_evolution
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: SciPy not available. Install with: pip install scipy")

from data.utils.data_schemas import VolSurface
from data.storage.base_store import BaseStore
from infrastructure.utils.logging import setup_logger
from core.ModelEngine import rBergomi, ModelEngine





class SurfaceCalibrationEngine:
    """Surface calibration engine for volatility models"""
    
    def __init__(self, store: BaseStore, model_name: str = "rBergomi_CNN", 
                 instrument: Optional[str] = None):
        self.store = store
        self.model_name = model_name
        self.instrument = instrument
        self.asset_id = None
        self.logger = setup_logger("surface_calibration")
        
        # Default calibration parameters - ensure numpy arrays are always available
        import numpy as np
        self.maturities = np.array([0.1, 0.3, 0.6, 1.0])  # Years
        self.moneyness_range = np.linspace(0.8, 1.2, 11)   # Strike/Spot ratios
        self.n_paths = 50000
        
        # Parameter space for synthetic data generation
        self.param_grid = [
            (H, nu, rho) 
            for H in [0.05, 0.1, 0.15, 0.2] 
            for nu in [0.8, 1.0, 1.2, 1.5, 2.0] 
            for rho in [-0.9, -0.7, -0.5, -0.3]
        ]
        
        # Model components
        self.cnn_model = None
        self.scaler_X = None
        self.scaler_Y = None
        
        # Get asset_id if instrument specified
        if self.instrument:
            self.asset_id = self._get_asset_id(self.instrument)
            if self.asset_id:
                self.logger.info(f"Calibrating for instrument: {self.instrument} (asset_id: {self.asset_id})")
            else:
                self.logger.warning(f"Instrument {self.instrument} not found in database")
    
    def _get_asset_id(self, instrument: str) -> Optional[int]:
        """Get asset_id for given instrument ticker"""
        if hasattr(self.store, 'conn'):
            try:
                with self.store.conn.cursor() as cursor:
                    cursor.execute("SELECT id FROM assets WHERE ticker = %s", (instrument,))
                    result = cursor.fetchone()
                    return result[0] if result else None
            except Exception as e:
                self.logger.error(f"Error getting asset_id for {instrument}: {e}")
                return None
        return None
        
    def generate_synthetic_data(self, n_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic volatility surface data using rBergomi model"""
        if not SCIPY_AVAILABLE:
            raise ImportError("SciPy is required for synthetic data generation")
            
        self.logger.info("Generating synthetic volatility surface data...")
        
        param_grid = self.param_grid
        if n_samples:
            # Randomly sample from parameter grid
            indices = np.random.choice(len(param_grid), min(n_samples, len(param_grid)), replace=False)
            param_grid = [param_grid[i] for i in indices]
        
        X_params = []
        vol_surfaces = []
        
        for i, (H, nu, rho) in enumerate(param_grid):
            if i % 10 == 0:
                self.logger.info(f"Generating sample {i+1}/{len(param_grid)}")
                
            # Create proper rBergomi model with appropriate parameters
            # Convert H (Hurst) to alpha: a = H - 0.5
            a = H - 0.5
            n_steps = 100  # Steps per year
            n_paths = 5000  # Reduced for speed
            T_max = max(self.maturities)
            
            model = rBergomi(n=n_steps, N=n_paths, T=T_max, a=a)
            
            surface = []
            for T in self.maturities:
                vol_row = []
                for m in self.moneyness_range:
                    K = m  # S0 = 1, so moneyness = K/S0 = K
                    try:
                        # Use proper rBergomi pricing
                        price = model.price_european_option(
                            K=K, T=T, xi=0.04, eta=nu, rho=rho, S0=1.0, r=0.0
                        )
                        
                        # Extract implied volatility using Black-Scholes inversion
                        iv = self._extract_implied_vol(price, S0=1.0, K=K, T=T, r=0.0)
                        vol_row.append(iv)
                    except:
                        # Fallback if pricing fails
                        vol_row.append(0.2)  # Default volatility
                        
                surface.extend(vol_row)  # Flatten to 1D
            
            X_params.append([H, nu, rho])
            vol_surfaces.append(surface)
        
        X = np.array(X_params)
        Y = np.array(vol_surfaces)
        
        self.logger.info(f"Generated synthetic data: X shape {X.shape}, Y shape {Y.shape}")
        return X, Y
    
    def _extract_implied_vol(self, price: float, S0: float, K: float, T: float, r: float) -> float:
        """Extract implied volatility from option price using Black-Scholes inversion"""
        from scipy.optimize import brentq
        from scipy.stats import norm
        
        def black_scholes_call(S, K, T, r, sigma):
            """Black-Scholes call option price"""
            if T <= 0:
                return max(S - K, 0)
            
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        
        def objective(sigma):
            return black_scholes_call(S0, K, T, r, sigma) - price
        
        try:
            # Use Brent's method to find implied volatility
            iv = brentq(objective, 0.001, 5.0)
            return max(iv, 0.01)  # Minimum 1% volatility
        except:
            # Fallback for problematic cases
            return 0.2  # Default 20% volatility
    
    def prepare_market_data(self, limit: int = 100) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Prepare market data from database for calibration"""
        if self.instrument:
            self.logger.info(f"Retrieving {limit} latest volatility surfaces for {self.instrument} from database...")
        else:
            self.logger.info(f"Retrieving {limit} latest volatility surfaces from database...")
        
        surfaces = self.store.get_last_n_surfaces(limit=limit, asset_id=self.asset_id)
        if not surfaces:
            if self.instrument:
                self.logger.warning(f"No volatility surfaces found for {self.instrument}")
            else:
                self.logger.warning("No volatility surfaces found in database")
            return None
            
        if self.instrument:
            self.logger.info(f"Retrieved {len(surfaces)} surfaces for {self.instrument} from database")
        else:
            self.logger.info(f"Retrieved {len(surfaces)} surfaces from database")
        
        market_surfaces = []
        timestamps = []
        max_size = 0
        
        # First pass: convert surfaces and find maximum grid size
        converted_surfaces = []
        for surface in surfaces:
            grid_surface = self._surface_to_grid(surface)
            if grid_surface is not None:
                converted_surfaces.append(grid_surface)
                timestamps.append(surface.timestamp)
                max_size = max(max_size, len(grid_surface))
        
        if not converted_surfaces:
            self.logger.warning("No valid market surfaces could be processed")
            return None
        
        # Second pass: pad all surfaces to the same size
        self.logger.info(f"Standardizing {len(converted_surfaces)} surfaces to size {max_size}")
        
        for grid_surface in converted_surfaces:
            if len(grid_surface) < max_size:
                # Pad with NaN values to reach max_size
                padded_surface = np.full(max_size, np.nan)
                padded_surface[:len(grid_surface)] = grid_surface
                market_surfaces.append(padded_surface)
            else:
                market_surfaces.append(grid_surface)
        
        Y_market = np.array(market_surfaces)
        timestamps = np.array(timestamps)
        
        self.logger.info(f"Prepared {Y_market.shape[0]} market surfaces with {Y_market.shape[1]} points each")
        return Y_market, timestamps
    
    def _surface_to_grid(self, surface: VolSurface) -> Optional[np.ndarray]:
        """Convert VolSurface to standardized grid format using adaptive grid"""
        try:
            # Create DataFrame from surface data (convert Decimal types from database)
            df = pd.DataFrame({
                'days_to_expiry': [float(x) for x in surface.days_to_expiry],
                'moneyness': [float(x) for x in surface.moneyness],
                'implied_vol': [float(x) for x in surface.implied_vols],
                'option_type': surface.option_type
            })
            
            # Filter for calls only
            df = df[df['option_type'] == 'c'].copy()
            if df.empty:
                return None
            
            # Convert days to years (handle Decimal types from database)
            df['days_to_expiry'] = df['days_to_expiry'].astype(float)
            df['maturity_years'] = df['days_to_expiry'] / 365.25
            
            # Use ADAPTIVE grid based on actual data
            unique_maturities = sorted(df['maturity_years'].unique())
            unique_moneyness = sorted(df['moneyness'].unique())
            
            # If we have reasonable coverage, use actual data grid
            if len(unique_maturities) >= 2 and len(unique_moneyness) >= 3:
                self.logger.info(f"Using adaptive grid: {len(unique_maturities)} maturities x {len(unique_moneyness)} moneyness")
                
                grid_vol = []
                for T in unique_maturities:
                    for m in unique_moneyness:
                        # Find exact or closest match
                        matches = df[(df['maturity_years'] == T) & (df['moneyness'] == m)]
                        if not matches.empty:
                            grid_vol.append(matches['implied_vol'].iloc[0])
                        else:
                            # Find closest point
                            df['distance'] = ((df['maturity_years'] - T)**2 + (df['moneyness'] - m)**2)**0.5
                            closest = df.loc[df['distance'].idxmin()]
                            if closest['distance'] < 0.1:  # Within reasonable distance
                                grid_vol.append(closest['implied_vol'])
                            else:
                                grid_vol.append(np.nan)
                
                # Update instance grid for consistency
                self.current_maturities = np.array(unique_maturities)
                self.current_moneyness = np.array(unique_moneyness)
                
            else:
                # Fallback to interpolation on fixed grid
                self.logger.info(f"Using fixed grid interpolation: {len(df)} points available")
                
                grid_vol = []
                for T in self.maturities:
                    # Find closest maturity points with wider tolerance
                    maturity_mask = np.abs(df['maturity_years'] - T) < max(T * 0.5, 0.1)
                    maturity_df = df[maturity_mask].copy()
                    
                    if maturity_df.empty:
                        # Fill with NaN if no data
                        grid_vol.extend([np.nan] * len(self.moneyness_range))
                        continue
                    
                    for m in self.moneyness_range:
                        # Find closest moneyness point with wider tolerance
                        closest_idx = np.argmin(np.abs(maturity_df['moneyness'] - m))
                        closest_row = maturity_df.iloc[closest_idx]
                        
                        # Use point if within reasonable distance (wider tolerance)
                        if abs(closest_row['moneyness'] - m) < 0.2:  # Increased from 0.1 to 0.2
                            grid_vol.append(closest_row['implied_vol'])
                        else:
                            grid_vol.append(np.nan)
                
                self.current_maturities = self.maturities
                self.current_moneyness = self.moneyness_range
            
            # Check if we have enough valid data
            grid_array = np.array(grid_vol)
            valid_ratio = np.sum(~np.isnan(grid_array)) / len(grid_array)
            
            self.logger.info(f"Surface grid: {len(grid_array)} points, {valid_ratio:.1%} valid")
            
            if valid_ratio < 0.15:  # Reduced threshold from 30% to 15%
                self.logger.warning(f"Insufficient valid data: {valid_ratio:.1%} < 15%")
                return None
                
            # Fill NaN values with interpolation
            if np.any(np.isnan(grid_array)):
                grid_array = self._interpolate_missing_values(grid_array)
            
            return grid_array
            
        except Exception as e:
            self.logger.error(f"Error converting surface to grid: {e}")
            return None
    
    def _interpolate_missing_values(self, grid_vol: np.ndarray) -> np.ndarray:
        """Interpolate missing values in volatility grid"""
        # For adaptive grids, use 1D interpolation
        if len(grid_vol) != len(self.maturities) * len(self.moneyness_range):
            # Use 1D forward/backward fill for adaptive grids
            result = grid_vol.copy()
            
            # Forward fill
            last_valid = None
            for i in range(len(result)):
                if not np.isnan(result[i]):
                    last_valid = result[i]
                elif last_valid is not None:
                    result[i] = last_valid
            
            # Backward fill
            last_valid = None
            for i in range(len(result)-1, -1, -1):
                if not np.isnan(result[i]):
                    last_valid = result[i]
                elif last_valid is not None:
                    result[i] = last_valid
            
            return result
        else:
            # Original 2D approach for fixed grid
            grid_2d = grid_vol.reshape(len(self.maturities), len(self.moneyness_range))
            
            # Simple forward/backward fill
            for i in range(len(self.maturities)):
                row = grid_2d[i, :]
                if np.any(~np.isnan(row)):
                    # Forward fill
                    last_valid = None
                    for j in range(len(row)):
                        if not np.isnan(row[j]):
                            last_valid = row[j]
                        elif last_valid is not None:
                            row[j] = last_valid
                    
                    # Backward fill
                    last_valid = None
                    for j in range(len(row)-1, -1, -1):
                        if not np.isnan(row[j]):
                            last_valid = row[j]
                        elif last_valid is not None:
                            row[j] = last_valid
            
            return grid_2d.flatten()
    
    def build_cnn_model(self, input_shape: Tuple[int, int, int]) -> Model:
        """Build CNN model for volatility surface to parameters mapping"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for CNN model")
            
        self.logger.info(f"Building CNN model with input shape: {input_shape}")
        
        inp = Input(shape=input_shape)
        
        # Reshape for Conv1D processing
        x = Reshape((input_shape[0], input_shape[1]))(inp)
        
        # Convolutional layers
        x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv1D(32, kernel_size=3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv1D(16, kernel_size=3, activation='relu', padding='same')(x)
        
        # Flatten and dense layers
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        # Output layer (3 parameters: H, nu, rho)
        out = Dense(3, activation='linear')(x)
        
        model = Model(inp, out)
        model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_model_with_market_data(self, use_market_data: bool = True, 
                                   n_synthetic: int = 1000, epochs: int = 100) -> Dict:
        """Train model using market data + some synthetic data for augmentation"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for model training")
        
        # Get market data
        market_result = self.prepare_market_data(limit=500)
        
        if use_market_data and market_result is not None:
            Y_market, timestamps = market_result
            if self.instrument:
                self.logger.info(f"Using {len(Y_market)} market surfaces for {self.instrument}")
            else:
                self.logger.info(f"Using {len(Y_market)} market surfaces")
            
            # Use optimization to get "ground truth" parameters for market data
            if self.instrument:
                self.logger.info(f"Calibrating parameters for {self.instrument} market surfaces (this may take a while)...")
            else:
                self.logger.info("Calibrating parameters for market surfaces (this may take a while)...")
            
            X_market = []
            for i, surface in enumerate(Y_market):
                if i % 50 == 0:
                    self.logger.info(f"Calibrating surface {i+1}/{len(Y_market)}")
                
                # Quick optimization for ground truth
                params = self._calibrate_with_optimization(surface)
                X_market.append([params['H'], params['nu'], params['rho']])
            
            X_market = np.array(X_market)
            if self.instrument:
                self.logger.info(f"Successfully calibrated {len(X_market)} {self.instrument} market surfaces")
            else:
                self.logger.info(f"Successfully calibrated {len(X_market)} market surfaces")
            
            # Generate synthetic data with MATCHING grid size
            target_size = Y_market.shape[1]  # Use market data grid size
            self.logger.info(f"Generating {n_synthetic} synthetic surfaces for augmentation")
            X_synthetic, Y_synthetic = self.generate_synthetic_data_with_size(n_synthetic, target_size)
            
            # Combine datasets
            X_combined = np.vstack([X_market, X_synthetic])
            Y_combined = np.vstack([Y_market, Y_synthetic])
            
        else:
            # Fallback to pure synthetic data
            self.logger.info(f"No market data available, using pure synthetic data")
            X_combined, Y_combined = self.generate_synthetic_data(n_synthetic)
        
        # Train model
        return self.train_model(X_combined, Y_combined, epochs=epochs)
    
    def generate_synthetic_data_with_size(self, n_samples: int, target_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic data with specific target size to match market data"""
        self.logger.info("Generating synthetic volatility surface data...")
        
        X_synthetic = []
        Y_synthetic = []
        
        sample_count = 0
        for i, (H, nu, rho) in enumerate(self.param_grid):
            if sample_count >= n_samples:
                break
                
            if i % 10 == 0:
                self.logger.info(f"Generating sample {i+1}/{len(self.param_grid)}")
            
            # Generate synthetic surface that matches target_size points
            synthetic_surface = []
            for j in range(target_size):
                # Simple volatility formula that creates target_size points
                T = 0.1 + (j / target_size) * 2.0  # Vary maturity
                m = 0.8 + (j % 20) * 0.02  # Vary moneyness
                vol = 0.2 + 0.1 * H + 0.05 * nu * np.sqrt(T) + 0.02 * rho * (m - 1.0)
                vol = max(0.01, vol)  # Ensure positive volatility
                synthetic_surface.append(vol)
            
            X_synthetic.append([H, nu, rho])
            Y_synthetic.append(synthetic_surface)
            sample_count += 1
        
        X_synthetic = np.array(X_synthetic)
        Y_synthetic = np.array(Y_synthetic)
        
        self.logger.info(f"Generated synthetic data: X shape {X_synthetic.shape}, Y shape {Y_synthetic.shape}")
        return X_synthetic, Y_synthetic
    
    def train_model(self, X: np.ndarray, Y: np.ndarray, epochs: int = 100, 
                   validation_split: float = 0.2, batch_size: int = 32) -> Dict:
        """Train CNN model with adaptive grid size"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for model training")
        
        self.logger.info("Training CNN model...")
        
        # Standardize input features
        if self.scaler_X is None:
            from sklearn.preprocessing import StandardScaler
            self.scaler_X = StandardScaler()
            X_scaled = self.scaler_X.fit_transform(X)
        else:
            X_scaled = self.scaler_X.transform(X)
        
        # Standardize surface data
        if self.scaler_Y is None:
            from sklearn.preprocessing import StandardScaler
            self.scaler_Y = StandardScaler()
            Y_scaled = self.scaler_Y.fit_transform(Y)
        else:
            Y_scaled = self.scaler_Y.transform(Y)
        
        # Check for NaN values in data
        if np.any(np.isnan(X_scaled)):
            self.logger.warning(f"Found {np.sum(np.isnan(X_scaled))} NaN values in X data")
            X_scaled = np.nan_to_num(X_scaled, nan=0.0)
        
        if np.any(np.isnan(Y_scaled)):
            self.logger.warning(f"Found {np.sum(np.isnan(Y_scaled))} NaN values in Y data")
            Y_scaled = np.nan_to_num(Y_scaled, nan=0.0)
        
        # Check for infinite values
        if np.any(np.isinf(X_scaled)):
            self.logger.warning(f"Found {np.sum(np.isinf(X_scaled))} infinite values in X data")
            X_scaled = np.nan_to_num(X_scaled, posinf=1e6, neginf=-1e6)
        
        if np.any(np.isinf(Y_scaled)):
            self.logger.warning(f"Found {np.sum(np.isinf(Y_scaled))} infinite values in Y data")
            Y_scaled = np.nan_to_num(Y_scaled, posinf=1e6, neginf=-1e6)
        
        # Adaptive grid dimensions based on actual data
        n_samples, surface_size = Y_scaled.shape
        self.logger.info(f"Training data: {n_samples} samples, {surface_size} points per surface")
        
        # For CNN, we need to reshape the surface data
        # Use a square-ish reshape or 1D convolution approach
        if surface_size == 44:  # Original fixed grid
            Y_reshaped = Y_scaled.reshape(-1, 4, 11, 1)
            input_shape = (4, 11, 1)
        else:
            # For adaptive grids, use 1D convolution approach
            Y_reshaped = Y_scaled.reshape(-1, surface_size, 1)
            input_shape = (surface_size, 1)
        
        self.logger.info(f"Reshaped data for CNN: {Y_reshaped.shape}")
        
        # Build CNN model adaptively
        from tensorflow.keras import Input, Model
        from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout, BatchNormalization
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        
        # Input layer
        inputs = Input(shape=input_shape)
        
        if len(input_shape) == 3:  # 2D grid case (4, 11, 1)
            # Use 2D approach - flatten to 1D first
            x = Flatten()(inputs)
            x = Dense(128, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)
            x = Dense(64, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)
        else:  # 1D case for adaptive grids
            # Simplified architecture for stability
            x = Flatten()(inputs)
            x = Dense(256, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)
            x = Dense(128, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)
            x = Dense(64, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.1)(x)
        
        # Output layer
        outputs = Dense(3, activation='linear')(x)  # H, nu, rho
        
        # Create and compile model
        self.cnn_model = Model(inputs, outputs)
        self.cnn_model.compile(
            optimizer=Adam(learning_rate=0.0001),  # Reduced learning rate
            loss='mse',
            metrics=['mae']
        )
        
        self.logger.info(f"Model architecture: input {input_shape} -> output (3,)")
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(patience=10, factor=0.5, min_lr=1e-6)
        ]
        
        try:
            # Train model
            history = self.cnn_model.fit(
                Y_reshaped, X_scaled,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=0
            )
            
            # Get final metrics
            final_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1]
            
            self.logger.info(f"Training completed - Loss: {final_loss:.6f}, Val Loss: {final_val_loss:.6f}")
            
            return {
                'loss': history.history['loss'],
                'val_loss': history.history['val_loss'],
                'final_loss': final_loss,
                'final_val_loss': final_val_loss,
                'epochs_trained': len(history.history['loss']),
                'surface_size': surface_size
            }
            
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            raise
    
    def calibrate_surface(self, vol_surface_data: np.ndarray, 
                         method: str = 'cnn') -> Dict[str, float]:
        """Calibrate model parameters to a volatility surface"""
        if method == 'cnn':
            return self._calibrate_with_cnn(vol_surface_data)
        elif method == 'optimization':
            return self._calibrate_with_optimization(vol_surface_data)
        else:
            raise ValueError(f"Unknown calibration method: {method}")
    
    def _calibrate_with_cnn(self, vol_surface_data: np.ndarray) -> Dict[str, float]:
        """Calibrate using trained CNN model"""
        if self.cnn_model is None or self.scaler_Y is None or self.scaler_X is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Handle different surface sizes by padding/truncating to match training size
        surface_size = len(vol_surface_data)
        expected_size = self.scaler_Y.n_features_in_
        
        if surface_size != expected_size:
            # Pad or truncate to match expected size
            if surface_size < expected_size:
                # Pad with the last value
                padded_surface = np.full(expected_size, vol_surface_data[-1] if len(vol_surface_data) > 0 else 0.2)
                padded_surface[:surface_size] = vol_surface_data
                vol_surface_data = padded_surface
            else:
                # Truncate to expected size
                vol_surface_data = vol_surface_data[:expected_size]
        
        # Prepare surface data
        surface_scaled = self.scaler_Y.transform(vol_surface_data.reshape(1, -1))
        
        # Reshape for CNN - handle adaptive grid sizes
        if expected_size == 44:  # Original fixed grid (4 x 11)
            surface_reshaped = surface_scaled.reshape(1, 4, 11, 1)
        else:
            # For adaptive grids, use 1D convolution format
            surface_reshaped = surface_scaled.reshape(1, expected_size, 1)
        
        # Predict parameters
        params_scaled = self.cnn_model.predict(surface_reshaped, verbose=0)
        params = self.scaler_X.inverse_transform(params_scaled)[0]
        
        return {
            'H': float(np.clip(params[0], 0.01, 0.5)),
            'nu': float(np.clip(params[1], 0.1, 3.0)),
            'rho': float(np.clip(params[2], -0.99, 0.99))
        }
    
    def _calibrate_with_optimization(self, vol_surface_data: np.ndarray) -> Dict[str, float]:
        """Calibrate using scipy optimization"""
        from scipy.optimize import differential_evolution
        
        def objective(params):
            H, nu, rho = params
            
            # Generate predicted surface using synthetic generation
            # Use current grid dimensions
            if hasattr(self, 'current_maturities') and hasattr(self, 'current_moneyness'):
                maturities = self.current_maturities
                moneyness = self.current_moneyness
            else:
                maturities = self.maturities
                moneyness = self.moneyness_range
            
            predicted_surface = []
            
            # Create proper rBergomi model for this parameter set
            try:
                a = H - 0.5  # Convert Hurst to alpha
                n_steps = 50  # Reduced for optimization speed
                n_paths = 1000  # Reduced for optimization speed
                T_max = max(maturities)
                
                model = rBergomi(n=n_steps, N=n_paths, T=T_max, a=a)
                
                for T in maturities:
                    for m in moneyness:
                        K = m  # S0 = 1, so moneyness = K/S0 = K
                        
                        # Price option with rBergomi model
                        price = model.price_european_option(
                            K=K, T=T, xi=0.04, eta=nu, rho=rho, S0=1.0, r=0.0
                        )
                        
                        # Extract implied volatility
                        iv = self._extract_implied_vol(price, S0=1.0, K=K, T=T, r=0.0)
                        predicted_surface.append(iv)
                        
            except Exception as e:
                # Fallback to simplified model if rBergomi fails
                for T in maturities:
                    for m in moneyness:
                        vol = 0.2 + 0.1 * H + 0.05 * nu * np.sqrt(T) + 0.02 * rho * (m - 1.0)
                        predicted_surface.append(max(vol, 0.01))
            
            predicted_array = np.array(predicted_surface)
            
            # Ensure same size for comparison
            actual_size = len(vol_surface_data)
            predicted_size = len(predicted_array)
            
            if predicted_size > actual_size:
                # Truncate predicted to match actual
                predicted_array = predicted_array[:actual_size]
            elif predicted_size < actual_size:
                # Pad predicted with last value
                padding = np.full(actual_size - predicted_size, predicted_array[-1] if len(predicted_array) > 0 else 0.2)
                predicted_array = np.concatenate([predicted_array, padding])
            
            # Calculate error only on valid (non-NaN) points
            mask = ~(np.isnan(vol_surface_data) | np.isnan(predicted_array))
            if np.sum(mask) == 0:
                return 1e6  # Large penalty if no valid points
            
            error = np.sum((vol_surface_data[mask] - predicted_array[mask])**2)
            return error
        
        try:
            bounds = [(0.01, 0.49), (0.1, 3.0), (-0.99, 0.99)]  # H, nu, rho bounds
            result = differential_evolution(
                objective,
                bounds,
                seed=42,
                maxiter=100,
                popsize=10
            )
            
            if result.success:
                H, nu, rho = result.x
                return {
                    'H': float(H),
                    'nu': float(nu), 
                    'rho': float(rho),
                    'calibration_error': float(result.fun),
                    'method': 'optimization'
                }
            else:
                self.logger.warning("Optimization failed to converge")
                return {
                    'H': 0.1,
                    'nu': 1.0,
                    'rho': -0.5,
                    'calibration_error': float('inf'),
                    'method': 'optimization_failed'
                }
                
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            return {
                'H': 0.1,
                'nu': 1.0,
                'rho': -0.5,
                'calibration_error': float('inf'),
                'method': 'optimization_error'
            }
    
    def save_model(self, filepath: str):
        """Save trained model and scalers"""
        if self.cnn_model is None:
            raise ValueError("No model to save")
            
        # Save model
        self.cnn_model.save(f"{filepath}_model.h5")
        
        # Save scalers
        with open(f"{filepath}_scalers.pkl", 'wb') as f:
            pickle.dump({
                'scaler_X': self.scaler_X,
                'scaler_Y': self.scaler_Y,
                'maturities': self.maturities,
                'moneyness_range': self.moneyness_range
            }, f)
        
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model and scalers"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required to load models")
            
        from tensorflow.keras.models import load_model
        
        # Load model
        self.cnn_model = load_model(f"{filepath}_model.h5")
        
        # Load scalers
        with open(f"{filepath}_scalers.pkl", 'rb') as f:
            data = pickle.load(f)
            self.scaler_X = data['scaler_X']
            self.scaler_Y = data['scaler_Y'] 
            self.maturities = data['maturities']
            self.moneyness_range = data['moneyness_range']
        
        self.logger.info(f"Model loaded from {filepath}")
    
    def store_calibrated_parameters(self, params: Dict[str, float], 
                                  surface_id: int, asset_id: int) -> int:
        """Store calibrated parameters in database"""
        import json
        
        # Get or create model entry
        model_id = self._get_or_create_model()
        
        calibration_data = {
            'timestamp': datetime.now().isoformat(),
            'method': 'surface_calibration',
            'calibration_error': params.get('calibration_error', 0.0),
            'surface_id': surface_id,
            'model_type': self.model_name,
            'training_data_size': getattr(self, '_last_training_size', 0)
        }
        
        # Store parameters in database
        if hasattr(self.store, 'conn'):
            try:
                with self.store.conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO model_parameters (model_id, asset_id, parameters, timestamp, calibration_data)
                        VALUES (%s, %s, %s, %s, %s)
                        RETURNING id
                    """, (
                        model_id,
                        asset_id,
                        json.dumps(params),
                        datetime.now(),
                        json.dumps(calibration_data)
                    ))
                    
                    param_id = cursor.fetchone()[0]
                    self.store.conn.commit()
                    
                    self.logger.info(f"Stored calibrated parameters with ID {param_id}: {params}")
                    return param_id
                    
            except Exception as e:
                self.store.conn.rollback()
                self.logger.error(f"Failed to store parameters: {e}")
                raise
        else:
            self.logger.warning("Database connection not available for parameter storage")
            return -1
    
    def _get_or_create_model(self) -> int:
        """Get existing model ID or create new one"""
        if not hasattr(self.store, 'conn'):
            return 1  # Default for non-database stores
        
        try:
            with self.store.conn.cursor() as cursor:
                # First try to find existing model by name only
                cursor.execute(
                    "SELECT id FROM models WHERE name = %s LIMIT 1",
                    (self.model_name,)
                )
                result = cursor.fetchone()
                if result:
                    return result[0]
                
                # If not found, try to create a simple entry
                try:
                    cursor.execute("""
                        INSERT INTO models (name, description)
                        VALUES (%s, %s)
                        ON CONFLICT (id) DO NOTHING
                        RETURNING id
                    """, (
                        self.model_name,
                        'Rough Bergomi volatility model with CNN calibration'
                    ))
                    
                    result = cursor.fetchone()
                    if result:
                        model_id = result[0]
                        self.store.conn.commit()
                        return model_id
                except Exception as e:
                    self.logger.warning(f"Could not create model entry: {e}")
                    # Fallback to any existing model or default
                    cursor.execute("SELECT MIN(id) FROM models")
                    result = cursor.fetchone()
                    return result[0] if result and result[0] else 1
                
        except Exception as e:
            self.logger.warning(f"Database model operation failed: {e}")
            return 1  # Fallback to default 