from datetime import datetime
import sys
import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from infrastructure.settings import Settings
from data.storage import StorageFactory
from data.utils.data_schemas import OptionContract
from data.utils.surface_helper import interpolate_surface, SurfaceType
import asyncio
import logging
from decimal import Decimal
from core.CalibrationEngine import CalibrationEngine, SurfaceCalibrationResult


# Add the backend directory to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Initialize FastAPI app
app = FastAPI(
    title="VolaSurfer API",
    description="API for options volatility surface analysis",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load settings and create storage instance
settings = Settings()
store = StorageFactory.create_storage(settings)

# Initialize calibration engine
calibration_engine = CalibrationEngine(store)


# Error handler for common exceptions
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.get("/api/v1/options-chain/{symbol}")
async def get_options_chain(symbol: str) -> List[dict]:
    """Retrieve the options chain for a given symbol."""
    options_df = store.get_options_chain(symbol)
    if options_df is None or options_df.empty:
        raise HTTPException(status_code=404, detail="Options chain not found")

    # Convert DataFrame to list of OptionContract objects
    options = []
    for _, row in options_df.iterrows():
        option = OptionContract(
            timestamp=row.get("timestamp", datetime.now()),
            base_currency=row["base_currency"],
            symbol=symbol,
            expiry_date=row["expiry_date"],
            days_to_expiry=row["days_to_expiry"],
            strike=row["strike"],
            moneyness=row["moneyness"],
            option_type=row["option_type"],
            last_price=row["last_price"],
            implied_vol=row["implied_vol"],
            delta=row.get("delta"),
            gamma=row.get("gamma"),
            vega=row.get("vega"),
            theta=row.get("theta"),
            open_interest=row.get("open_interest"),
            snapshot_id=row.get("snapshot_id"),
        )
        options.append(option.to_dict())

    return JSONResponse(content=options)

@app.get("/api/v1/latest-vol-surface")
async def get_latest_vol_surface(
    method: SurfaceType = Query(
        SurfaceType.NEAREST,
        description="Interpolation method to use: raw, cubic, or nearest"
    )
):
    """
    Retrieve the latest volatility surface with optional interpolation.
    
    Args:
        method: Interpolation method (raw, cubic, or nearest)
    
    Returns:
        JSON response with surface data
    """
    # Get raw data from store
    surface_data = store.get_latest_vol_surface()
    if surface_data is None:
        raise HTTPException(status_code=404, detail="Latest volatility surface not found")
    
    # Apply interpolation if requested
    try:
        interpolated_data = interpolate_surface(surface_data, method)
        return JSONResponse(content=interpolated_data)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error interpolating surface: {str(e)}"
        )



@app.websocket("/api/v1/ws/latest-vol-surface")
async def websocket_latest_vol_surface(websocket: WebSocket):
    await websocket.accept()

    params = websocket.query_params
    method = params.get("method")  # Removed default value to make it truly dependent on the query

    if method.upper() not in SurfaceType.__members__:
        await websocket.send_json({"error": "Invalid interpolation method"})
        await websocket.close()
        return

    method = SurfaceType.__members__[method.upper()]

    client_connected = True
    try:
        while True:
            # Here you would typically fetch the latest surface data
            surface_data = store.get_latest_vol_surface()
            interpolated_data = interpolate_surface(surface_data, method)
            if surface_data is not None:
                await websocket.send_json(interpolated_data)
            await asyncio.sleep(5)  # Adjust the frequency of updates as needed
    except WebSocketDisconnect:
        print("Client disconnected")
        client_connected = False
    except Exception as e:
        print(f"Error in WebSocket connection: {str(e)}")
    finally:
        if client_connected:
            await websocket.close()

@app.get("/api/v1/assets")
async def get_available_assets() -> List[dict]:
    """
    Retrieve all available assets in the database.
    """
    try:
        assets = store.get_available_assets()
        return JSONResponse(content=assets)
    except Exception as e:
        logging.error(f"Error retrieving assets: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving assets: {str(e)}")

@app.get("/api/v1/vol_surface/history")
async def get_vol_surface_history(
    limit: int = Query(100, description="Number of surfaces to retrieve"),
    min_dte: Optional[int] = Query(None, description="Minimum days to expiry"),
    max_dte: Optional[int] = Query(None, description="Maximum days to expiry"),
    asset_id: Optional[int] = Query(None, description="Asset ID to filter by")
) -> List[dict]:
    """
    Retrieve the last N volatility surfaces ordered by timestamp descending.
    """
    try:
        if limit <= 0:
            logging.error("Limit must be a positive integer")
            raise HTTPException(status_code=422, detail="Limit must be a positive integer")

        logging.info(f"Fetching last {limit} volatility surfaces with DTE filter: {min_dte}-{max_dte}, asset_id: {asset_id}")
        surfaces = store.get_last_n_surfaces(limit, min_dte=min_dte, max_dte=max_dte, asset_id=asset_id)
        
        if len(surfaces) == 0:
            logging.warning("No volatility surfaces found")
            return JSONResponse(status_code=404, content={"detail": "No volatility surfaces found"})
            
        # Convert all Decimal objects to float for JSON serialization
        def convert_decimal_to_float(data):
            if isinstance(data, list):
                return [convert_decimal_to_float(item) for item in data]
            elif isinstance(data, dict):
                return {k: convert_decimal_to_float(v) for k, v in data.items()}
            elif isinstance(data, Decimal):
                return float(data)
            else:
                return data

        response_data = [convert_decimal_to_float(surface.to_dict()) for surface in surfaces]
        return JSONResponse(content=response_data)
    except HTTPException as http_exc:
        logging.error(f"HTTP error: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logging.error(f"Error retrieving surfaces: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving surfaces: {str(e)}")

@app.post("/api/v1/calibrate-surface")
async def calibrate_surface(
    asset_id: str,
    snapshot_id: Optional[str] = None,
    method: str = Query("svi", description="Calibration method: svi, ssvi, sabr, spline, rbf"),
    min_moneyness: float = Query(0.7, description="Minimum moneyness filter"),
    max_moneyness: float = Query(1.3, description="Maximum moneyness filter"),
    min_dte: int = Query(7, description="Minimum days to expiry"),
    max_dte: int = Query(365, description="Maximum days to expiry")
):
    """
    Calibrate volatility surface using specified method and parameters.
    
    Returns calibrated surface with performance metrics and fitted parameters.
    """
    try:
        logging.info(f"Starting surface calibration for asset {asset_id} using {method}")
        
        # Validate method
        valid_methods = ['svi', 'ssvi', 'sabr', 'spline', 'rbf']
        if method not in valid_methods:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid method '{method}'. Must be one of: {valid_methods}"
            )
        
        # Perform calibration
        result = calibration_engine.calibrate_surface_from_db(
            asset_id=asset_id,
            snapshot_id=snapshot_id,
            method=method,
            min_moneyness=min_moneyness,
            max_moneyness=max_moneyness,
            min_dte=min_dte,
            max_dte=max_dte
        )
        
        if not result.success:
            raise HTTPException(status_code=400, detail=result.message)
        
        # Prepare response
        response_data = {
            "success": result.success,
            "message": result.message,
            "surface": result.surface.to_dict() if result.surface else None,
            "metrics": {
                "rmse": result.metrics.rmse,
                "mae": result.metrics.mae,
                "max_error": result.metrics.max_error,
                "r_squared": result.metrics.r_squared,
                "calibration_time": result.metrics.calibration_time,
                "num_points": result.metrics.num_points,
                "fit_quality": result.metrics.fit_quality,
                "timestamp": result.metrics.timestamp.isoformat()
            } if result.metrics else None,
            "model_parameters": result.model_parameters,
            "fitted_surface": {
                "data": result.fitted_surface.tolist() if result.fitted_surface is not None else None,
                "moneyness_grid": result.interpolation_grid[0].tolist() if result.interpolation_grid[0] is not None else None,
                "dte_grid": result.interpolation_grid[1].tolist() if result.interpolation_grid[1] is not None else None
            }
        }
        
        return JSONResponse(content=response_data)
        
    except HTTPException as http_exc:
        logging.error(f"HTTP error in calibration: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logging.error(f"Error in surface calibration: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Calibration failed: {str(e)}")

@app.get("/api/v1/calibration-performance/{asset_id}")
async def get_calibration_performance(
    asset_id: str,
    lookback_days: int = Query(30, description="Number of days to analyze")
):
    """
    Analyze calibration performance over time for a specific asset.
    
    Returns performance trends, stability metrics, and recommendations.
    """
    try:
        logging.info(f"Analyzing calibration performance for asset {asset_id}")
        
        analysis = calibration_engine.analyze_calibration_performance(
            asset_id=asset_id,
            lookback_days=lookback_days
        )
        
        return JSONResponse(content=analysis)
        
    except Exception as e:
        logging.error(f"Error analyzing calibration performance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Performance analysis failed: {str(e)}")

@app.get("/api/v1/calibration-methods")
async def get_calibration_methods():
    """
    Get available calibration methods with descriptions.
    """
    methods = {
        "svi": {
            "name": "SVI (Stochastic Volatility Inspired)",
            "description": "Parametric model with 5 parameters providing smooth volatility surfaces",
            "parameters": ["a (level)", "b (angle)", "rho (correlation)", "m (shift)", "sigma (smoothness)"],
            "best_for": "General purpose, well-behaved surfaces"
        },
        "ssvi": {
            "name": "Surface SVI",
            "description": "Extension of SVI that ensures arbitrage-free surfaces across time",
            "parameters": ["SVI parameters varying with time"],
            "best_for": "Multi-expiry surfaces with arbitrage-free constraints"
        },
        "sabr": {
            "name": "SABR (Stochastic Alpha Beta Rho)",
            "description": "Stochastic volatility model with closed-form approximation",
            "parameters": ["alpha (vol of vol)", "beta (CEV parameter)", "rho (correlation)", "nu (vol of vol)"],
            "best_for": "Interest rate derivatives, FX options"
        },
        "spline": {
            "name": "Bicubic Spline",
            "description": "Non-parametric interpolation using smooth splines",
            "parameters": ["Smoothing factor", "Spline degrees"],
            "best_for": "High-quality data with dense coverage"
        },
        "rbf": {
            "name": "Radial Basis Function",
            "description": "Non-parametric interpolation using radial basis functions",
            "parameters": ["Kernel type"],
            "best_for": "Irregular grids, sparse data"
        }
    }
    
    return JSONResponse(content=methods)

@app.get("/api/v1/surface-metrics/{asset_id}")
async def get_surface_metrics(
    asset_id: str,
    snapshot_id: Optional[str] = None
):
    """
    Calculate comprehensive surface metrics for quality assessment.
    """
    try:
        # Get latest snapshot if not provided
        if snapshot_id is None:
            snapshot_id = store.get_latest_snapshot_id(asset_id)
            
        if snapshot_id is None:
            raise HTTPException(status_code=404, detail=f"No data found for asset {asset_id}")
        
        # Get quality metrics from database
        quality_metrics = store.get_surface_quality_metrics(snapshot_id, asset_id)
        
        if not quality_metrics:
            raise HTTPException(status_code=404, detail=f"No surface data found for asset {asset_id}, snapshot {snapshot_id}")
        
        # Get options data for additional calculations
        options_df = store.get_options_by_snapshot(snapshot_id, asset_id)
        
        # Calculate risk metrics
        risk_metrics = {}
        if not options_df.empty:
            # ATM volatility (moneyness close to 1.0)
            atm_data = options_df[abs(options_df['moneyness'] - 1.0) < 0.05]
            atm_vol = atm_data['implied_vol'].mean() if not atm_data.empty else 0.0
            
            # 25-delta skew (approximate using moneyness)
            otm_put_data = options_df[(options_df['moneyness'] < 0.9) & (options_df['option_type'] == 'P')]
            otm_call_data = options_df[(options_df['moneyness'] > 1.1) & (options_df['option_type'] == 'C')]
            
            otm_put_vol = otm_put_data['implied_vol'].mean() if not otm_put_data.empty else atm_vol
            otm_call_vol = otm_call_data['implied_vol'].mean() if not otm_call_data.empty else atm_vol
            skew_25d = otm_put_vol - otm_call_vol
            
            # Term structure slope (simplified)
            short_term = options_df[options_df['days_to_expiry'] <= 30]
            long_term = options_df[options_df['days_to_expiry'] >= 90]
            
            short_vol = short_term['implied_vol'].mean() if not short_term.empty else atm_vol
            long_vol = long_term['implied_vol'].mean() if not long_term.empty else atm_vol
            term_structure_slope = (long_vol - short_vol) / 60  # per day
            
            # Convexity (simplified measure based on vol spread)
            vol_range = options_df['implied_vol'].max() - options_df['implied_vol'].min()
            convexity = vol_range / atm_vol if atm_vol > 0 else 0
            
            risk_metrics = {
                "atm_vol": round(atm_vol, 4),
                "skew_25d": round(skew_25d, 4),
                "convexity": round(convexity, 4),
                "term_structure_slope": round(term_structure_slope, 6)
            }
        
        # Generate recommendations
        recommendations = []
        if quality_metrics.get('data_coverage', 0) < 0.7:
            recommendations.append("Low data coverage - consider additional data sources")
        if quality_metrics.get('smoothness_score', 0) < 0.8:
            recommendations.append("Surface may be noisy - consider smoothing parameters")
        if quality_metrics.get('outlier_count', 0) > 5:
            recommendations.append(f"High outlier count ({quality_metrics['outlier_count']}) - review data quality")
        
        # Check for arbitrage conditions (simplified)
        arbitrage_violations = 0
        if not options_df.empty:
            # Check for negative volatilities (should not happen but worth checking)
            negative_vols = options_df[options_df['implied_vol'] <= 0]
            arbitrage_violations += len(negative_vols)
            
            # Check for extreme volatility values
            extreme_vols = options_df[options_df['implied_vol'] > 2.0]  # 200% vol
            arbitrage_violations += len(extreme_vols)
        
        if arbitrage_violations == 0:
            recommendations.append("No arbitrage violations detected")
        else:
            recommendations.append(f"Detected {arbitrage_violations} potential arbitrage violations")
        
        if not recommendations:
            recommendations.append("Surface quality appears good")
        
        # Combine all metrics
        metrics = {
            "asset_id": asset_id,
            "snapshot_id": snapshot_id,
            "timestamp": datetime.now().isoformat(),
            "quality_metrics": {
                "data_coverage": round(quality_metrics.get('data_coverage', 0), 3),
                "smoothness_score": round(quality_metrics.get('smoothness_score', 0), 3),
                "arbitrage_violations": arbitrage_violations,
                "outlier_count": quality_metrics.get('outlier_count', 0),
                "total_points": quality_metrics.get('total_points', 0),
                "unique_expiries": quality_metrics.get('unique_expiries', 0),
                "unique_strikes": quality_metrics.get('unique_strikes', 0)
            },
            "risk_metrics": risk_metrics,
            "recommendations": recommendations
        }
        
        return JSONResponse(content=metrics)
        
    except HTTPException as http_exc:
        logging.error(f"HTTP error in surface metrics: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logging.error(f"Error calculating surface metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Metrics calculation failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": app.version,
    }

