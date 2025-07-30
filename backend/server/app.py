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
    ),
    min_moneyness: float = Query(settings.MIN_MONEYNESS, description="Minimum moneyness to include"),
    max_moneyness: float = Query(settings.MAX_MONEYNESS, description="Maximum moneyness to include"),
    min_maturity: int = Query(settings.MIN_MATURITY, description="Minimum days to expiry to include"),
    max_maturity: int = Query(settings.MAX_MATURITY, description="Maximum days to expiry to include")
):
    """
    Retrieve the latest volatility surface with optional interpolation and moneyness/maturity filtering.
    """
    surface_data = store.get_latest_vol_surface()
    if surface_data is None:
        raise HTTPException(status_code=404, detail="Latest volatility surface not found")
    try:
        interpolated_data = interpolate_surface(surface_data, method)
        moneyness = interpolated_data["moneyness"]
        days_to_expiry = interpolated_data["days_to_expiry"]
        mask = [
            (min_moneyness <= m <= max_moneyness) and (min_maturity <= dte <= max_maturity)
            for m, dte in zip(moneyness, days_to_expiry)
        ]
        filtered = {
            k: [v[i] for i in range(len(moneyness)) if mask[i]] if isinstance(v, list) and len(v) == len(moneyness) else v
            for k, v in interpolated_data.items()
        }
        return JSONResponse(content=filtered)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error interpolating surface: {str(e)}"
        )

@app.websocket("/api/v1/ws/latest-vol-surface")
async def websocket_latest_vol_surface(websocket: WebSocket):
    await websocket.accept()
    params = websocket.query_params
    method = params.get("method")
    min_moneyness = float(params.get("min_moneyness", settings.MIN_MONEYNESS))
    max_moneyness = float(params.get("max_moneyness", settings.MAX_MONEYNESS))
    min_maturity = int(params.get("min_maturity", settings.MIN_MATURITY))
    max_maturity = int(params.get("max_maturity", settings.MAX_MATURITY))
    if method is None or method.upper() not in SurfaceType.__members__:
        await websocket.send_json({"error": "Invalid interpolation method"})
        await websocket.close()
        return
    method = SurfaceType.__members__[method.upper()]
    client_connected = True
    try:
        while True:
            surface_data = store.get_latest_vol_surface()
            interpolated_data = interpolate_surface(surface_data, method)
            moneyness = interpolated_data["moneyness"]
            days_to_expiry = interpolated_data["days_to_expiry"]
            mask = [
                (min_moneyness <= m <= max_moneyness) and (min_maturity <= dte <= max_maturity)
                for m, dte in zip(moneyness, days_to_expiry)
            ]
            filtered = {
                k: [v[i] for i in range(len(moneyness)) if mask[i]] if isinstance(v, list) and len(v) == len(moneyness) else v
                for k, v in interpolated_data.items()
            }
            if surface_data is not None:
                await websocket.send_json(filtered)
            await asyncio.sleep(5)
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
    asset_id: Optional[int] = Query(None, description="Asset ID to filter by"),
    min_moneyness: float = Query(settings.MIN_MONEYNESS, description="Minimum moneyness to include"),
    max_moneyness: float = Query(settings.MAX_MONEYNESS, description="Maximum moneyness to include"),
    min_maturity: int = Query(settings.MIN_MATURITY, description="Minimum days to expiry to include"),
    max_maturity: int = Query(settings.MAX_MATURITY, description="Maximum days to expiry to include")
) -> List[dict]:
    """
    Retrieve the last N volatility surfaces ordered by timestamp descending, with optional moneyness/maturity filtering.
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
        def convert_decimal_to_float(data):
            if isinstance(data, list):
                return [convert_decimal_to_float(item) for item in data]
            elif isinstance(data, dict):
                return {k: convert_decimal_to_float(v) for k, v in data.items()}
            elif isinstance(data, Decimal):
                return float(data)
            else:
                return data
        response_data = []
        for surface in surfaces:
            d = surface.to_dict()
            moneyness = d.get("moneyness", [])
            days_to_expiry = d.get("days_to_expiry", [])
            mask = [
                (min_moneyness <= m <= max_moneyness) and (min_maturity <= dte <= max_maturity)
                for m, dte in zip(moneyness, days_to_expiry)
            ]
            for k in ["moneyness", "strikes", "maturities", "days_to_expiry", "implied_vols", "option_type"]:
                if k in d and isinstance(d[k], list) and len(d[k]) == len(moneyness):
                    d[k] = [d[k][i] for i in range(len(moneyness)) if mask[i]]
            response_data.append(convert_decimal_to_float(d))
        return JSONResponse(content=response_data)
    except HTTPException as http_exc:
        logging.error(f"HTTP error: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logging.error(f"Error retrieving surfaces: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving surfaces: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": app.version,
    }


# WebGL Wallpaper API Endpoints
@app.get("/api/surface_snapshot")
async def get_surface_snapshot():
    """
    Returns historical surface data in x, y, z format for WebGL rendering.
    """
    try:
        surface_data = store.get_latest_vol_surface()
        if surface_data is None:
            raise HTTPException(status_code=404, detail="No surface data available")
        
        # Convert to x, y, z format for WebGL
        interpolated_data = interpolate_surface(surface_data, SurfaceType.LINEAR)
        
        # Create x, y, z coordinates
        x_coords = []
        y_coords = []
        z_coords = []
        
        # Handle 2D array structure from interpolation
        if isinstance(interpolated_data["implied_vols"], list) and len(interpolated_data["implied_vols"]) > 0:
            if isinstance(interpolated_data["implied_vols"][0], list):
                # 2D array structure
                for j, dte in enumerate(interpolated_data["days_to_expiry"]):
                    for i, moneyness in enumerate(interpolated_data["moneyness"]):
                        if i < len(interpolated_data["implied_vols"][j]):
                            vol = interpolated_data["implied_vols"][j][i]
                            if vol is not None:
                                x_coords.append(float(moneyness))
                                y_coords.append(float(dte))
                                z_coords.append(float(vol))
            else:
                # 1D array structure
                for i, moneyness in enumerate(interpolated_data["moneyness"]):
                    for j, dte in enumerate(interpolated_data["days_to_expiry"]):
                        vol_index = j * len(interpolated_data["moneyness"]) + i
                        if vol_index < len(interpolated_data["implied_vols"]):
                            vol = interpolated_data["implied_vols"][vol_index]
                            if vol is not None:
                                x_coords.append(float(moneyness))
                                y_coords.append(float(dte))
                                z_coords.append(float(vol))
        
        return {
            "x": x_coords,
            "y": y_coords,
            "z": z_coords,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logging.error(f"Error getting surface snapshot: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting surface snapshot: {str(e)}")


@app.get("/api/trendline")
async def get_trendline(limit: int = Query(100, description="Number of historical points")):
    """
    Returns historical price/volatility trend line data.
    """
    try:
        surfaces = store.get_last_n_surfaces(limit)
        if not surfaces:
            raise HTTPException(status_code=404, detail="No historical data available")
        
        trend_data = []
        for i, surface in enumerate(surfaces):
            if surface.implied_vols and len(surface.implied_vols) > 0:
                avg_vol = sum(float(vol) for vol in surface.implied_vols) / len(surface.implied_vols)
                trend_data.append({
                    "time": i,  # Index as time proxy
                    "price": float(avg_vol),  # Average volatility as price proxy
                    "timestamp": surface.timestamp.isoformat() if surface.timestamp else None
                })
        
        return {
            "trendline": trend_data,
            "count": len(trend_data)
        }
    except Exception as e:
        logging.error(f"Error getting trendline: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting trendline: {str(e)}")


@app.get("/api/price_history")
async def get_price_history(
    limit: int = Query(100, description="Number of price points to retrieve"),
    asset_id: Optional[int] = Query(None, description="Asset ID to filter by")
):
    """
    Returns historical price data for the underlying asset.
    """
    try:
        # Get price history from the database
        price_data = store.get_price_history(limit, asset_id)
        
        if not price_data:
            raise HTTPException(status_code=404, detail="No price data available")
        
        # Convert to list of dictionaries
        price_history = []
        for row in price_data:
            price_value = float(row["price"])
            logging.info(f"Price data point: {row['timestamp']} - {price_value} (raw: {row['price']})")
            price_history.append({
                "timestamp": row["timestamp"].isoformat() if hasattr(row["timestamp"], 'isoformat') else str(row["timestamp"]),
                "price": price_value,
                "asset_id": row.get("asset_id"),
                "symbol": row.get("symbol")
            })
        
        return {
            "prices": price_history,
            "count": len(price_history)
        }
    except Exception as e:
        logging.error(f"Error getting price history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting price history: {str(e)}")


@app.get("/api/stats")
async def get_stats():
    """
    Returns latest spread, volume, and timestamp statistics.
    """
    try:
        surface_data = store.get_latest_vol_surface()
        if surface_data is None:
            raise HTTPException(status_code=404, detail="No surface data available")
        
        interpolated_data = interpolate_surface(surface_data, SurfaceType.LINEAR)
        
        if not interpolated_data["implied_vols"]:
            raise HTTPException(status_code=404, detail="No volatility data available")
        
        # Handle the case where implied_vols is a 2D array from interpolation
        if isinstance(interpolated_data["implied_vols"], list) and len(interpolated_data["implied_vols"]) > 0:
            if isinstance(interpolated_data["implied_vols"][0], list):
                # 2D array - flatten it
                vols = []
                for row in interpolated_data["implied_vols"]:
                    for vol in row:
                        if vol is not None:
                            vols.append(float(vol))
            else:
                # 1D array
                vols = [float(v) for v in interpolated_data["implied_vols"] if v is not None]
        else:
            vols = []
        
        if not vols:
            raise HTTPException(status_code=404, detail="No valid volatility data available")
        
        max_vol = max(vols)
        min_vol = min(vols)
        spread = (max_vol - min_vol) * 100  # Convert to percentage
        
        volume = len(vols)  # Number of data points as volume proxy
        
        return {
            "spread": round(spread, 2),
            "volume": volume,
            "timestamp": datetime.now().isoformat(),
            "max_vol": round(max_vol * 100, 2),
            "min_vol": round(min_vol * 100, 2),
            "avg_vol": round(sum(vols) / len(vols) * 100, 2)
        }
    except Exception as e:
        logging.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")


@app.websocket("/api/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time updates.
    """
    await websocket.accept()
    client_connected = True
    
    try:
        while client_connected:
            # Send surface snapshot
            surface_data = await get_surface_snapshot()
            await websocket.send_json({
                "type": "surface_snapshot",
                "data": surface_data
            })
            
            # Send stats
            stats_data = await get_stats()
            await websocket.send_json({
                "type": "stats",
                "data": stats_data
            })
            
            # Wait 5 seconds before next update
            await asyncio.sleep(5)
            
    except WebSocketDisconnect:
        print("WebSocket client disconnected")
        client_connected = False
    except Exception as e:
        print(f"Error in WebSocket stream: {str(e)}")
        client_connected = False
    finally:
        if client_connected:
            await websocket.close()

