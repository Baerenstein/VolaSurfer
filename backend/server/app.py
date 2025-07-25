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

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": app.version,
    }

