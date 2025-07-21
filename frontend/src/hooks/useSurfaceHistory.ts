import { useState, useEffect, useCallback } from 'react';
import { HistoricalSurfaceData, SurfaceHistoryState, SurfaceHistoryHookParams, Asset } from '../types/surface';

const API_CONFIG = {
  baseUrl: 'http://127.0.0.1:8000/api/v1',
  endpoints: {
    volSurfaceHistory: '/vol_surface/history',
    assets: '/assets',
  },
  defaultHeaders: {
    'Content-Type': 'application/json',
  },
};

// Hook for fetching available assets
export function useAssets() {
  const [assets, setAssets] = useState<Asset[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchAssets = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      const url = `${API_CONFIG.baseUrl}${API_CONFIG.endpoints.assets}`;
      const response = await fetch(url, {
        method: 'GET',
        headers: API_CONFIG.defaultHeaders,
        mode: 'cors',
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setAssets(data);
    } catch (error) {
      console.error('Error fetching assets:', error);
      setError(error instanceof Error ? error.message : 'Unknown error fetching assets');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchAssets();
  }, [fetchAssets]);

  return { assets, isLoading, error, refetch: fetchAssets };
}

export function useSurfaceHistory({
  limit = 50,
  autoRefresh = false,
  refreshInterval = 30000, // 30 seconds default
  asset_id = null, // Add asset filter
}: SurfaceHistoryHookParams & { 
  asset_id?: number | null;
} = {}) {
  const [state, setState] = useState<SurfaceHistoryState>({
    isLoading: true,
    error: null,
    data: null,
  });

  const fetchHistory = useCallback(async () => {
    try {
      setState(prev => ({ ...prev, isLoading: true, error: null }));
      
      // Build URL with parameters
      const params = new URLSearchParams({ limit: limit.toString() });
      if (asset_id !== null) {
        params.append('asset_id', asset_id.toString());
      }
      
      const url = `${API_CONFIG.baseUrl}${API_CONFIG.endpoints.volSurfaceHistory}?${params.toString()}`;
      console.log('Fetching from URL:', url); // Debug log
      
      const response = await fetch(url, {
        method: 'GET',
        headers: API_CONFIG.defaultHeaders,
        mode: 'cors', // Explicitly set CORS mode
      });

      console.log('Response status:', response.status); // Debug log
      console.log('Response ok:', response.ok); // Debug log

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
      }

      const rawData = await response.json();
      console.log('Raw backend response:', rawData); // Debug log
      
      // Transform the data to match frontend interface
      const data: HistoricalSurfaceData[] = rawData.map((surface: any) => ({
        timestamp: surface.timestamp,
        moneyness: surface.moneyness || [],
        daysToExpiry: surface.days_to_expiry || [], // Map snake_case to camelCase
        impliedVols: surface.implied_vols || [], // Map snake_case to camelCase
        interpolationMethod: surface.method || 'unknown',
        asset_id: surface.asset_id,
        snapshot_id: surface.snapshot_id,
        strikes: surface.strikes || [],
        maturities: surface.maturities || [],
        option_type: surface.option_type || [],
        method: surface.method,
        spot_price: surface.spot_price,
      }));
      
      console.log('Transformed data:', data); // Debug log
      console.log('First surface sample:', data[0]); // Debug log
      
      setState({
        isLoading: false,
        error: null,
        data,
      });
    } catch (error) {
      console.error('Error fetching surface history:', error);
      setState({
        isLoading: false,
        error: error instanceof Error ? error.message : 'Unknown error fetching history',
        data: null,
      });
    }
  }, [limit, asset_id]);

  useEffect(() => {
    fetchHistory();
  }, [fetchHistory]);

  useEffect(() => {
    if (!autoRefresh) return;

    const intervalId = setInterval(fetchHistory, refreshInterval);
    return () => clearInterval(intervalId);
  }, [autoRefresh, refreshInterval, fetchHistory]);

  return {
    ...state,
    refetch: fetchHistory,
  };
} 