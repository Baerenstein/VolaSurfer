import { useState, useCallback } from 'react';
import { SurfaceData } from '../types/surface';

const API_CONFIG = {
  baseUrl: 'http://127.0.0.1:8000/api/v1',
  endpoints: {
    volSurface: '/latest-vol-surface',
  },
  defaultHeaders: {
    'Content-Type': 'application/json',
  },
};

export function useSurfaceData() {
  const [state, setState] = useState<SurfaceState>({
    isLoading: false,
    error: null,
    data: null,
  });

  const fetchSurfaceData = useCallback(async () => {
    console.log('Fetching surface data');
    setState(prev => ({ ...prev, isLoading: true, error: null }));

    try {
      const requestUrl = `${API_CONFIG.baseUrl}${API_CONFIG.endpoints.volSurface}?method=nearest`;
      console.log('Request URL:', requestUrl);

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);

      const response = await fetch(requestUrl, {
        headers: API_CONFIG.defaultHeaders,
        signal: controller.signal,
        mode: 'cors',
      });

      clearTimeout(timeoutId);
      
      if (!response.ok) {
        const errorBody = await response.text();
        console.error('Server response:', {
          status: response.status,
          statusText: response.statusText,
          body: errorBody,
        });
        throw new Error(`Server error: ${response.status} ${response.statusText}\n${errorBody}`);
      }

      const rawData = await response.json();
      console.log('Raw data received:', rawData);
      
      // Validate the required fields
      if (!rawData.moneyness || !rawData.days_to_expiry || !rawData.implied_vols || !rawData.timestamp) {
        throw new Error('Missing required fields in server response');
      }

      const surfaceData: SurfaceData = {
        timestamp: rawData.timestamp,
        moneyness: rawData.moneyness,
        daysToExpiry: rawData.days_to_expiry,
        impliedVols: rawData.implied_vols,
        interpolationMethod: rawData.interpolation_method || 'nearest'
      };

      setState({
        isLoading: false,
        error: null,
        data: surfaceData,
      });
    } catch (error) {
      console.error('Error fetching surface data:', {
        error,
        message: error instanceof Error ? error.message : 'Unknown error',
        stack: error instanceof Error ? error.stack : undefined,
      });

      let errorMessage = 'An unknown error occurred';
      
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          errorMessage = 'Request timed out';
        } else if (error.message.includes('Failed to fetch')) {
          errorMessage = 'Could not connect to server. Please check if the backend is running.';
        } else {
          errorMessage = error.message;
        }
      }

      setState({
        isLoading: false,
        error: errorMessage,
        data: null,
      });
    }
  }, []);

  return {
    ...state,
    refetch: fetchSurfaceData,
  };
}