import { useState, useEffect } from 'react';
import { SurfaceData } from '../types/surface';

const API_CONFIG = {
  baseUrl: 'http://127.0.0.1:8000/api/v1',
  wsBaseUrl: 'ws://127.0.0.1:8000/api/v1',
  endpoints: {
    volSurface: '/latest-vol-surface',
    volSurfaceWs: '/ws/latest-vol-surface',
  },
  defaultHeaders: {
    'Content-Type': 'application/json',
  },
};

export function useSurfaceData() {
  const [state, setState] = useState({
    isLoading: true,
    error: null,
    data: null,
  });

  useEffect(() => {
    const wsUrl = `${API_CONFIG.wsBaseUrl}${API_CONFIG.endpoints.volSurfaceWs}?method=nearest`;
    console.log('Connecting to WebSocket:', wsUrl);

    const socket = new WebSocket(wsUrl);

    socket.onopen = () => {
      console.log('WebSocket connection established');
      setState(prev => ({ ...prev, isLoading: true }));
    };

    socket.onmessage = (event) => {
      try {
        const rawData = JSON.parse(event.data);
        console.log('WebSocket message received:', rawData);

        if (!rawData.moneyness || !rawData.days_to_expiry || !rawData.implied_vols || !rawData.timestamp) {
          throw new Error('Missing required fields in server response');
        }

        const surfaceData: SurfaceData = {
          timestamp: rawData.timestamp,
          moneyness: rawData.moneyness,
          daysToExpiry: rawData.days_to_expiry,
          impliedVols: rawData.implied_vols,
          interpolationMethod: rawData.interpolation_method || 'nearest',
        };

        setState({
          isLoading: false,
          error: null,
          data: surfaceData,
        });
      } catch (error) {
        console.error('Error processing WebSocket message:', error);
        setState({
          isLoading: false,
          error: error instanceof Error ? error.message : 'Unknown error',
          data: null,
        });
      }
    };

    socket.onerror = (error) => {
      console.error('WebSocket error:', error);
      setState({
        isLoading: false,
        error: 'WebSocket connection error',
        data: null,
      });
    };

    socket.onclose = (event) => {
      console.warn('WebSocket connection closed:', event);
      setState((prev) => ({ ...prev, error: 'WebSocket connection closed' }));
    };

    return () => {
      console.log('Closing WebSocket connection');
      socket.close();
    };
  }, []);

  return state;
}
