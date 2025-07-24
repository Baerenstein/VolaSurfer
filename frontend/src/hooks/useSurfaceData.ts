import { useState, useEffect, useRef } from 'react';
import { SurfaceData } from '../types/surface';

interface SurfaceDataState {
  isLoading: boolean;
  error: string | null;
  data: SurfaceData | null;
}

const useSurfaceData = (url: string) => {
  const [state, setState] = useState<SurfaceDataState>({
    isLoading: true,
    error: null,
    data: null,
  });
  
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      setState(prev => ({ ...prev, isLoading: true, error: null }));

      try {
        const response = await fetch(url);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (!data || typeof data !== 'object') {
          throw new Error('Invalid data format received');
        }

        // Basic validation for surface data
        const surfaceData: SurfaceData = {
          asset_id: data.asset_id || 'unknown',
          timestamp: data.timestamp || new Date().toISOString(),
          surface_type: data.surface_type || 'implied_volatility',
          moneyness: Array.isArray(data.moneyness) ? data.moneyness : [],
          daysToExpiry: Array.isArray(data.daysToExpiry) ? data.daysToExpiry : [],
          impliedVols: Array.isArray(data.impliedVols) ? data.impliedVols : [],
          strikes: Array.isArray(data.strikes) ? data.strikes : [],
          expiry_dates: Array.isArray(data.expiry_dates) ? data.expiry_dates : [],
          underlying_price: typeof data.underlying_price === 'number' ? data.underlying_price : 100,
          bid_ask_spreads: Array.isArray(data.bid_ask_spreads) ? data.bid_ask_spreads : [],
          delta: Array.isArray(data.delta) ? data.delta : [],
          gamma: Array.isArray(data.gamma) ? data.gamma : [],
          vega: Array.isArray(data.vega) ? data.vega : [],
          theta: Array.isArray(data.theta) ? data.theta : []
        };

        setState({
          isLoading: false,
          data: surfaceData,
          error: null,
        });
      } catch (error) {
        setState({
          isLoading: false,
          data: null,
          error: error instanceof Error ? error.message : 'Unknown error',
        });
      }
    };

    const connectWebSocket = () => {
      try {
        const wsUrl = url.replace(/^http/, 'ws') + '/ws';
        wsRef.current = new WebSocket(wsUrl);
        
        wsRef.current.onopen = () => {
          console.log('WebSocket connected');
        };
        
        wsRef.current.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            setState(prev => ({ ...prev, data: data }));
          } catch (error) {
            console.error('Error parsing WebSocket message:', error);
          }
        };
        
        wsRef.current.onerror = () => {
          setState(prev => ({ ...prev, error: 'WebSocket connection error' }));
        };
        
        wsRef.current.onclose = () => {
          setState(prev => ({ ...prev, error: 'WebSocket connection closed' }));
        };
      } catch (error) {
        console.error('Error connecting to WebSocket:', error);
      }
    };

    fetchData();
    // connectWebSocket(); // Uncomment if WebSocket is needed

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [url]);

  return state;
};

export default useSurfaceData;
