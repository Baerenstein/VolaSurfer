export interface SurfaceData {
  timestamp: string;
  moneyness: number[];
  daysToExpiry: number[];
  impliedVols: number[][];
  interpolationMethod: string;
}

export interface SurfaceState {
  isLoading: boolean;
  error: string | null;
  data: SurfaceData | null;
}

// Updated types for history functionality
export interface HistoricalSurfaceData {
  timestamp: string;
  moneyness: number[];
  daysToExpiry: number[];
  // Allow both flat array (from VolSurface) and 2D array (from interpolated data)
  impliedVols: number[] | number[][];
  interpolationMethod?: string;
  asset_id?: number;
  snapshot_id?: string;
  strikes?: number[];
  maturities?: string[];
  option_type?: string[];
  method?: string;
}

export interface SurfaceHistoryState {
  isLoading: boolean;
  error: string | null;
  data: HistoricalSurfaceData[] | null;
}

export interface SurfaceHistoryHookParams {
  limit?: number;
  autoRefresh?: boolean;
  refreshInterval?: number;
}