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