export interface CalibrationMetrics {
  rmse: number;
  mae: number;
  max_error: number;
  r_squared: number;
  calibration_time: number;
  num_points: number;
  fit_quality: string;
  timestamp: string;
}

export interface CalibrationResult {
  success: boolean;
  message: string;
  metrics: CalibrationMetrics | null;
  model_parameters: Record<string, number>;
  fitted_surface: {
    data: number[][];
    moneyness_grid: number[][];
    dte_grid: number[][];
  };
}

export interface PerformanceAnalysis {
  asset_id: string;
  analysis_period: number;
  metrics_history: CalibrationMetrics[];
  trend_analysis: {
    rmse_trend: 'improving' | 'stable' | 'deteriorating';
    calibration_time_trend: 'improving' | 'stable' | 'deteriorating';
    fit_quality_distribution: Record<string, number>;
  };
  last_updated: string;
}

export interface SurfaceQualityMetrics {
  data_coverage: number;
  smoothness_score: number;
  outlier_count: number;
  total_points: number;
  unique_expiries: number;
  unique_strikes: number;
  arbitrage_violations: number;
  calendar_violations: number;
  bid_ask_spread_avg: number;
  last_updated: string;
}

export interface CalibrationRequest {
  asset_id: string;
  snapshot_id?: string;
  method: 'svi' | 'ssvi' | 'sabr' | 'spline' | 'rbf';
  min_moneyness: number;
  max_moneyness: number;
  min_dte: number;
  max_dte: number;
}

class CalibrationApiService {
  private baseUrl: string;

  constructor() {
    this.baseUrl = process.env.REACT_APP_API_URL || 'http://localhost:8000';
  }

  async calibrateSurface(request: CalibrationRequest): Promise<CalibrationResult> {
    const queryParams = new URLSearchParams({
      method: request.method,
      min_moneyness: request.min_moneyness.toString(),
      max_moneyness: request.max_moneyness.toString(),
      min_dte: request.min_dte.toString(),
      max_dte: request.max_dte.toString(),
    });

    if (request.snapshot_id) {
      queryParams.append('snapshot_id', request.snapshot_id);
    }

    const response = await fetch(
      `${this.baseUrl}/api/v1/calibrate-surface?asset_id=${request.asset_id}&${queryParams}`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      }
    );

    if (!response.ok) {
      throw new Error(`Calibration failed: ${response.statusText}`);
    }

    return response.json();
  }

  async getPerformanceAnalysis(
    assetId: string, 
    analysisPeriod: number = 30
  ): Promise<PerformanceAnalysis> {
    const response = await fetch(
      `${this.baseUrl}/api/v1/calibration-performance/${assetId}?analysis_period=${analysisPeriod}`
    );

    if (!response.ok) {
      throw new Error(`Performance analysis failed: ${response.statusText}`);
    }

    return response.json();
  }

  async getSurfaceMetrics(
    assetId: string, 
    snapshotId?: string
  ): Promise<SurfaceQualityMetrics> {
    let url = `${this.baseUrl}/api/v1/surface-metrics/${assetId}`;
    if (snapshotId) {
      url += `?snapshot_id=${snapshotId}`;
    }

    const response = await fetch(url);

    if (!response.ok) {
      throw new Error(`Surface metrics failed: ${response.statusText}`);
    }

    return response.json();
  }

  async getAssets(): Promise<string[]> {
    const response = await fetch(`${this.baseUrl}/api/v1/assets`);
    
    if (!response.ok) {
      throw new Error(`Failed to fetch assets: ${response.statusText}`);
    }

    return response.json();
  }

  async getSnapshots(assetId: string): Promise<Array<{id: string, timestamp: string}>> {
    const response = await fetch(`${this.baseUrl}/api/v1/snapshots/${assetId}`);
    
    if (!response.ok) {
      throw new Error(`Failed to fetch snapshots: ${response.statusText}`);
    }

    return response.json();
  }

  async getCalibrationMethods(): Promise<Array<{
    name: string;
    description: string;
    parameters: string[];
    best_for: string;
  }>> {
    return [
      {
        name: 'svi',
        description: 'Stochastic Volatility Inspired (SVI) - Simple parametric model',
        parameters: ['a', 'b', 'rho', 'm', 'sigma'],
        best_for: 'Smooth interpolation with strong no-arbitrage guarantees'
      },
      {
        name: 'ssvi',
        description: 'Surface SVI - Extension of SVI across term structure',
        parameters: ['theta', 'phi', 'rho', 'eta'],
        best_for: 'Term structure modeling with calendar arbitrage constraints'
      },
      {
        name: 'sabr',
        description: 'Stochastic Alpha Beta Rho - Stochastic volatility model',
        parameters: ['alpha', 'beta', 'rho', 'nu'],
        best_for: 'Capturing volatility smile dynamics and forward skew'
      },
      {
        name: 'spline',
        description: 'Bicubic spline interpolation with smoothing',
        parameters: ['smoothing', 'kx', 'ky'],
        best_for: 'Flexible fitting with minimal parametric assumptions'
      },
      {
        name: 'rbf',
        description: 'Radial Basis Function interpolation',
        parameters: ['kernel'],
        best_for: 'Exact interpolation for scattered data points'
      }
    ];
  }

  // Helper method to format calibration metrics for display
  formatMetrics(metrics: CalibrationMetrics): Record<string, string> {
    return {
      'RMSE': metrics.rmse.toFixed(4),
      'MAE': metrics.mae.toFixed(4),
      'Max Error': metrics.max_error.toFixed(4),
      'R²': metrics.r_squared.toFixed(4),
      'Calibration Time': `${metrics.calibration_time.toFixed(2)}s`,
      'Data Points': metrics.num_points.toString(),
      'Fit Quality': metrics.fit_quality,
      'Timestamp': new Date(metrics.timestamp).toLocaleString()
    };
  }

  // Helper method to assess calibration quality
  assessCalibrationQuality(metrics: CalibrationMetrics): {
    overall: 'Excellent' | 'Good' | 'Fair' | 'Poor';
    recommendations: string[];
  } {
    const recommendations: string[] = [];
    let qualityScore = 0;

    // RMSE scoring
    if (metrics.rmse < 0.01) qualityScore += 3;
    else if (metrics.rmse < 0.05) qualityScore += 2;
    else if (metrics.rmse < 0.1) qualityScore += 1;
    else recommendations.push('High RMSE suggests poor fit - consider different method');

    // R² scoring
    if (metrics.r_squared > 0.9) qualityScore += 3;
    else if (metrics.r_squared > 0.7) qualityScore += 2;
    else if (metrics.r_squared > 0.5) qualityScore += 1;
    else recommendations.push('Low R² indicates poor explanatory power');

    // Data points consideration
    if (metrics.num_points < 20) {
      recommendations.push('Limited data points - consider expanding strike/expiry range');
    }

    // Calibration time consideration
    if (metrics.calibration_time > 10) {
      recommendations.push('Long calibration time - consider simpler model');
    }

    // Overall assessment
    let overall: 'Excellent' | 'Good' | 'Fair' | 'Poor';
    if (qualityScore >= 5) overall = 'Excellent';
    else if (qualityScore >= 3) overall = 'Good';
    else if (qualityScore >= 1) overall = 'Fair';
    else overall = 'Poor';

    if (recommendations.length === 0) {
      recommendations.push('Calibration quality is satisfactory');
    }

    return { overall, recommendations };
  }
}

export const calibrationApi = new CalibrationApiService();