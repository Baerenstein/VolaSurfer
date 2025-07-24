import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import { 
  calibrationApi, 
  CalibrationResult, 
  CalibrationRequest, 
  PerformanceAnalysis,
  SurfaceQualityMetrics,
  CalibrationMetrics 
} from '../../services/calibrationApi';

interface CalibrationPageProps {}

const CalibrationPage: React.FC<CalibrationPageProps> = () => {
  // State management
  const [selectedAsset, setSelectedAsset] = useState<string>('');
  const [selectedSnapshot, setSelectedSnapshot] = useState<string>('');
  const [selectedMethod, setSelectedMethod] = useState<'svi' | 'ssvi' | 'sabr' | 'spline' | 'rbf'>('svi');
  const [calibrationParams, setCalibrationParams] = useState({
    min_moneyness: 0.7,
    max_moneyness: 1.3,
    min_dte: 7,
    max_dte: 365
  });

  // Data state
  const [assets, setAssets] = useState<string[]>([]);
  const [snapshots, setSnapshots] = useState<Array<{id: string, timestamp: string}>>([]);
  const [calibrationMethods, setCalibrationMethods] = useState<Array<{
    name: string;
    description: string;
    parameters: string[];
    best_for: string;
  }>>([]);

  // Results state
  const [calibrationResult, setCalibrationResult] = useState<CalibrationResult | null>(null);
  const [performanceAnalysis, setPerformanceAnalysis] = useState<PerformanceAnalysis | null>(null);
  const [surfaceMetrics, setSurfaceMetrics] = useState<SurfaceQualityMetrics | null>(null);

  // UI state
  const [isCalibrating, setIsCalibrating] = useState(false);
  const [isLoadingPerformance, setIsLoadingPerformance] = useState(false);
  const [isLoadingMetrics, setIsLoadingMetrics] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'calibration' | 'performance' | 'metrics'>('calibration');

  // Load initial data
  useEffect(() => {
    loadInitialData();
  }, []);

  // Load snapshots when asset changes
  useEffect(() => {
    if (selectedAsset) {
      loadSnapshots();
    }
  }, [selectedAsset]);

  const loadInitialData = async () => {
    try {
      const [assetsData, methodsData] = await Promise.all([
        calibrationApi.getAssets(),
        calibrationApi.getCalibrationMethods()
      ]);
      setAssets(assetsData);
      setCalibrationMethods(methodsData);
      if (assetsData.length > 0) {
        setSelectedAsset(assetsData[0]);
      }
    } catch (err) {
      setError(`Failed to load initial data: ${err instanceof Error ? err.message : 'Unknown error'}`);
    }
  };

  const loadSnapshots = async () => {
    try {
      const snapshotsData = await calibrationApi.getSnapshots(selectedAsset);
      setSnapshots(snapshotsData);
      if (snapshotsData.length > 0) {
        setSelectedSnapshot(snapshotsData[0].id);
      }
    } catch (err) {
      setError(`Failed to load snapshots: ${err instanceof Error ? err.message : 'Unknown error'}`);
    }
  };

  const handleCalibration = async () => {
    if (!selectedAsset) {
      setError('Please select an asset');
      return;
    }

    setIsCalibrating(true);
    setError(null);

    try {
      const request: CalibrationRequest = {
        asset_id: selectedAsset,
        snapshot_id: selectedSnapshot || undefined,
        method: selectedMethod,
        ...calibrationParams
      };

      const result = await calibrationApi.calibrateSurface(request);
      setCalibrationResult(result);
    } catch (err) {
      setError(`Calibration failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setIsCalibrating(false);
    }
  };

  const loadPerformanceAnalysis = async () => {
    if (!selectedAsset) return;

    setIsLoadingPerformance(true);
    try {
      const analysis = await calibrationApi.getPerformanceAnalysis(selectedAsset);
      setPerformanceAnalysis(analysis);
    } catch (err) {
      setError(`Failed to load performance analysis: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setIsLoadingPerformance(false);
    }
  };

  const loadSurfaceMetrics = async () => {
    if (!selectedAsset) return;

    setIsLoadingMetrics(true);
    try {
      const metrics = await calibrationApi.getSurfaceMetrics(selectedAsset, selectedSnapshot);
      setSurfaceMetrics(metrics);
    } catch (err) {
      setError(`Failed to load surface metrics: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setIsLoadingMetrics(false);
    }
  };

  const renderSurfacePlot = () => {
    if (!calibrationResult?.fitted_surface) return null;

    const { data, moneyness_grid, dte_grid } = calibrationResult.fitted_surface;

    return (
      <div className="mt-6">
        <h3 className="text-lg font-semibold mb-4">Calibrated Surface</h3>
        <Plot
          data={[
            {
              z: data,
              x: moneyness_grid[0],
              y: dte_grid.map(row => row[0]),
              type: 'surface',
              colorscale: 'Viridis',
              showscale: true,
              name: 'Implied Volatility'
            }
          ]}
          layout={{
            title: `${selectedMethod.toUpperCase()} Calibrated Surface - ${selectedAsset}`,
            scene: {
              xaxis: { title: 'Moneyness' },
              yaxis: { title: 'Days to Expiry' },
              zaxis: { title: 'Implied Volatility' },
              camera: {
                eye: { x: 1.5, y: 1.5, z: 1.5 }
              }
            },
            width: 800,
            height: 600
          }}
        />
      </div>
    );
  };

  const renderMetricsTable = (metrics: CalibrationMetrics) => {
    const formattedMetrics = calibrationApi.formatMetrics(metrics);
    const assessment = calibrationApi.assessCalibrationQuality(metrics);

    return (
      <div className="mt-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h3 className="text-lg font-semibold mb-4">Calibration Metrics</h3>
            <div className="bg-gray-50 rounded-lg p-4">
              <table className="w-full">
                <tbody>
                  {Object.entries(formattedMetrics).map(([key, value]) => (
                    <tr key={key} className="border-b border-gray-200">
                      <td className="py-2 font-medium text-gray-700">{key}:</td>
                      <td className="py-2 text-right">{value}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div>
            <h3 className="text-lg font-semibold mb-4">Quality Assessment</h3>
            <div className="bg-gray-50 rounded-lg p-4">
              <div className="mb-4">
                <span className="font-medium">Overall Quality: </span>
                <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                  assessment.overall === 'Excellent' ? 'bg-green-100 text-green-800' :
                  assessment.overall === 'Good' ? 'bg-blue-100 text-blue-800' :
                  assessment.overall === 'Fair' ? 'bg-yellow-100 text-yellow-800' :
                  'bg-red-100 text-red-800'
                }`}>
                  {assessment.overall}
                </span>
              </div>
              <div>
                <span className="font-medium">Recommendations:</span>
                <ul className="mt-2 list-disc list-inside text-sm text-gray-600">
                  {assessment.recommendations.map((rec, idx) => (
                    <li key={idx}>{rec}</li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderPerformanceAnalysis = () => {
    if (!performanceAnalysis) return null;

    return (
      <div className="space-y-6">
        <h3 className="text-lg font-semibold">Performance Analysis</h3>
        
        {performanceAnalysis.metrics_history.length > 0 ? (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div>
              <h4 className="font-medium mb-4">RMSE Trend</h4>
              <Plot
                data={[
                  {
                    x: performanceAnalysis.metrics_history.map((_, idx) => idx + 1),
                    y: performanceAnalysis.metrics_history.map(m => m.rmse),
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'RMSE',
                    line: { color: 'blue' }
                  }
                ]}
                layout={{
                  title: 'RMSE Over Time',
                  xaxis: { title: 'Calibration #' },
                  yaxis: { title: 'RMSE' },
                  width: 400,
                  height: 300
                }}
              />
            </div>

            <div>
              <h4 className="font-medium mb-4">Calibration Time Trend</h4>
              <Plot
                data={[
                  {
                    x: performanceAnalysis.metrics_history.map((_, idx) => idx + 1),
                    y: performanceAnalysis.metrics_history.map(m => m.calibration_time),
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Time (s)',
                    line: { color: 'orange' }
                  }
                ]}
                layout={{
                  title: 'Calibration Time Over Time',
                  xaxis: { title: 'Calibration #' },
                  yaxis: { title: 'Time (seconds)' },
                  width: 400,
                  height: 300
                }}
              />
            </div>
          </div>
        ) : (
          <div className="text-gray-500 text-center py-8">
            No historical calibration data available for this asset.
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-gray-50 rounded-lg p-4">
            <h4 className="font-medium mb-2">RMSE Trend</h4>
            <span className={`px-2 py-1 rounded text-sm font-medium ${
              performanceAnalysis.trend_analysis.rmse_trend === 'improving' ? 'bg-green-100 text-green-800' :
              performanceAnalysis.trend_analysis.rmse_trend === 'stable' ? 'bg-yellow-100 text-yellow-800' :
              'bg-red-100 text-red-800'
            }`}>
              {performanceAnalysis.trend_analysis.rmse_trend}
            </span>
          </div>

          <div className="bg-gray-50 rounded-lg p-4">
            <h4 className="font-medium mb-2">Calibration Time Trend</h4>
            <span className={`px-2 py-1 rounded text-sm font-medium ${
              performanceAnalysis.trend_analysis.calibration_time_trend === 'improving' ? 'bg-green-100 text-green-800' :
              performanceAnalysis.trend_analysis.calibration_time_trend === 'stable' ? 'bg-yellow-100 text-yellow-800' :
              'bg-red-100 text-red-800'
            }`}>
              {performanceAnalysis.trend_analysis.calibration_time_trend}
            </span>
          </div>

          <div className="bg-gray-50 rounded-lg p-4">
            <h4 className="font-medium mb-2">Fit Quality Distribution</h4>
            <div className="text-sm">
              {Object.entries(performanceAnalysis.trend_analysis.fit_quality_distribution).map(([quality, count]) => (
                <div key={quality} className="flex justify-between">
                  <span>{quality}:</span>
                  <span>{count}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderSurfaceMetrics = () => {
    if (!surfaceMetrics) return null;

    return (
      <div className="space-y-6">
        <h3 className="text-lg font-semibold">Surface Quality Metrics</h3>
        
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-blue-50 rounded-lg p-4 text-center">
            <div className="text-2xl font-bold text-blue-600">{surfaceMetrics.total_points}</div>
            <div className="text-sm text-gray-600">Total Points</div>
          </div>
          
          <div className="bg-green-50 rounded-lg p-4 text-center">
            <div className="text-2xl font-bold text-green-600">
              {(surfaceMetrics.data_coverage * 100).toFixed(1)}%
            </div>
            <div className="text-sm text-gray-600">Data Coverage</div>
          </div>
          
          <div className="bg-purple-50 rounded-lg p-4 text-center">
            <div className="text-2xl font-bold text-purple-600">
              {surfaceMetrics.smoothness_score.toFixed(2)}
            </div>
            <div className="text-sm text-gray-600">Smoothness Score</div>
          </div>
          
          <div className="bg-yellow-50 rounded-lg p-4 text-center">
            <div className="text-2xl font-bold text-yellow-600">{surfaceMetrics.outlier_count}</div>
            <div className="text-sm text-gray-600">Outliers</div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-gray-50 rounded-lg p-4">
            <h4 className="font-medium mb-4">Data Structure</h4>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span>Unique Expiries:</span>
                <span className="font-medium">{surfaceMetrics.unique_expiries}</span>
              </div>
              <div className="flex justify-between">
                <span>Unique Strikes:</span>
                <span className="font-medium">{surfaceMetrics.unique_strikes}</span>
              </div>
              <div className="flex justify-between">
                <span>Avg Bid-Ask Spread:</span>
                <span className="font-medium">{(surfaceMetrics.bid_ask_spread_avg * 100).toFixed(2)}%</span>
              </div>
            </div>
          </div>

          <div className="bg-gray-50 rounded-lg p-4">
            <h4 className="font-medium mb-4">Arbitrage Analysis</h4>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span>Arbitrage Violations:</span>
                <span className={`font-medium ${surfaceMetrics.arbitrage_violations > 0 ? 'text-red-600' : 'text-green-600'}`}>
                  {surfaceMetrics.arbitrage_violations}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Calendar Violations:</span>
                <span className={`font-medium ${surfaceMetrics.calendar_violations > 0 ? 'text-red-600' : 'text-green-600'}`}>
                  {surfaceMetrics.calendar_violations}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gray-100 pt-16">
      <div className="container mx-auto px-4 py-8">
        <h1 className="text-3xl font-bold text-gray-800 mb-8">Surface Calibration</h1>

        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-6">
            {error}
          </div>
        )}

        {/* Control Panel */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-8">
          <h2 className="text-xl font-semibold mb-6">Calibration Parameters</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Asset Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Asset</label>
              <select
                value={selectedAsset}
                onChange={(e) => setSelectedAsset(e.target.value)}
                className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="">Select Asset</option>
                {assets.map(asset => (
                  <option key={asset} value={asset}>{asset}</option>
                ))}
              </select>
            </div>

            {/* Snapshot Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Snapshot</label>
              <select
                value={selectedSnapshot}
                onChange={(e) => setSelectedSnapshot(e.target.value)}
                className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                disabled={!selectedAsset}
              >
                <option value="">Latest Snapshot</option>
                {snapshots.map(snapshot => (
                  <option key={snapshot.id} value={snapshot.id}>
                    {new Date(snapshot.timestamp).toLocaleString()}
                  </option>
                ))}
              </select>
            </div>

            {/* Method Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Calibration Method</label>
              <select
                value={selectedMethod}
                onChange={(e) => setSelectedMethod(e.target.value as any)}
                className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              >
                {calibrationMethods.map(method => (
                  <option key={method.name} value={method.name}>
                    {method.name.toUpperCase()} - {method.description.split(' - ')[1]}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Parameter Ranges */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Min Moneyness</label>
              <input
                type="number"
                step="0.1"
                min="0.1"
                max="2.0"
                value={calibrationParams.min_moneyness}
                onChange={(e) => setCalibrationParams(prev => ({
                  ...prev,
                  min_moneyness: parseFloat(e.target.value)
                }))}
                className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Max Moneyness</label>
              <input
                type="number"
                step="0.1"
                min="0.1"
                max="2.0"
                value={calibrationParams.max_moneyness}
                onChange={(e) => setCalibrationParams(prev => ({
                  ...prev,
                  max_moneyness: parseFloat(e.target.value)
                }))}
                className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Min DTE</label>
              <input
                type="number"
                min="1"
                max="1000"
                value={calibrationParams.min_dte}
                onChange={(e) => setCalibrationParams(prev => ({
                  ...prev,
                  min_dte: parseInt(e.target.value)
                }))}
                className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Max DTE</label>
              <input
                type="number"
                min="1"
                max="1000"
                value={calibrationParams.max_dte}
                onChange={(e) => setCalibrationParams(prev => ({
                  ...prev,
                  max_dte: parseInt(e.target.value)
                }))}
                className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
            </div>
          </div>

          {/* Method Description */}
          {calibrationMethods.find(m => m.name === selectedMethod) && (
            <div className="mt-6 p-4 bg-blue-50 rounded-lg">
              <h3 className="font-medium text-blue-900 mb-2">
                {calibrationMethods.find(m => m.name === selectedMethod)?.name.toUpperCase()} Method
              </h3>
              <p className="text-blue-800 text-sm mb-2">
                {calibrationMethods.find(m => m.name === selectedMethod)?.description}
              </p>
              <p className="text-blue-700 text-xs">
                <strong>Best for:</strong> {calibrationMethods.find(m => m.name === selectedMethod)?.best_for}
              </p>
              <p className="text-blue-700 text-xs mt-1">
                <strong>Parameters:</strong> {calibrationMethods.find(m => m.name === selectedMethod)?.parameters.join(', ')}
              </p>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex flex-wrap gap-4 mt-6">
            <button
              onClick={handleCalibration}
              disabled={isCalibrating || !selectedAsset}
              className="px-6 py-3 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed font-medium"
            >
              {isCalibrating ? 'Calibrating...' : 'Run Calibration'}
            </button>

            <button
              onClick={loadPerformanceAnalysis}
              disabled={isLoadingPerformance || !selectedAsset}
              className="px-6 py-3 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed font-medium"
            >
              {isLoadingPerformance ? 'Loading...' : 'Analyze Performance'}
            </button>

            <button
              onClick={loadSurfaceMetrics}
              disabled={isLoadingMetrics || !selectedAsset}
              className="px-6 py-3 bg-purple-600 text-white rounded-md hover:bg-purple-700 disabled:bg-gray-400 disabled:cursor-not-allowed font-medium"
            >
              {isLoadingMetrics ? 'Loading...' : 'Get Surface Metrics'}
            </button>
          </div>
        </div>

        {/* Results Tabs */}
        <div className="bg-white rounded-lg shadow-md">
          <div className="border-b border-gray-200">
            <nav className="flex space-x-8">
              {[
                { id: 'calibration', label: 'Calibration Results', count: calibrationResult ? 1 : 0 },
                { id: 'performance', label: 'Performance Analysis', count: performanceAnalysis ? 1 : 0 },
                { id: 'metrics', label: 'Surface Metrics', count: surfaceMetrics ? 1 : 0 }
              ].map(tab => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  className={`py-4 px-6 border-b-2 font-medium text-sm ${
                    activeTab === tab.id
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  {tab.label}
                  {tab.count > 0 && (
                    <span className="ml-2 bg-blue-100 text-blue-600 py-0.5 px-2 rounded-full text-xs">
                      {tab.count}
                    </span>
                  )}
                </button>
              ))}
            </nav>
          </div>

          <div className="p-6">
            {activeTab === 'calibration' && calibrationResult && (
              <div>
                {calibrationResult.success ? (
                  <div>
                    <div className="mb-6 p-4 bg-green-50 border border-green-200 rounded-lg">
                      <div className="flex">
                        <div className="flex-shrink-0">
                          <div className="w-5 h-5 text-green-400">✓</div>
                        </div>
                        <div className="ml-3">
                          <h3 className="text-sm font-medium text-green-800">
                            Calibration Successful
                          </h3>
                          <div className="mt-2 text-sm text-green-700">
                            <p>{calibrationResult.message}</p>
                          </div>
                        </div>
                      </div>
                    </div>

                    {calibrationResult.metrics && renderMetricsTable(calibrationResult.metrics)}
                    {renderSurfacePlot()}

                    {/* Model Parameters */}
                    <div className="mt-6">
                      <h3 className="text-lg font-semibold mb-4">Model Parameters</h3>
                      <div className="bg-gray-50 rounded-lg p-4">
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                          {Object.entries(calibrationResult.model_parameters).map(([param, value]) => (
                            <div key={param} className="text-center">
                              <div className="text-sm text-gray-600">{param}</div>
                              <div className="font-medium">
                                {typeof value === 'number' ? value.toFixed(4) : value}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
                    <div className="flex">
                      <div className="flex-shrink-0">
                        <div className="w-5 h-5 text-red-400">✗</div>
                      </div>
                      <div className="ml-3">
                        <h3 className="text-sm font-medium text-red-800">
                          Calibration Failed
                        </h3>
                        <div className="mt-2 text-sm text-red-700">
                          <p>{calibrationResult.message}</p>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}

            {activeTab === 'calibration' && !calibrationResult && (
              <div className="text-center py-12">
                <div className="text-gray-400 text-6xl mb-4">📊</div>
                <h3 className="text-lg font-medium text-gray-900 mb-2">No Calibration Results</h3>
                <p className="text-gray-500">Run a calibration to see results here.</p>
              </div>
            )}

            {activeTab === 'performance' && performanceAnalysis && renderPerformanceAnalysis()}

            {activeTab === 'performance' && !performanceAnalysis && (
              <div className="text-center py-12">
                <div className="text-gray-400 text-6xl mb-4">📈</div>
                <h3 className="text-lg font-medium text-gray-900 mb-2">No Performance Analysis</h3>
                <p className="text-gray-500">Click "Analyze Performance" to load performance data.</p>
              </div>
            )}

            {activeTab === 'metrics' && surfaceMetrics && renderSurfaceMetrics()}

            {activeTab === 'metrics' && !surfaceMetrics && (
              <div className="text-center py-12">
                <div className="text-gray-400 text-6xl mb-4">🔍</div>
                <h3 className="text-lg font-medium text-gray-900 mb-2">No Surface Metrics</h3>
                <p className="text-gray-500">Click "Get Surface Metrics" to load quality metrics.</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default CalibrationPage;