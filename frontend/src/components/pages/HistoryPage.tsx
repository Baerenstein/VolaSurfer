import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';

interface HistoricalSurfaceData {
  id: number;
  timestamp: string;
  asset_id: number;
  moneyness: number[];
  daysToExpiry: number[];
  impliedVols: number[][];
  strikes: number[];
  expiry_dates: string[];
  underlying_price: number;
}

interface Asset {
  id: number;
  ticker: string;
  asset_type: string;
}

const HistoryPage: React.FC = () => {
  const [assets, setAssets] = useState<Asset[]>([]);
  const [selectedAsset, setSelectedAsset] = useState<string>('');
  const [dateRange, setDateRange] = useState({ start: '', end: '' });
  const [isLoading, setIsLoading] = useState(false);
  const [surfaces, setSurfaces] = useState<HistoricalSurfaceData[]>([]);
  const [selectedSurface, setSelectedSurface] = useState<HistoricalSurfaceData | null>(null);
  const [viewMode, setViewMode] = useState<'list' | 'comparison' | 'evolution'>('list');
  const [comparisonSurfaces, setComparisonSurfaces] = useState<HistoricalSurfaceData[]>([]);
  const [isLoadingLargeDataset] = useState(false);

  useEffect(() => {
    fetchAssets();
    
    // Set default date range (last 30 days)
    const end = new Date();
    const start = new Date();
    start.setDate(start.getDate() - 30);
    
    setDateRange({
      start: start.toISOString().split('T')[0],
      end: end.toISOString().split('T')[0]
    });
  }, []);

  const fetchAssets = async () => {
    try {
      const response = await fetch('/api/v1/assets');
      const data = await response.json();
      setAssets(data);
      if (data.length > 0) {
        setSelectedAsset(data[0].id.toString());
      }
    } catch (error) {
      console.error('Error fetching assets:', error);
    }
  };

  const handleSearch = async () => {
    if (!selectedAsset || !dateRange.start || !dateRange.end) return;
    
    setIsLoading(true);
    try {
      const params = new URLSearchParams({
        asset_id: selectedAsset,
        start_date: dateRange.start,
        end_date: dateRange.end
      });
      
      const response = await fetch(`/api/v1/historical-surfaces?${params}`);
      const data = await response.json();
      setSurfaces(data);
      setSelectedSurface(null);
      setComparisonSurfaces([]);
    } catch (error) {
      console.error('Error fetching historical surfaces:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSurfaceSelect = (surface: HistoricalSurfaceData) => {
    if (viewMode === 'comparison') {
      if (comparisonSurfaces.find(s => s.id === surface.id)) {
        setComparisonSurfaces(comparisonSurfaces.filter(s => s.id !== surface.id));
      } else if (comparisonSurfaces.length < 4) {
        setComparisonSurfaces([...comparisonSurfaces, surface]);
      }
    } else {
      setSelectedSurface(surface);
    }
  };

  const renderSurfacePlot = (surface: HistoricalSurfaceData, title?: string, showColorBar = true) => {
    const plotData = [{
      type: 'surface' as const,
      x: surface.moneyness,
      y: surface.daysToExpiry,
      z: surface.impliedVols,
      colorscale: 'Viridis',
      showscale: showColorBar,
      colorbar: showColorBar ? {
        title: 'Implied Vol',
        titleside: 'right' as const
      } : undefined
    }];

    const layout = {
      title: {
        text: title || `Volatility Surface - ${new Date(surface.timestamp).toLocaleDateString()}`,
        font: { color: 'white', size: 14 }
      },
      scene: {
        xaxis: { title: 'Moneyness', color: 'white' },
        yaxis: { title: 'Days to Expiry', color: 'white' },
        zaxis: { title: 'Implied Volatility', color: 'white' },
        bgcolor: 'rgba(0,0,0,0)',
        camera: {
          eye: { x: 1.5, y: 1.5, z: 1.5 }
        }
      },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: { color: 'white' },
      margin: { l: 0, r: 0, t: 30, b: 0 },
      width: 500,
      height: 400
    };

    return (
      <Plot
        data={plotData}
        layout={layout}
        config={{ responsive: true, displayModeBar: false }}
      />
    );
  };

  const renderTimeSeriesAnalysis = () => {
    if (surfaces.length < 2) return null;

    // Calculate average implied volatility for each surface
    const timeSeriesData = surfaces.map(surface => {
      const flatVols = surface.impliedVols.flat().filter(v => typeof v === 'number' && !isNaN(v));
      const avgVol = flatVols.reduce((sum, vol) => sum + vol, 0) / flatVols.length;
      return {
        timestamp: surface.timestamp,
        avgVol: avgVol,
        surface: surface
      };
    }).sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());

    const plotData = [{
      type: 'scatter' as const,
      mode: 'lines+markers' as const,
      x: timeSeriesData.map(d => d.timestamp),
      y: timeSeriesData.map(d => d.avgVol),
      name: 'Average Implied Volatility',
      line: { color: '#3B82F6' },
      marker: { size: 6 }
    }];

    const layout = {
      title: {
        text: 'Implied Volatility Evolution',
        font: { color: 'white' }
      },
      xaxis: { 
        title: 'Date',
        color: 'white',
        tickformat: '%Y-%m-%d'
      },
      yaxis: { 
        title: 'Average Implied Volatility',
        color: 'white'
      },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: { color: 'white' },
      grid: { rows: 1, columns: 1 }
    };

    return (
      <div className="mb-6">
        <Plot
          data={plotData}
          layout={layout}
          style={{ width: '100%', height: '400px' }}
          config={{ responsive: true, displayModeBar: true }}
        />
      </div>
    );
  };

  const renderSurfacesList = () => (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {surfaces.map((surface) => {
        const isSelected = viewMode === 'comparison' 
          ? comparisonSurfaces.find(s => s.id === surface.id)
          : selectedSurface?.id === surface.id;
        
        const flatVols = surface.impliedVols.flat().filter(v => typeof v === 'number' && !isNaN(v));
        const avgVol = flatVols.length > 0 ? flatVols.reduce((sum, vol) => sum + vol, 0) / flatVols.length : 0;
        const minVol = flatVols.length > 0 ? Math.min(...flatVols) : 0;
        const maxVol = flatVols.length > 0 ? Math.max(...flatVols) : 0;
        
        return (
          <div
            key={surface.id}
            onClick={() => handleSurfaceSelect(surface)}
            className={`p-4 border rounded-lg cursor-pointer transition-all ${
              isSelected 
                ? 'border-blue-500 bg-blue-50' 
                : 'border-gray-300 hover:border-gray-400 bg-white'
            }`}
          >
            <div className="flex justify-between items-center mb-2">
              <h3 className="font-semibold text-gray-800">
                {new Date(surface.timestamp).toLocaleDateString()}
              </h3>
              <span className="text-sm text-gray-500">
                {new Date(surface.timestamp).toLocaleTimeString()}
              </span>
            </div>
            
            <div className="space-y-1 text-sm text-gray-600">
              <p>Avg Vol: {(avgVol * 100).toFixed(1)}%</p>
              <p>Vol Range: {(minVol * 100).toFixed(1)}% - {(maxVol * 100).toFixed(1)}%</p>
              <p>Data Points: {surface.moneyness.length * surface.daysToExpiry.length}</p>
              <p>Underlying: ${surface.underlying_price.toFixed(2)}</p>
            </div>
            
            {viewMode === 'comparison' && isSelected && (
              <div className="mt-2 text-xs text-blue-600 font-medium">
                ✓ Selected for comparison
              </div>
            )}
          </div>
        );
      })}
    </div>
  );

  const renderComparison = () => {
    if (comparisonSurfaces.length === 0) {
      return (
        <div className="text-center py-12">
          <div className="text-gray-400 text-6xl mb-4">📊</div>
          <h3 className="text-lg font-medium text-gray-900 mb-2">No Surfaces Selected</h3>
          <p className="text-gray-500">Select up to 4 surfaces to compare them side by side.</p>
        </div>
      );
    }

    return (
      <div className={`grid grid-cols-1 ${comparisonSurfaces.length === 2 ? 'md:grid-cols-2' : comparisonSurfaces.length > 2 ? 'md:grid-cols-2 lg:grid-cols-2' : ''} gap-6`}>
        {comparisonSurfaces.map((surface, index) => (
          <div key={surface.id} className="bg-gray-800 rounded-lg p-4">
            <div className="mb-4">
              <h3 className="text-white font-semibold mb-2">
                Surface {index + 1} - {new Date(surface.timestamp).toLocaleDateString()}
              </h3>
              <div className="grid grid-cols-2 gap-4 text-sm text-gray-300">
                <div>
                  <p>Timestamp: {new Date(surface.timestamp).toLocaleString()}</p>
                  <p>Underlying: ${surface.underlying_price.toFixed(2)}</p>
                </div>
                <div>
                  <p>Strikes: {surface.moneyness.length}</p>
                  <p>Expiries: {surface.daysToExpiry.length}</p>
                </div>
              </div>
            </div>
            {renderSurfacePlot(surface, undefined, false)}
            <button
              onClick={() => setComparisonSurfaces(comparisonSurfaces.filter(s => s.id !== surface.id))}
              className="mt-2 px-3 py-1 bg-red-600 hover:bg-red-700 text-white text-xs rounded"
            >
              Remove
            </button>
          </div>
        ))}
      </div>
    );
  };

  const renderEvolution = () => (
    <div>
      {renderTimeSeriesAnalysis()}
      {surfaces.length > 0 && (
        <div className="mt-6">
          <h3 className="text-lg font-semibold text-white mb-4">Surface Evolution Animation</h3>
          <p className="text-gray-300 text-sm mb-4">
            Click through the surfaces below to see how the volatility surface evolved over time.
          </p>
          
          <div className="flex flex-wrap gap-2 mb-4">
            {surfaces.map((surface, index) => (
              <button
                key={surface.id}
                onClick={() => setSelectedSurface(surface)}
                className={`px-3 py-1 text-xs rounded ${
                  selectedSurface?.id === surface.id
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }`}
              >
                {index + 1}. {new Date(surface.timestamp).toLocaleDateString()}
              </button>
            ))}
          </div>
          
          {selectedSurface && (
            <div className="bg-gray-800 rounded-lg p-6">
              {renderSurfacePlot(selectedSurface)}
              <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4 text-sm text-gray-300">
                <div>
                  <p className="font-medium">Timestamp:</p>
                  <p>{new Date(selectedSurface.timestamp).toLocaleString()}</p>
                </div>
                <div>
                  <p className="font-medium">Underlying Price:</p>
                  <p>${selectedSurface.underlying_price.toFixed(2)}</p>
                </div>
                <div>
                  <p className="font-medium">Strikes:</p>
                  <p>{selectedSurface.moneyness.length}</p>
                </div>
                <div>
                  <p className="font-medium">Expiries:</p>
                  <p>{selectedSurface.daysToExpiry.length}</p>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );

  const renderMainContent = () => {
    if (isLoading) {
      return (
        <div className="flex items-center justify-center py-12">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
          <span className="ml-3 text-white">Loading historical surfaces...</span>
        </div>
      );
    }

    if (surfaces.length === 0) {
      return (
        <div className="text-center py-12">
          <div className="text-gray-400 text-6xl mb-4">📈</div>
          <h3 className="text-lg font-medium text-gray-200 mb-2">No Historical Data</h3>
          <p className="text-gray-400">Select an asset and date range to view historical volatility surfaces.</p>
        </div>
      );
    }

    switch (viewMode) {
      case 'list':
        return (
          <div>
            {renderSurfacesList()}
            {selectedSurface && (
              <div className="mt-8 bg-gray-800 rounded-lg p-6">
                <h3 className="text-xl font-semibold text-white mb-4">
                  Selected Surface - {new Date(selectedSurface.timestamp).toLocaleDateString()}
                </h3>
                
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <div>
                    {renderSurfacePlot(selectedSurface)}
                  </div>
                  
                  <div className="space-y-4">
                    <div className="bg-gray-700 rounded p-4">
                      <h4 className="font-semibold text-white mb-3">Surface Statistics</h4>
                      <div className="space-y-2 text-sm text-gray-300">
                        <p>Timestamp: {new Date(selectedSurface.timestamp).toLocaleString()}</p>
                        <p>Underlying Price: ${selectedSurface.underlying_price.toFixed(2)}</p>
                        <p>Number of Strikes: {selectedSurface.moneyness.length}</p>
                        <p>Number of Expiries: {selectedSurface.daysToExpiry.length}</p>
                        <p>Moneyness Range: {Math.min(...selectedSurface.moneyness).toFixed(3)} - {Math.max(...selectedSurface.moneyness).toFixed(3)}</p>
                        <p>DTE Range: {Math.min(...selectedSurface.daysToExpiry).toFixed(1)} - {Math.max(...selectedSurface.daysToExpiry).toFixed(1)} days</p>
                        {(() => {
                          const flatVols = selectedSurface.impliedVols.flat().filter(v => typeof v === 'number' && !isNaN(v));
                          const minVol = flatVols.length > 0 ? Math.min(...flatVols) : 0;
                          const maxVol = flatVols.length > 0 ? Math.max(...flatVols) : 0;
                          return <p>Vol Range: {(minVol * 100).toFixed(1)}% - {(maxVol * 100).toFixed(1)}%</p>;
                        })()}
                      </div>
                    </div>
                    
                    <div className="bg-gray-700 rounded p-4">
                      <h4 className="font-semibold text-white mb-3">Actions</h4>
                      <div className="space-y-2">
                        <button 
                          onClick={() => {
                            setViewMode('comparison');
                            setComparisonSurfaces([selectedSurface]);
                          }}
                          className="w-full px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded"
                        >
                          Add to Comparison
                        </button>
                        <button 
                          onClick={() => {/* TODO: Export functionality */}}
                          className="w-full px-3 py-2 bg-green-600 hover:bg-green-700 text-white text-sm rounded"
                        >
                          Export Data
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        );
      
      case 'comparison':
        return renderComparison();
      
      case 'evolution':
        return renderEvolution();
      
      default:
        return null;
    }
  };

  return (
    <div className="min-h-screen bg-black p-6">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-white mb-2">Historical Analysis</h1>
          <p className="text-gray-400">Explore historical volatility surfaces and analyze evolution over time</p>
        </div>

        {/* Search Controls */}
        <div className="bg-gray-800 rounded-lg p-6 mb-6">
          <h3 className="text-xl font-semibold text-white mb-4">Search Parameters</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">Asset</label>
              <select
                value={selectedAsset}
                onChange={(e) => setSelectedAsset(e.target.value)}
                className="w-full p-2 bg-gray-700 border border-gray-600 rounded text-white"
              >
                <option value="">Select Asset</option>
                {assets.map((asset) => (
                  <option key={asset.id} value={asset.id.toString()}>
                    {asset.ticker} ({asset.asset_type})
                  </option>
                ))}
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">Start Date</label>
              <input
                type="date"
                value={dateRange.start}
                onChange={(e) => setDateRange(prev => ({ ...prev, start: e.target.value }))}
                className="w-full p-2 bg-gray-700 border border-gray-600 rounded text-white"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">End Date</label>
              <input
                type="date"
                value={dateRange.end}
                onChange={(e) => setDateRange(prev => ({ ...prev, end: e.target.value }))}
                className="w-full p-2 bg-gray-700 border border-gray-600 rounded text-white"
              />
            </div>
            
            <div className="flex items-end">
              <button
                onClick={handleSearch}
                disabled={isLoading || !selectedAsset}
                className="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white rounded transition-colors"
              >
                {isLoading ? 'Searching...' : 'Search'}
              </button>
            </div>
          </div>

          {/* View Mode Selector */}
          <div className="flex gap-2">
            <button
              onClick={() => setViewMode('list')}
              className={`px-4 py-2 rounded text-sm ${
                viewMode === 'list' 
                  ? 'bg-blue-600 text-white' 
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              List View
            </button>
            <button
              onClick={() => setViewMode('comparison')}
              className={`px-4 py-2 rounded text-sm ${
                viewMode === 'comparison' 
                  ? 'bg-blue-600 text-white' 
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              Comparison ({comparisonSurfaces.length}/4)
            </button>
            <button
              onClick={() => setViewMode('evolution')}
              className={`px-4 py-2 rounded text-sm ${
                viewMode === 'evolution' 
                  ? 'bg-blue-600 text-white' 
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              Evolution
            </button>
          </div>
        </div>

        {/* Results */}
        <div className="bg-gray-800 rounded-lg p-6">
          <div className="flex justify-between items-center mb-6">
            <h3 className="text-xl font-semibold text-white">
              {viewMode === 'list' && 'Historical Surfaces'}
              {viewMode === 'comparison' && 'Surface Comparison'}
              {viewMode === 'evolution' && 'Surface Evolution'}
            </h3>
            {surfaces.length > 0 && (
              <span className="text-gray-400 text-sm">
                {surfaces.length} surface{surfaces.length !== 1 ? 's' : ''} found
              </span>
            )}
          </div>
          
          {renderMainContent()}
        </div>
      </div>
    </div>
  );
};

export default HistoryPage; 