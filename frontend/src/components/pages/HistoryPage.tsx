import React, { useState, useMemo, useEffect, useRef } from 'react';
import Plot from 'react-plotly.js';
import { useSurfaceHistory } from '../../hooks/useSurfaceHistory';
import { HistoricalSurfaceData } from '../../types/surface';

interface ContainerProps {
  title: string;
  children?: React.ReactNode;
}

const Container: React.FC<ContainerProps> = ({ title, children }) => (
  <div className="flex-1 m-4 bg-white rounded-lg shadow-lg">
    <div className="p-6">
      <h2 className="text-xl font-semibold mb-4">{title}</h2>
      {children}
    </div>
  </div>
);

const HistoryPage: React.FC = () => {
  const [limit, setLimit] = useState(100);
  const [selectedSurfaceIndex, setSelectedSurfaceIndex] = useState<number>(0);
  const [showWireframe, setShowWireframe] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1000); // milliseconds between frames
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  
  const { data, isLoading, error, refetch } = useSurfaceHistory({ limit });

  const selectedSurface = useMemo(() => {
    if (!data || data.length === 0) return null;
    return data[selectedSurfaceIndex] || data[0];
  }, [data, selectedSurfaceIndex]);

  // Playback controls
  const startPlayback = () => {
    if (!data || data.length === 0) return;
    
    setIsPlaying(true);
    intervalRef.current = setInterval(() => {
      setSelectedSurfaceIndex(prevIndex => {
        const nextIndex = prevIndex + 1;
        if (nextIndex >= data.length) {
          // Stop at the end or loop back to beginning
          setIsPlaying(false);
          return 0; // Loop back to start
          // return prevIndex; // Stop at end
        }
        return nextIndex;
      });
    }, playbackSpeed);
  };

  const stopPlayback = () => {
    setIsPlaying(false);
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  };

  const resetToStart = () => {
    stopPlayback();
    setSelectedSurfaceIndex(0);
  };

  const goToEnd = () => {
    stopPlayback();
    if (data && data.length > 0) {
      setSelectedSurfaceIndex(data.length - 1);
    }
  };

  const stepForward = () => {
    if (data && selectedSurfaceIndex < data.length - 1) {
      setSelectedSurfaceIndex(prev => prev + 1);
    }
  };

  const stepBackward = () => {
    if (selectedSurfaceIndex > 0) {
      setSelectedSurfaceIndex(prev => prev - 1);
    }
  };

  // Cleanup interval on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  // Stop playback when reaching the end
  useEffect(() => {
    if (data && selectedSurfaceIndex >= data.length - 1 && isPlaying) {
      stopPlayback();
    }
  }, [selectedSurfaceIndex, data, isPlaying]);

  const plotData = useMemo(() => {
    if (!selectedSurface) return [];
    
    console.log(`Processing surface ${selectedSurfaceIndex + 1}/${data?.length || 0}:`, selectedSurface);
    
    // Check if we have the required data
    if (!selectedSurface.moneyness || !selectedSurface.daysToExpiry || !selectedSurface.impliedVols) {
      console.warn('Missing required data for plotting');
      return [];
    }

    if (selectedSurface.moneyness.length === 0 || 
        selectedSurface.daysToExpiry.length === 0 || 
        selectedSurface.impliedVols.length === 0) {
      console.warn('Empty arrays detected');
      return [];
    }
    
    // Get unique values for matrix dimensions
    const uniqueMoneyness = [...new Set(selectedSurface.moneyness)].sort((a, b) => a - b);
    const uniqueDaysToExpiry = [...new Set(selectedSurface.daysToExpiry)].sort((a, b) => a - b);
    
    console.log('Grid dimensions:', {
      moneynessPoints: uniqueMoneyness.length,
      dtePoints: uniqueDaysToExpiry.length,
      totalDataPoints: selectedSurface.impliedVols.length
    });
    
    // Handle flat array from backend (VolSurface.to_dict())
    if (Array.isArray(selectedSurface.impliedVols) && 
        !Array.isArray(selectedSurface.impliedVols[0])) {
      
      // Create a matrix organized by [days_to_expiry][moneyness]
      const matrix: number[][] = [];
      
      // Initialize matrix with NaN values
      for (let i = 0; i < uniqueDaysToExpiry.length; i++) {
        matrix[i] = new Array(uniqueMoneyness.length).fill(NaN);
      }
      
      // Fill matrix with actual values
      for (let i = 0; i < selectedSurface.impliedVols.length; i++) {
        const moneyness = selectedSurface.moneyness[i];
        const dte = selectedSurface.daysToExpiry[i];
        const vol = selectedSurface.impliedVols[i];
        
        const dteIndex = uniqueDaysToExpiry.indexOf(dte);
        const moneynessIndex = uniqueMoneyness.indexOf(moneyness);
        
        if (dteIndex >= 0 && moneynessIndex >= 0 && typeof vol === 'number' && !isNaN(vol)) {
          matrix[dteIndex][moneynessIndex] = vol / 100; // Convert percentage to decimal
        }
      }
      
      // Filter out rows/columns that are all NaN
      const validMatrix = matrix.filter(row => row.some(val => !isNaN(val)));
      
      if (validMatrix.length === 0 || validMatrix[0].length === 0) {
        console.warn('No valid data points found');
        return [];
      }
      
      // Calculate value range for better color scaling
      const allValues = validMatrix.flat().filter(val => !isNaN(val));
      const minVal = Math.min(...allValues);
      const maxVal = Math.max(...allValues);
      
      return [{
        type: 'surface' as const,
        x: uniqueMoneyness,
        y: uniqueDaysToExpiry,
        z: validMatrix,
        showscale: true,
        colorscale: "Viridis",
        // Add explicit color range for better visualization
        cmin: minVal,
        cmax: maxVal,
        contours: {
          z: {
            show: true,
            usecolormap: true,
            highlightcolor: "#42f462",
            project: { z: true },
          },
        },
        // Add wireframe option for debugging
        ...(showWireframe && {
          surfacecolor: validMatrix,
          showscale: false,
          opacity: 0.7,
          contours: {
            x: { show: true, color: "white", width: 2 },
            y: { show: true, color: "white", width: 2 },
            z: { show: true, color: "white", width: 2 },
          }
        }),
        opacity: showWireframe ? 0.7 : 1,
        hoverongaps: false,
        hoverlabel: {
          bgcolor: "#FFF",
          font: { color: "#000" },
        },
        hovertemplate: 'Moneyness: %{x:.3f}<br>Days to Expiry: %{y:.0f}<br>Implied Vol: %{z:.2%}<extra></extra>',
      }];
    }
    
    return [];
  }, [selectedSurface, showWireframe, selectedSurfaceIndex, data]);

  const layout = useMemo(() => ({
    title: {
      text: `Historical Volatility Surface - ${selectedSurface ? new Date(selectedSurface.timestamp).toLocaleString() : ''} (${selectedSurfaceIndex + 1}/${data?.length || 0})`,
      font: { size: 14 }
    },
    scene: {
      xaxis: { 
        title: "Moneyness", 
        tickformat: ".3f",
        autorange: true,
      },
      yaxis: { 
        title: "Days to Expiry", 
        tickformat: ".0f",
        autorange: true,
      },
      zaxis: { 
        title: "Implied Volatility", 
        tickformat: ".1%",
        autorange: true,
      },
      camera: {
        eye: { x: 1.5, y: 1.5, z: 1.5 },
        center: { x: 0, y: 0, z: 0 },
        up: { x: 0, y: 0, z: 1 },
      },
      aspectratio: { x: 1, y: 1, z: 0.8 },
      bgcolor: 'rgba(0,0,0,0)',
    },
    margin: { l: 0, r: 0, b: 0, t: 50 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    autosize: true,
  }), [selectedSurface, selectedSurfaceIndex, data]);

  if (isLoading) {
    return (
      <div className="flex flex-col w-full min-h-screen bg-gray-100">
        <div className="flex flex-1 p-4 justify-center items-center">
          <div className="text-lg">Loading historical data...</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col w-full min-h-screen bg-gray-100">
        <div className="flex flex-1 p-4 justify-center items-center">
          <div className="text-red-500 text-center">
            <div className="text-lg mb-4">Error loading historical data</div>
            <div className="text-sm mb-4">{error}</div>
            <button 
              onClick={refetch}
              className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
            >
              Retry
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col w-full min-h-screen bg-gray-100">
      <div className="flex flex-1 p-4">
        <Container title="Historical Volatility Surfaces">
          <div className="space-y-4">
            {/* Main Controls */}
            <div className="flex items-center space-x-4 p-4 bg-gray-50 rounded">
              <div className="flex items-center space-x-2">
                <label htmlFor="limit" className="text-sm text-gray-600">
                  Limit:
                </label>
                <select
                  id="limit"
                  value={limit}
                  onChange={(e) => {
                    setLimit(Number(e.target.value));
                    stopPlayback(); // Stop playback when changing limit
                  }}
                  className="px-3 py-1 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value={10}>10</option>
                  <option value={25}>25</option>
                  <option value={50}>50</option>
                  <option value={100}>100</option>
                </select>
              </div>

              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  id="wireframe"
                  checked={showWireframe}
                  onChange={(e) => setShowWireframe(e.target.checked)}
                  className="rounded"
                />
                <label htmlFor="wireframe" className="text-sm text-gray-600">
                  Wireframe
                </label>
              </div>

              <button 
                onClick={refetch}
                className="px-3 py-1 bg-blue-500 text-white rounded text-sm hover:bg-blue-600"
              >
                Refresh
              </button>
            </div>

            {/* Animation Controls */}
            {data && data.length > 1 && (
              <div className="p-4 bg-blue-50 border border-blue-200 rounded">
                <h3 className="text-blue-800 font-medium mb-3">Timeline Playback</h3>
                
                {/* Playback Controls */}
                <div className="flex items-center space-x-2 mb-3">
                  <button
                    onClick={resetToStart}
                    className="px-3 py-1 bg-gray-500 text-white rounded text-sm hover:bg-gray-600"
                    title="Go to start"
                  >
                    ⏮
                  </button>
                  
                  <button
                    onClick={stepBackward}
                    disabled={selectedSurfaceIndex === 0}
                    className="px-3 py-1 bg-gray-500 text-white rounded text-sm hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed"
                    title="Step backward"
                  >
                    ⏪
                  </button>
                  
                  <button
                    onClick={isPlaying ? stopPlayback : startPlayback}
                    className={`px-4 py-2 rounded text-sm font-medium ${
                      isPlaying 
                        ? 'bg-red-500 hover:bg-red-600 text-white' 
                        : 'bg-green-500 hover:bg-green-600 text-white'
                    }`}
                  >
                    {isPlaying ? '⏸ Pause' : '▶ Play'}
                  </button>
                  
                  <button
                    onClick={stepForward}
                    disabled={selectedSurfaceIndex >= data.length - 1}
                    className="px-3 py-1 bg-gray-500 text-white rounded text-sm hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed"
                    title="Step forward"
                  >
                    ⏩
                  </button>
                  
                  <button
                    onClick={goToEnd}
                    className="px-3 py-1 bg-gray-500 text-white rounded text-sm hover:bg-gray-600"
                    title="Go to end"
                  >
                    ⏭
                  </button>
                </div>

                {/* Timeline Slider */}
                <div className="mb-3">
                  <input
                    type="range"
                    min={0}
                    max={data.length - 1}
                    value={selectedSurfaceIndex}
                    onChange={(e) => {
                      stopPlayback();
                      setSelectedSurfaceIndex(Number(e.target.value));
                    }}
                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                  />
                  <div className="flex justify-between text-xs text-gray-500 mt-1">
                    <span>Oldest</span>
                    <span>Frame {selectedSurfaceIndex + 1} of {data.length}</span>
                    <span>Newest</span>
                  </div>
                </div>

                {/* Speed Control */}
                <div className="flex items-center space-x-2">
                  <label htmlFor="speed" className="text-sm text-gray-600">
                    Speed:
                  </label>
                  <select
                    id="speed"
                    value={playbackSpeed}
                    onChange={(e) => setPlaybackSpeed(Number(e.target.value))}
                    className="px-2 py-1 border border-gray-300 rounded text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value={2000}>0.5x (2s)</option>
                    <option value={1000}>1x (1s)</option>
                    <option value={500}>2x (0.5s)</option>
                    <option value={250}>4x (0.25s)</option>
                    <option value={100}>10x (0.1s)</option>
                  </select>
                </div>
              </div>
            )}

            {/* Manual Surface Selection (for single selection) */}
            {data && data.length > 0 && (
              <div className="flex items-center space-x-4 p-4 bg-gray-50 rounded">
                <div className="flex items-center space-x-2">
                  <label htmlFor="surface-select" className="text-sm text-gray-600">
                    Manual Selection:
                  </label>
                  <select
                    id="surface-select"
                    value={selectedSurfaceIndex}
                    onChange={(e) => {
                      stopPlayback();
                      setSelectedSurfaceIndex(Number(e.target.value));
                    }}
                    className="px-3 py-1 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    {data.map((surface, index) => (
                      <option key={index} value={index}>
                        {index + 1}: {new Date(surface.timestamp).toLocaleString()}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            )}

            {/* Surface Plot */}
            {selectedSurface && plotData.length > 0 && (
              <div className="bg-white rounded-lg shadow-lg overflow-hidden">
                <div className="p-4">
                  <Plot
                    data={plotData}
                    layout={layout}
                    style={{
                      width: "100%",
                      height: "70vh",
                    }}
                    config={{
                      responsive: true,
                      displayModeBar: true,
                      modeBarButtonsToRemove: ["toImage", "sendDataToCloud"],
                      displaylogo: false,
                    }}
                    useResizeHandler={true}
                  />
                </div>
              </div>
            )}
            
            {/* Show debug info when no plot data */}
            {selectedSurface && plotData.length === 0 && (
              <div className="bg-yellow-50 border border-yellow-200 rounded p-4">
                <h3 className="text-yellow-800 font-medium mb-2">No Surface Data Available</h3>
                <div className="text-sm text-yellow-700">
                  <p>This surface doesn't have enough data points to create a visualization.</p>
                  <p>Surface {selectedSurfaceIndex + 1} has:</p>
                  <ul className="ml-4 mt-2">
                    <li>• {selectedSurface.moneyness?.length || 0} moneyness points</li>
                    <li>• {selectedSurface.daysToExpiry?.length || 0} days to expiry points</li>
                    <li>• {selectedSurface.impliedVols?.length || 0} implied vol points</li>
                    <li>• {selectedSurface.daysToExpiry ? [...new Set(selectedSurface.daysToExpiry)].length : 0} unique expiries</li>
                  </ul>
                  <p className="mt-2">Try selecting a different surface or use the playback controls to find surfaces with more data.</p>
                </div>
              </div>
            )}
          </div>
        </Container>

        {/* Surface Data Summary */}
        <Container title="Surface Information">
          {selectedSurface ? (
            <div className="space-y-4">
              <div className="text-sm text-gray-600">
                <p><strong>Frame:</strong> {selectedSurfaceIndex + 1} of {data?.length || 0}</p>
                <p><strong>Timestamp:</strong> {new Date(selectedSurface.timestamp).toLocaleString()}</p>
                <p><strong>Snapshot ID:</strong> {selectedSurface.snapshot_id || 'N/A'}</p>
                <p><strong>Asset ID:</strong> {selectedSurface.asset_id || 'N/A'}</p>
                <p><strong>Method:</strong> {selectedSurface.method || 'N/A'}</p>
                <p><strong>Status:</strong> {isPlaying ? '▶ Playing' : '⏸ Paused'}</p>
              </div>
              
              <div className="space-y-2">
                <h4 className="font-medium text-gray-700">Surface Dimensions:</h4>
                <div className="text-sm text-gray-600">
                  <p>Total Points: {selectedSurface.moneyness?.length || 0}</p>
                  <p>Unique Strikes: {selectedSurface.moneyness ? [...new Set(selectedSurface.moneyness)].length : 0}</p>
                  <p>Unique Expiries: {selectedSurface.daysToExpiry ? [...new Set(selectedSurface.daysToExpiry)].length : 0}</p>
                  {selectedSurface.moneyness && selectedSurface.moneyness.length > 0 && (
                    <p>Moneyness Range: {Math.min(...selectedSurface.moneyness).toFixed(3)} - {Math.max(...selectedSurface.moneyness).toFixed(3)}</p>
                  )}
                  {selectedSurface.daysToExpiry && selectedSurface.daysToExpiry.length > 0 && (
                    <p>DTE Range: {Math.min(...selectedSurface.daysToExpiry).toFixed(1)} - {Math.max(...selectedSurface.daysToExpiry).toFixed(1)} days</p>
                  )}
                  {selectedSurface.impliedVols && selectedSurface.impliedVols.length > 0 && (
                    <p>Vol Range: {(Math.min(...selectedSurface.impliedVols.filter(v => typeof v === 'number' && !isNaN(v))) / 100).toFixed(1)}% - {(Math.max(...selectedSurface.impliedVols.filter(v => typeof v === 'number' && !isNaN(v))) / 100).toFixed(1)}%</p>
                  )}
                </div>
              </div>

              {data && (
                <div className="space-y-2">
                  <h4 className="font-medium text-gray-700">History Overview:</h4>
                  <div className="text-sm text-gray-600">
                    <p>Total Surfaces: {data.length}</p>
                    <p>Oldest: {data.length > 0 ? new Date(data[data.length - 1].timestamp).toLocaleString() : 'N/A'}</p>
                    <p>Newest: {data.length > 0 ? new Date(data[0].timestamp).toLocaleString() : 'N/A'}</p>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="text-gray-500">Select a surface to view details</div>
          )}
        </Container>
      </div>
    </div>
  );
};

export default HistoryPage; 