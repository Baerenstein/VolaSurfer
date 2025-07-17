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
  const [limit, setLimit] = useState(500); // Changed from 100 to 500
  const [selectedSurfaceIndex, setSelectedSurfaceIndex] = useState<number>(0);
  const [showWireframe, setShowWireframe] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1000);
  const [interpolationDensity, setInterpolationDensity] = useState(20); // New interpolation control
  const [isLoadingLargeDataset, setIsLoadingLargeDataset] = useState(false);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  
  const { data, isLoading, error, refetch } = useSurfaceHistory({ limit });

  const selectedSurface = useMemo(() => {
    if (!data || data.length === 0) return null;
    return data[selectedSurfaceIndex] || data[0];
  }, [data, selectedSurfaceIndex]);

  // Playback controls - FIXED to play chronologically (oldest to newest)
  const startPlayback = () => {
    if (!data || data.length === 0) return;
    
    setIsPlaying(true);
    intervalRef.current = setInterval(() => {
      setSelectedSurfaceIndex(prevIndex => {
        const nextIndex = prevIndex - 1; // Changed from +1 to -1 (going backward through the array = forward in time)
        if (nextIndex < 0) { // Changed from >= data.length to < 0
          setIsPlaying(false);
          return data.length - 1; // Loop back to end (oldest surface)
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

  const stepForward = () => {
    stopPlayback();
    setSelectedSurfaceIndex(prev => Math.max(prev - 1, 0)); // Changed: going backward in array = forward in time
  };

  const stepBackward = () => {
    stopPlayback();
    setSelectedSurfaceIndex(prev => Math.min(prev + 1, (data?.length || 1) - 1)); // Changed: going forward in array = backward in time
  };

  const resetToStart = () => {
    stopPlayback();
    setSelectedSurfaceIndex((data?.length || 1) - 1); // Changed: start at oldest (last index)
  };

  const goToEnd = () => {
    stopPlayback();
    setSelectedSurfaceIndex(0); // Changed: end at newest (first index)
  };

  // Cleanup interval on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  // Auto-update playback interval when speed changes
  useEffect(() => {
    if (isPlaying && intervalRef.current) {
      // Just update the existing interval without stopping/starting
      clearInterval(intervalRef.current);
      intervalRef.current = setInterval(() => {
        setSelectedSurfaceIndex(prevIndex => {
          const nextIndex = prevIndex - 1;
          if (nextIndex < 0) {
            setIsPlaying(false);
            return data?.length ? data.length - 1 : 0;
          }
          return nextIndex;
        });
      }, playbackSpeed);
    }
  }, [playbackSpeed, data]);

  // Stop playback when we reach the end (newest surface at index 0)
  useEffect(() => {
    if (isPlaying && data && selectedSurfaceIndex <= 0) {
      stopPlayback();
    }
  }, [selectedSurfaceIndex, data, isPlaying]);

  // Enhanced plot data with better interpolation
  const plotData = useMemo(() => {
    if (!selectedSurface) return [];
    
    console.log(`Processing surface ${selectedSurfaceIndex + 1}/${data?.length || 0}:`, selectedSurface);
    
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
    
    if (Array.isArray(selectedSurface.impliedVols) && !Array.isArray(selectedSurface.impliedVols[0])) {
      
      // Create interpolated grid for smoother surface
      const moneynessMin = Math.min(...uniqueMoneyness);
      const moneynessMax = Math.max(...uniqueMoneyness);
      const dteMin = Math.min(...uniqueDaysToExpiry);
      const dteMax = Math.max(...uniqueDaysToExpiry);
      
      // Create denser grid for smoother interpolation
      const gridMoneyness = Array.from({length: interpolationDensity}, (_, i) => 
        moneynessMin + (moneynessMax - moneynessMin) * i / (interpolationDensity - 1)
      );
      const gridDTE = Array.from({length: Math.max(interpolationDensity, uniqueDaysToExpiry.length)}, (_, i) => 
        dteMin + (dteMax - dteMin) * i / (Math.max(interpolationDensity, uniqueDaysToExpiry.length) - 1)
      );
      
      // Create matrix with bilinear interpolation
      const matrix: number[][] = [];
      
      for (let dteIdx = 0; dteIdx < gridDTE.length; dteIdx++) {
        const currentDTE = gridDTE[dteIdx];
        matrix[dteIdx] = [];
        
        for (let mIdx = 0; mIdx < gridMoneyness.length; mIdx++) {
          const currentMoneyness = gridMoneyness[mIdx];
          
          // Find the interpolated value at this grid point
          let interpolatedValue = NaN;
          
          // Simple nearest neighbor for now, could enhance with bilinear interpolation
          let minDistance = Infinity;
          for (let i = 0; i < selectedSurface.impliedVols.length; i++) {
            const dataMoneyness = selectedSurface.moneyness[i];
            const dataDTE = selectedSurface.daysToExpiry[i];
            const dataVol = selectedSurface.impliedVols[i];
            
            if (typeof dataVol === 'number' && !isNaN(dataVol)) {
              const distance = Math.sqrt(
                Math.pow((currentMoneyness - dataMoneyness) / (moneynessMax - moneynessMin), 2) +
                Math.pow((currentDTE - dataDTE) / (dteMax - dteMin), 2)
              );
              
              if (distance < minDistance) {
                minDistance = distance;
                interpolatedValue = dataVol / 100; // Convert percentage to decimal
              }
            }
          }
          
          // Only include values that are reasonably close
          matrix[dteIdx][mIdx] = minDistance < 0.3 ? interpolatedValue : NaN;
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
        x: gridMoneyness,
        y: gridDTE,
        z: validMatrix,
        showscale: true,
        colorscale: "Viridis",
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
        // Enhanced surface properties for smoother rendering
        surfacecolor: validMatrix,
        opacity: showWireframe ? 0.7 : 0.9,
        lighting: {
          ambient: 0.4,
          diffuse: 0.6,
          fresnel: 0.2,
          specular: 0.05,
          roughness: 0.1,
        },
        lightposition: {
          x: 100,
          y: 200,
          z: 0
        },
        hoverongaps: false,
        hoverlabel: {
          bgcolor: 'rgba(0,0,0,0.8)',
          bordercolor: 'white',
          font: { color: 'white', size: 12 }
        },
        hovertemplate: 
          '<b>Moneyness:</b> %{x:.3f}<br>' +
          '<b>Days to Expiry:</b> %{y:.1f}<br>' +
          '<b>Implied Vol:</b> %{z:.2%}<br>' +
          '<extra></extra>',
        ...(showWireframe && {
          contours: {
            x: { show: true, color: "white", width: 2 },
            y: { show: true, color: "white", width: 2 },
            z: { show: true, color: "white", width: 2 },
          }
        }),
      }];
    }
    
    return [];
  }, [selectedSurface, showWireframe, interpolationDensity]);

  const layout = useMemo(() => ({
    title: {
      text: `Historical Volatility Surface - ${selectedSurface ? new Date(selectedSurface.timestamp).toLocaleString() : ''}`,
      font: { size: 16 }
    },
    scene: {
      xaxis: {
        title: 'Moneyness',
        titlefont: { size: 14 },
        tickfont: { size: 12 }
      },
      yaxis: {
        title: 'Days to Expiry',
        titlefont: { size: 14 },
        tickfont: { size: 12 }
      },
      zaxis: {
        title: 'Implied Volatility',
        titlefont: { size: 14 },
        tickfont: { size: 12 },
        tickformat: '.1%'
      },
      camera: {
        eye: { x: 1.2, y: 1.2, z: 1.2 }
      },
      aspectmode: 'cube'
    },
    margin: { l: 0, r: 0, b: 0, t: 40 },
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
                    const newLimit = Number(e.target.value);
                    if (newLimit >= 2000) {
                      const confirmed = window.confirm(
                        `Loading ${newLimit.toLocaleString()} surfaces may take some time and use significant memory. Continue?`
                      );
                      if (!confirmed) return;
                    }
                    setLimit(newLimit);
                    stopPlayback();
                  }}
                  className="px-3 py-1 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value={10}>10</option>
                  <option value={25}>25</option>
                  <option value={50}>50</option>
                  <option value={100}>100</option>
                  <option value={200}>200</option>
                  <option value={500}>500</option>
                  <option value={1000}>1,000</option>
                  <option value={2000}>2,000 ⚠️</option>
                  <option value={5000}>5,000 ⚠️</option>
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

              <div className="flex items-center space-x-2">
                <label htmlFor="interpolation" className="text-sm text-gray-600">
                  Smoothness:
                </label>
                <select
                  id="interpolation"
                  value={interpolationDensity}
                  onChange={(e) => setInterpolationDensity(Number(e.target.value))}
                  className="px-3 py-1 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value={10}>Low (10x10)</option>
                  <option value={20}>Medium (20x20)</option>
                  <option value={30}>High (30x30)</option>
                  <option value={50}>Ultra (50x50)</option>
                </select>
              </div>

              <button 
                onClick={refetch}
                className="px-3 py-1 bg-blue-500 text-white rounded text-sm hover:bg-blue-600"
              >
                Refresh
              </button>
            </div>
            {isLoadingLargeDataset && isLoading && (
              <div className="p-3 bg-yellow-50 border border-yellow-200 rounded text-sm text-yellow-800">
                Loading {limit} surfaces... This may take a moment for large datasets.
              </div>
            )}

            {/* Manual Surface Selection */}
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

            {/* Animation Controls - NOW BELOW THE PLOT */}
            {data && data.length > 1 && (
              <div className="p-4 bg-blue-50 border border-blue-200 rounded">
                <h3 className="text-blue-800 font-medium mb-3">Timeline Playback</h3>
                
                {/* Playback Controls */}
                <div className="flex items-center space-x-2 mb-3">
                  <button
                    onClick={resetToStart}
                    className="px-3 py-1 bg-gray-500 text-white rounded text-sm hover:bg-gray-600"
                    title="Go to oldest (start of timeline)"
                  >
                    ⏮
                  </button>
                  
                  <button
                    onClick={stepBackward}
                    disabled={selectedSurfaceIndex >= data.length - 1}
                    className="px-3 py-1 bg-gray-500 text-white rounded text-sm hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed"
                    title="Step backward in time"
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
                    {isPlaying ? '⏸ Pause' : '▶ Play Forward in Time'}
                  </button>
                  
                  <button
                    onClick={stepForward}
                    disabled={selectedSurfaceIndex === 0}
                    className="px-3 py-1 bg-gray-500 text-white rounded text-sm hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed"
                    title="Step forward in time"
                  >
                    ⏩
                  </button>
                  
                  <button
                    onClick={goToEnd}
                    className="px-3 py-1 bg-gray-500 text-white rounded text-sm hover:bg-gray-600"
                    title="Go to newest (end of timeline)"
                  >
                    ⏭
                  </button>
                </div>

                {/* Timeline Slider - Fixed to show oldest on left, newest on right */}
                <div className="mb-3">
                  <input
                    type="range"
                    min={0}
                    max={data.length - 1}
                    value={data.length - 1 - selectedSurfaceIndex} // Flip the slider value
                    onChange={(e) => {
                      stopPlayback();
                      setSelectedSurfaceIndex(data.length - 1 - Number(e.target.value)); // Flip back when setting
                    }}
                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                  />
                  <div className="flex justify-between text-xs text-gray-500 mt-1">
                    <span>Oldest</span> {/* Left side = oldest */}
                    <span>Frame {data.length - selectedSurfaceIndex} of {data.length}</span> {/* Frame 1 = oldest */}
                    <span>Newest</span> {/* Right side = newest */}
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

        {/* Surface Data Summary - Update the history overview to be clearer */}
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
                    <p>Newest: {data.length > 0 ? new Date(data[0].timestamp).toLocaleString() : 'N/A'}</p> {/* Clarified which is which */}
                    <p>Oldest: {data.length > 0 ? new Date(data[data.length - 1].timestamp).toLocaleString() : 'N/A'}</p>
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