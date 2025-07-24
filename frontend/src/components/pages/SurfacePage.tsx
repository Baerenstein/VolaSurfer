import React, { useEffect, useState, useRef, memo } from 'react';
import Plot from 'react-plotly.js';
import VolHeatContent from '../VolHeatContent';
import { SurfaceData } from '../../types/surface';
import useSurfaceData from '../../hooks/useSurfaceData';

interface CacheItem {
  timestamp: number;
  data: SurfaceData;
}

const cache = new Map<string, CacheItem>();
const CACHE_DURATION = 30000; // 30 seconds

interface SurfacePageProps {}

interface ContainerProps {
  title: string;
  children?: React.ReactNode;
}

const Container: React.FC<ContainerProps> = ({ title, children }) => (
  <div className="flex-1 m-4 bg-black border border-gray-600 rounded-lg shadow-lg">
    <div className="p-6">
      <h2 className="text-xl font-semibold mb-4 text-white">{title}</h2>
      {children}
    </div>
  </div>
);

const SurfacePage: React.FC<SurfacePageProps> = memo(() => {
  const [interpolationMethod, setInterpolationMethod] = useState<'linear' | 'nearest'>('nearest');
  const surfaceData = useSurfaceData('http://localhost:8000/api/v1/latest-vol-surface');

  // 🔥 useRef instead of useState to avoid re-renders
  const scrollingRef = useRef(false);
  const mouseDownRef = useRef(false);
  const timeoutRef = useRef<number | null>(null);

  // Optimized Mouse Down & Up Handlers (No re-renders)
  const handleMouseDown = () => {
    mouseDownRef.current = true;
  };

  const handleMouseUp = () => {
    mouseDownRef.current = false;
  };

  // Optimized Scroll Handler (Throttled)
  const handleScroll = () => {
    scrollingRef.current = true; // Flag as scrolling (no re-render)

    if (timeoutRef.current) clearTimeout(timeoutRef.current);
    timeoutRef.current = window.setTimeout(() => {
      scrollingRef.current = false; // Reset after 300ms
    }, 300);
  };

  const cameraRef = useRef({
    eye: { x: 2, y: 2, z: 1.5 },
    center: { x: 0, y: 0, z: -0.1 },
    up: { x: 0, y: 0, z: 1 },
  });

  const stableRevisionId = useRef(`plot-${Date.now()}`);

  const layout = {
    title: {
      text: "Volatility Surface",
      font: { size: 16, color: 'white' }
    },
    paper_bgcolor: 'black',
    plot_bgcolor: 'black',
    scene: {
      bgcolor: 'black',
      xaxis: { 
        title: "Moneyness", 
        tickformat: ".2f", 
        autorange: "reversed",
        titlefont: { size: 14, color: 'white' },
        tickfont: { size: 12, color: 'white' },
        gridcolor: '#444444',
        zerolinecolor: '#666666'
      },
      yaxis: { 
        title: "Days to Expiry", 
        tickformat: ".0f", 
        autorange: "reversed",
        titlefont: { size: 14, color: 'white' },
        tickfont: { size: 12, color: 'white' },
        gridcolor: '#444444',
        zerolinecolor: '#666666'
      },
      zaxis: { 
        title: "Implied Volatility", 
        tickformat: ".2%",
        titlefont: { size: 14, color: 'white' },
        tickfont: { size: 12, color: 'white' },
        gridcolor: '#444444',
        zerolinecolor: '#666666'
      },
      camera: cameraRef.current,
      aspectratio: { x: 1.5, y: 1.5, z: 1 },
    },
    margin: { l: 0, r: 0, b: 0, t: 40 },
    uirevision: stableRevisionId.current,
  };

  // Update the plot data (remove the smoothing property since we're using server-side interpolation)
  const plotData = [{
    type: 'surface' as const,
    x: surfaceData.data?.moneyness || [],
    y: surfaceData.data?.daysToExpiry || [],
    z: surfaceData.data?.impliedVols?.map((row: any) => 
      Array.isArray(row) ? row.map((vol: any) => vol / 100) : vol / 100
    ) || [],
    showscale: true,
    colorscale: 'Viridis',
    contours: {
      z: {
        show: true,
        usecolormap: true,
        highlightcolor: "#42f462",
        project: { z: true },
      },
    },
    opacity: 1,
    hoverongaps: false,
    hoverlabel: {
      bgcolor: "#FFF",
      font: { color: "#000" },
    },
    hoverinfo: 'x+y+z',
  }];

  return (
    <div className="flex flex-col w-full min-h-screen bg-black">
      <div className="flex flex-1 p-4">
        <Container title="VolaSurfer">
          {surfaceData.isLoading && <div className="text-white">Loading...</div>}
          {surfaceData.error && <div className="text-red-400">{surfaceData.error}</div>}
          <div 
            className="bg-black border border-gray-600 rounded-lg shadow-lg overflow-hidden"
            onWheel={handleScroll} // 🔥 Detect scroll without re-rendering
            onTouchMove={handleScroll} // 🔥 Mobile scroll detection
            onMouseDown={handleMouseDown} // 🔥 Detect mouse press
            onMouseUp={handleMouseUp} // 🔥 Detect mouse release
          >
            <div className="px-6 py-4 border-b border-gray-600">
              <div className="flex justify-between items-center">
                <h3 className="text-lg font-semibold text-white">Surface</h3>
                <div className="flex items-center space-x-2">
                  <label htmlFor="interpolation" className="text-sm text-gray-300">
                    Interpolation:
                  </label>
                  <select
                    id="interpolation"
                    value={interpolationMethod}
                    onChange={(e) => setInterpolationMethod(e.target.value as 'linear' | 'nearest')}
                    className="px-3 py-1 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="nearest">Nearest</option>
                    <option value="linear">Linear</option>
                  </select>
                </div>
              </div>
            </div>
            <div className="p-4">
              <Plot 
                data={plotData}
                layout={layout}
                style={{
                  width: "100%",
                  height: "50vh",
                }}
                config={{
                  responsive: true,
                  displayModeBar: true,
                  modeBarButtonsToRemove: ["toImage", "sendDataToCloud"],
                }}
              />
            </div>
          </div>
        </Container>
        <Container title="VolaHeat">
          <VolHeatContent {...surfaceData} />
        </Container>
      </div>
    </div>
  );
});

export default SurfacePage;
