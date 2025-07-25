import React, { useEffect, useState, useRef, useCallback, memo } from 'react';
import Plot from 'react-plotly.js';
import { useSurfaceData } from '../../hooks/useSurfaceData';
import VolaHeatContent from '../VolHeatContent';

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

// Memoized Plot Component
const MemoizedPlot = memo(({ data, layout }: { data: any; layout: any }) => {
  return (
    <div className="overflow-hidden">
      <Plot
        data={data}
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
  );
});

const SurfacePage: React.FC = () => {
  const [interpolationMethod, setInterpolationMethod] = useState<'linear' | 'nearest'>('nearest');
  const [minMoneyness, setMinMoneyness] = useState(0.8);
  const [maxMoneyness, setMaxMoneyness] = useState(1.3);
  const [minMaturity, setMinMaturity] = useState(0);
  const [maxMaturity, setMaxMaturity] = useState(30);
  const surfaceData = useSurfaceData(interpolationMethod, minMoneyness, maxMoneyness, minMaturity, maxMaturity);
  const [shouldRender, setShouldRender] = useState(false);

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

  const [layout, setLayout] = useState({
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
  });

  // 🔥 Only re-render when both scrolling & mouseDown are inactive
  useEffect(() => {
    if (surfaceData.data === null || mouseDownRef.current || scrollingRef.current) {
      setShouldRender(false);
    } else {
      setShouldRender(true);
    }
  }, [surfaceData.data]);

  // Update the plot data (remove the smoothing property since we're using server-side interpolation)
  const plotData = [{
    type: 'surface',
    x: (surfaceData.data as any)?.moneyness || [],
    y: (surfaceData.data as any)?.daysToExpiry || [],
    z: (surfaceData.data as any)?.impliedVols?.map((row: any) => row.map((vol: any) => vol / 100)) || [],
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
                  <label htmlFor="min-moneyness" className="text-sm text-gray-300 ml-4">Min M:</label>
                  <input
                    id="min-moneyness"
                    type="number"
                    step="0.01"
                    min={0}
                    max={maxMoneyness}
                    value={minMoneyness}
                    onChange={e => setMinMoneyness(Number(e.target.value))}
                    className="w-16 px-2 py-1 border border-gray-300 rounded-md text-sm focus:outline-none"
                  />
                  <label htmlFor="max-moneyness" className="text-sm text-gray-300 ml-2">Max M:</label>
                  <input
                    id="max-moneyness"
                    type="number"
                    step="0.01"
                    min={minMoneyness}
                    max={2}
                    value={maxMoneyness}
                    onChange={e => setMaxMoneyness(Number(e.target.value))}
                    className="w-16 px-2 py-1 border border-gray-300 rounded-md text-sm focus:outline-none"
                  />
                  <label htmlFor="min-maturity" className="text-sm text-gray-300 ml-4">Min DTE:</label>
                  <input
                    id="min-maturity"
                    type="number"
                    step="1"
                    min={0}
                    max={maxMaturity}
                    value={minMaturity}
                    onChange={e => setMinMaturity(Number(e.target.value))}
                    className="w-16 px-2 py-1 border border-gray-300 rounded-md text-sm focus:outline-none"
                  />
                  <label htmlFor="max-maturity" className="text-sm text-gray-300 ml-2">Max DTE:</label>
                  <input
                    id="max-maturity"
                    type="number"
                    step="1"
                    min={minMaturity}
                    max={365}
                    value={maxMaturity}
                    onChange={e => setMaxMaturity(Number(e.target.value))}
                    className="w-16 px-2 py-1 border border-gray-300 rounded-md text-sm focus:outline-none"
                  />
                </div>
              </div>
            </div>
            <div className="p-4">
              <MemoizedPlot 
                data={plotData}
                layout={layout}
              />
            </div>
          </div>
        </Container>
        <Container title="VolaHeat">
          <VolaHeatContent {...surfaceData} />
        </Container>
      </div>
    </div>
  );
};

export default SurfacePage;
