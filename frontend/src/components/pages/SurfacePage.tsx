import React, { useEffect, useState, useRef, useCallback, memo } from 'react';
import Plot from 'react-plotly.js';
import { useSurfaceData } from '../../hooks/useSurfaceData';
import VolaHeatContent from '../VolHeatContent';

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
}, (prevProps, nextProps) => {
  return !nextProps.shouldRender; // Prevent unnecessary updates
});

const SurfacePage: React.FC = () => {
  const [interpolationMethod, setInterpolationMethod] = useState<'linear' | 'nearest'>('nearest');
  const surfaceData = useSurfaceData(interpolationMethod);
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
    title: "Volatility Surface",
    scene: {
      xaxis: { title: "Moneyness", tickformat: ".2f", autorange: "reversed" },
      yaxis: { title: "Days to Expiry", tickformat: ".0f", autorange: "reversed" },
      zaxis: { title: "Implied Volatility", tickformat: ".2%" },
      camera: cameraRef.current,
      aspectratio: { x: 1.5, y: 1.5, z: 1 },
    },
    margin: { l: 0, r: 0, b: 0, t: 0 },
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
    x: surfaceData.data?.moneyness || [],
    y: surfaceData.data?.daysToExpiry || [],
    z: surfaceData.data?.impliedVols?.map(row => row.map(vol => vol / 100)) || [],
    showscale: true,
    colorscale: "Viridis",
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
    <div className="flex flex-col w-full min-h-screen bg-gray-100">
      <div className="flex flex-1 p-4">
        <Container title="VolaSurfer">
          {surfaceData.isLoading && <div>Loading...</div>}
          {surfaceData.error && <div className="text-red-500">{surfaceData.error}</div>}
          <div 
            className="bg-white rounded-lg shadow-lg overflow-hidden"
            onWheel={handleScroll} // 🔥 Detect scroll without re-rendering
            onTouchMove={handleScroll} // 🔥 Mobile scroll detection
            onMouseDown={handleMouseDown} // 🔥 Detect mouse press
            onMouseUp={handleMouseUp} // 🔥 Detect mouse release
          >
            <div className="px-6 py-4 border-b border-gray-200">
              <div className="flex justify-between items-center">
                <h3 className="text-lg font-semibold text-gray-800">Surface</h3>
                <div className="flex items-center space-x-2">
                  <label htmlFor="interpolation" className="text-sm text-gray-600">
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
              <MemoizedPlot 
                data={plotData}
                layout={layout}
                shouldRender={shouldRender}
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
