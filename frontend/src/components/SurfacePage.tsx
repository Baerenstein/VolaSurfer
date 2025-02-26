import React, { useEffect, useState, useRef, useCallback, memo } from 'react';
import Plot from 'react-plotly.js';
import { useSurfaceData } from '../hooks/useSurfaceData';
import VolaHeatContent from './VolHeatContent';

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

// Memoized Plot Component (Only updates when shouldRender is true)
const MemoizedPlot = memo(({ data, layout, onScroll }: { data: any; layout: any; onScroll: () => void }) => {
  return (
    <div 
      onWheel={onScroll} // Detects scrolling inside the plot container
      onTouchMove={onScroll} // Captures scrolling on mobile (touch-based)
      className="overflow-hidden"
    >
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
  return !nextProps.shouldRender; // Prevent re-renders unless shouldRender is true
});

const SurfacePage: React.FC = () => {
  const surfaceData = useSurfaceData();
  const [shouldRender, setShouldRender] = useState(false);
  const [mouseDown, setMouseDown] = useState(false);
  const [scrolling, setScrolling] = useState(false);

  const scrollTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Event handlers for mouse interactions
  const handleMouseDown = useCallback(() => setMouseDown(true), []);
  const handleMouseUp = useCallback(() => setMouseDown(false), []);

  // Detect scrolling inside the Plotly chart
  const handleScroll = () => {
    setScrolling(true);
    if (scrollTimeout.current) clearTimeout(scrollTimeout.current);
    scrollTimeout.current = setTimeout(() => setScrolling(false), 1000); // Reset after 300ms
  };

  useEffect(() => {
    document.addEventListener('mousedown', handleMouseDown);
    document.addEventListener('mouseup', handleMouseUp);
    return () => {
      document.removeEventListener('mousedown', handleMouseDown);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, []);

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

  useEffect(() => {
    if (surfaceData.data === null || mouseDown || scrolling) {
      setShouldRender(false);
    } else {
      setShouldRender(true);
    }
  }, [surfaceData.data, mouseDown, scrolling]);

  return (
    <div className="flex flex-col w-full min-h-screen bg-gray-100">
      <div className="flex flex-1 p-4">
        <Container title="VolaSurfer">
          {surfaceData.isLoading && <div>Loading...</div>}
          {surfaceData.error && <div className="text-red-500">{surfaceData.error}</div>}
          {(
            <div className="bg-white rounded-lg shadow-lg overflow-hidden">
              <div className="px-6 py-4 border-b border-gray-200">
                <h3 className="text-lg font-semibold text-gray-800">Surface</h3>
              </div>
              <div className="p-4">
                <MemoizedPlot 
                  data={[
                    {
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
                    },
                  ]}
                  layout={layout}
                  shouldRender={shouldRender}
                  onScroll={handleScroll} // Pass scroll handler
                />
              </div>
            </div>
          )}
        </Container>
        <Container title="VolaHeat">
          <VolaHeatContent {...surfaceData} />
        </Container>
      </div>
    </div>
  );
};

export default SurfacePage;
