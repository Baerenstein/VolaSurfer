import React from 'react';
import Plot from 'react-plotly.js';

interface SurfaceProps {
  moneyness: number[];
  daysToExpiry: number[];
  impliedVols: number[][];
}

const Surface: React.FC<SurfaceProps> = ({ moneyness, daysToExpiry, impliedVols }) => {
  // Common styles for better appearance
  const commonLayout = {
    autosize: true,
    font: {
      family: 'Inter, sans-serif',
      size: 12,
    },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
  };

  return (
    <div className="grid grid-cols-1 gap-6 p-4">
      {/* Surface Plot Card */}
      <div className="bg-white rounded-lg shadow-lg overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-800">
            Volatility Surface
          </h3>
        </div>
        <div className="p-4">
          <Plot
            data={[
              {
                type: 'surface',
                x: moneyness,
                y: daysToExpiry,
                z: impliedVols,
                showscale: false,
                colorscale: 'Viridis',
                contours: {
                  z: {
                    show: true,
                    usecolormap: true,
                    highlightcolor: "#42f462",
                    project: { z: true }
                  }
                },
                opacity: 0.9,
                hoverongaps: false,
                hoverlabel: {
                  bgcolor: "#FFF",
                  font: { color: "#000" }
                },
              }
            ]}
            layout={{
              ...commonLayout,
              scene: {
                xaxis: { 
                  title: 'Moneyness',
                  tickformat: '.2f', //make this a percentage
                },
                yaxis: { 
                  title: 'Days to Expiry',
                  tickformat: '.0f',
                },
                zaxis: { 
                  title: 'Implied Volatility',
                  tickformat: '.2%', // double check precision
                },
                camera: {
                  eye: { x: 2, y: 2, z: 1.5 },
                  center: { x: 0, y: 0, z: -0.1 }
                },
                aspectratio: { x: 1.5, y: 1.5, z: 1 }
              },
              margin: { l: 0, r: 0, b: 0, t: 0 }
            }}
            style={{
              width: '100%',
              height: '50vh',
            }}
            config={{
              responsive: true,
              displayModeBar: false,
              modeBarButtonsToRemove: ['toImage', 'sendDataToCloud'],
            }}
          />
        </div>
      </div>

      {/* Heatmap Card */}
      <div className="bg-white rounded-lg shadow-lg overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-800">
            Surface Heatmap
          </h3>
        </div>
        <div className="p-4">
          <Plot
            data={[
              {
                type: 'heatmap',
                x: daysToExpiry,
                y: moneyness,
                z: impliedVols,
                showscale: false,
                colorscale: 'Viridis',
                colorbar: {
                  title: 'IV',
                  tickformat: '.2%',
                },
                hoverongaps: false,
                hoverlabel: {
                  bgcolor: "#FFF",
                  font: { color: "#000" }
                },
              }
            ]}
            layout={{
              ...commonLayout,
              xaxis: { 
                title: 'Days to Expiry',
                tickformat: '.0f',
                side: 'top',
              },
              yaxis: { 
                title: 'Moneyness',
                tickformat: '.2f',
                autorange: 'reversed',
              },
              margin: { l: 60, r: 30, t: 60, b: 30 }  // Adjusted margins for labels
            }}
            style={{
              width: '100%',
              height: '50vh',
            }}
            config={{
              responsive: true,
              displayModeBar: false,
              modeBarButtonsToRemove: ['toImage', 'sendDataToCloud'],
            }}
          />
        </div>
      </div>
    </div>
  );
};

export default Surface;