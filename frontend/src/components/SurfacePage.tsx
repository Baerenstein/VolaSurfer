import React, { useEffect } from 'react';
import Plot from 'react-plotly.js';
import { useSurfaceData } from '../hooks/useSurfaceData';

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

const VolaHeatContent: React.FC<ReturnType<typeof useSurfaceData>> = ({ 
  data, 
  isLoading, 
  error 
}) => {
  if (isLoading) {
    return (
      <div className="h-96 flex items-center justify-center text-gray-500">
        Loading surface data...
      </div>
    );
  }

  if (error) {
    return (
      <div className="h-96 flex items-center justify-center text-red-500">
        {error}
      </div>
    );
  }

  if (!data) {
    return (
      <div className="h-96 flex items-center justify-center text-gray-500">
        No data available
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-lg overflow-hidden">
      <div className="px-6 py-4 border-b border-gray-200">
        <h3 className="text-lg font-semibold text-gray-800">Volatility Data Summary:</h3>
      </div>
      <div className="p-4 space-y-4">
        <div className="text-sm text-gray-500">
          Last Updated: {new Date(data.timestamp).toLocaleString()}
          <br />
          Method: {data.interpolationMethod}
        </div>

        <div>
          <h3 className="font-medium text-gray-700">Surface Data Summary:</h3>
          <div className="mt-2 space-y-2 text-sm text-gray-600">
            <p>Number of Strikes: {data.moneyness.length}</p>
            <p>Number of Expiries: {data.daysToExpiry.length}</p>
            <p>Moneyness Range: {Math.min(...data.moneyness).toFixed(3)} - {Math.max(...data.moneyness).toFixed(3)}</p>
            <p>DTE Range: {Math.min(...data.daysToExpiry).toFixed(1)} - {Math.max(...data.daysToExpiry).toFixed(1)} days</p>
          </div>
        </div>

        <div className="p-4">
          <Plot
            data={[
              {
                type: 'heatmap',
                x: data.moneyness,
                y: data.daysToExpiry,
                z: data.impliedVols,
                showscale: true,
                colorscale: 'Viridis',
                colorbar: {
                  title: 'IV',
                },
                hoverongaps: false,
                hoverlabel: {
                  bgcolor: "#FFF",
                  font: { color: "#000" }
                },
              }
            ]}
            layout={{
              title: 'Volatility Heatmap',
              xaxis: { 
                title: 'Moneyness',
                tickformat: '.2f',
              },
              yaxis: { 
                title: 'Days to Expiry',
                tickformat: '.0f',
                side: 'top',
              },
              margin: { l: 60, r: 30, t: 60, b: 30 }
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

const SurfacePage: React.FC = () => {
  const surfaceData = useSurfaceData();

  useEffect(() => {
    surfaceData.refetch();
  }, [surfaceData.refetch]);

  return (
    <div className="flex flex-col w-full min-h-screen bg-gray-100">
      <div className="flex flex-1 p-4">
        <Container title="VolaSurfer">
          {surfaceData.isLoading && <div>Loading...</div>}
          {surfaceData.error && <div className="text-red-500">{surfaceData.error}</div>}
          {surfaceData.data && (
            <div className="bg-white rounded-lg shadow-lg overflow-hidden">
              <div className="px-6 py-4 border-b border-gray-200">
                <h3 className="text-lg font-semibold text-gray-800">Surface</h3>
              </div>
              <div className="p-4">
                <Plot
                  data={[
                    {
                      type: 'surface',
                      x: surfaceData.data.moneyness,
                      y: surfaceData.data.daysToExpiry,
                      z: surfaceData.data.impliedVols,
                      showscale: true,
                      colorscale: 'Viridis',
                      contours: {
                        z: {
                          show: true,
                          usecolormap: true,
                          highlightcolor: "#42f462",
                          project: { z: true }
                        }
                      },
                      opacity: 1,
                      hoverongaps: false,
                      hoverlabel: {
                        bgcolor: "#FFF",
                        font: { color: "#000" }
                      },
                    }
                  ]}
                  layout={{
                    title: 'Volatility Surface',
                    scene: {
                      xaxis: { 
                        title: 'Moneyness',
                        tickformat: '.2f',
                        autorange: 'reversed',
                      },
                      yaxis: { 
                        title: 'Days to Expiry',
                        tickformat: '.0f',
                        autorange: 'reversed',
                      },
                      zaxis: { 
                        title: 'Implied Volatility',
                        tickformat: '.2%',
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
          )}
        </Container>
        
        <Container title="VolaHeat">
          <VolaHeatContent {...surfaceData} />
        </Container>
        
        <button 
          onClick={surfaceData.refetch} 
          className="mt-4 px-2 py-1 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition text-sm"
        >
          Refresh Data
        </button>
      </div>
    </div>
  );
};

export default SurfacePage;