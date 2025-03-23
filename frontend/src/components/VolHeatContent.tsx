import React, { useEffect, useState } from 'react';
import Plot from 'react-plotly.js';
import { useSurfaceData } from '../hooks/useSurfaceData';


const VolaHeatContent: React.FC<ReturnType<typeof useSurfaceData>> = ({ 
  data, 
  isLoading, 
  error 
}) => {

  const [layout, setLayout] = useState({
    title: "Volatility Heatmap",
    xaxis: { title: "Moneyness", tickformat: ".2f" },
    yaxis: { title: "Days to Expiry", tickformat: ".0f", side: "top" },
    margin: { l: 60, r: 30, t: 60, b: 30 },
  });

  useEffect(() => {
    setLayout((prevLayout) => ({
      ...prevLayout,
    }));
  }, [data]);


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
                colorscale: 'Viridis', // change colorscale to something else, including read, maybe have universal color scalex
                colorbar: {
                  title: 'IV',
                },
                uirevision: 'true',
                hoverongaps: false,
                hoverlabel: {
                  bgcolor: "#FFF",
                  font: { color: "#000" }
                },
              }
            ]}
            layout={layout} // Use stored layout to preserve zoom/pan
            // onRelayout={(newLayout) => {
            //   setLayout((prevLayout) => ({
            //     ...prevLayout,
            //     xaxis: newLayout["xaxis.range"]
            //       ? { ...prevLayout.xaxis, range: newLayout["xaxis.range"] }
            //       : prevLayout.xaxis,
            //     yaxis: newLayout["yaxis.range"]
            //       ? { ...prevLayout.yaxis, range: newLayout["yaxis.range"] }
            //       : prevLayout.yaxis,
            //   }));
            // }}
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

export default VolaHeatContent;