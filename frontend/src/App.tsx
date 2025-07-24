import React, { useState } from 'react';
import Header from './components/layout/Header';
import SurfacePage from './components/pages/SurfacePage';
import HistoryPage from './components/pages/HistoryPage';
import CalibrationPage from './components/pages/CalibrationPage';

const App: React.FC = () => {
  const [currentPage, setCurrentPage] = useState('landing');

  const renderLandingPage = () => (
    <div className="flex flex-col items-center justify-center min-h-screen w-full bg-gray-900">
      <h1 className="text-4xl font-bold mb-8 text-white">VolaSurfer Dashboard</h1>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 w-full max-w-4xl px-4">
        <button
          onClick={() => setCurrentPage('surface')}
          className="px-8 py-6 bg-white rounded-lg shadow-lg hover:shadow-xl transition-shadow"
        >
          <h2 className="text-xl font-semibold mb-2">Latest Surface</h2>
          <p className="text-gray-600">View the latest volatility surface</p>
        </button>
        <button
          onClick={() => setCurrentPage('history')}
          className="px-8 py-6 bg-white rounded-lg shadow-lg hover:shadow-xl transition-shadow"
        >
          <h2 className="text-xl font-semibold mb-2">Surface History</h2>
          <p className="text-gray-600">Browse historical volatility surfaces</p>
        </button>
        <button
          onClick={() => setCurrentPage('calibration')}
          className="px-8 py-6 bg-blue-50 border-2 border-blue-200 rounded-lg shadow-lg hover:shadow-xl transition-shadow"
        >
          <h2 className="text-xl font-semibold mb-2 text-blue-800">Surface Calibration</h2>
          <p className="text-blue-600">Calibrate surfaces and analyze performance</p>
        </button>
      </div>
    </div>
  );

  return (
    <div className="w-full h-screen">
      <Header onPageChange={setCurrentPage} currentPage={currentPage} />
      <div className="pt-16">
        {currentPage === 'landing' && renderLandingPage()}
        {currentPage === 'surface' && <SurfacePage />}
        {currentPage === 'history' && <HistoryPage />}
        {currentPage === 'calibration' && <CalibrationPage />}
      </div>
    </div>
  );
};

export default App;