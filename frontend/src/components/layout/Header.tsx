import React from 'react';

interface HeaderProps {
  onPageChange: (page: string) => void;
  currentPage: string;
}

const Header: React.FC<HeaderProps> = ({ onPageChange, currentPage }) => {
  return (
    <header className="fixed top-0 left-0 w-full bg-gray-800 text-white shadow-md z-50">
      <div className="container mx-auto flex justify-between items-center p-4">
        <div className="text-2xl font-bold">VolaSurfer</div>
        <nav className="space-x-4">
          <button
            onClick={() => onPageChange('landing')}
            className={`px-3 py-2 rounded ${currentPage === 'landing' ? 'bg-gray-700' : 'hover:bg-gray-700'}`}
          >
            Home
          </button>
          <button
            onClick={() => onPageChange('surface')}
            className={`px-3 py-2 rounded ${currentPage === 'surface' ? 'bg-gray-700' : 'hover:bg-gray-700'}`}
          >
            Latest Surface
          </button>
          <button
            onClick={() => onPageChange('history')}
            className={`px-3 py-2 rounded ${currentPage === 'history' ? 'bg-gray-700' : 'hover:bg-gray-700'}`}
          >
            History
          </button>
        </nav>
      </div>
    </header>
  );
};

export default Header;