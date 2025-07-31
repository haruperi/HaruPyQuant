'use client';

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import PlotlyNoSSR from '@/components/PlotlyNoSSR';

const timeframes = ['D1', 'H4', 'H1', 'M30', 'M15', 'M5', 'M1'];

interface CorrelationData {
  z: (number | null)[][];
  x: string[];
  y: string[];
}

export default function CorrelationPage() {
  const [date, setDate] = useState(new Date().toISOString().split('T')[0]);
  const [timeframe, setTimeframe] = useState('D1');
  const [correlationData, setCorrelationData] = useState<CorrelationData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const fetchCorrelationData = async () => {
    setLoading(true);
    setError('');
    try {
      const response = await axios.get('/api/correlation-matrix', {
        params: { date, timeframe },
      });
      setCorrelationData(response.data);
    } catch (err) {
      setError('Failed to fetch correlation data. Please check the server logs.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchCorrelationData();
  }, []);

  const handleFetchClick = () => {
    fetchCorrelationData();
  };

  return (
    <div className="p-6 bg-[#181A20] min-h-screen text-white">
      <h1 className="text-3xl font-bold mb-6">Correlation Matrix</h1>
      
      <div className="bg-[#23272F] rounded-lg p-6 mb-6 flex items-center space-x-4">
        <div>
          <label htmlFor="date-picker" className="block text-sm font-medium text-gray-300 mb-1">
            Date
          </label>
          <input
            type="date"
            id="date-picker"
            value={date}
            onChange={(e) => setDate(e.target.value)}
            className="bg-[#181A20] border border-gray-600 rounded-md px-3 py-2 text-white focus:ring-blue-500 focus:border-blue-500"
          />
        </div>
        <div>
          <label htmlFor="timeframe-select" className="block text-sm font-medium text-gray-300 mb-1">
            Timeframe
          </label>
          <select
            id="timeframe-select"
            value={timeframe}
            onChange={(e) => setTimeframe(e.target.value)}
            className="bg-[#181A20] border border-gray-600 rounded-md px-3 py-2 text-white focus:ring-blue-500 focus:border-blue-500"
          >
            {timeframes.map((tf) => (
              <option key={tf} value={tf}>
                {tf}
              </option>
            ))}
          </select>
        </div>
        <button
          onClick={handleFetchClick}
          disabled={loading}
          className="self-end px-4 py-2 bg-blue-600 rounded-md hover:bg-blue-700 disabled:bg-gray-500 transition-colors"
        >
          {loading ? 'Loading...' : 'Get Matrix'}
        </button>
      </div>

      {error && <div className="bg-red-500/20 border border-red-500 text-red-300 rounded-lg p-4 mb-6">{error}</div>}

      <div className="bg-[#23272F] rounded-lg p-2">
        {correlationData ? (
          <PlotlyNoSSR
            data={[
              {
                z: correlationData.z,
                x: correlationData.x,
                y: correlationData.y,
                type: 'heatmap',
                text: correlationData.z,
                texttemplate: '%{text}',
                hoverinfo: 'none',
                colorscale: [
                  [0.0, 'rgb(165,0,38)'],
                  [0.45, 'rgb(249, 249, 249)'],
                  [0.5, 'rgb(255, 255, 255)'],
                  [0.55, 'rgb(249, 249, 249)'],
                  [1.0, 'rgb(0,104,55)'],
                ],
                zmin: -100,
                zmax: 100,
                showscale: false,
                transpose: true,
              },
            ]}
            layout={{
              title: `Correlation Matrix for ${date} (${timeframe})`,
              paper_bgcolor: '#23272F',
              plot_bgcolor: '#23272F',
              font: {
                color: 'white',
              },
              xaxis: {
                side: 'top',
                tickangle: -45,
              },
              yaxis: {
                autorange: 'reversed',
              },
              autosize: true,
              margin: { l: 100, r: 50, b: 100, t: 100, pad: 4 },
            }}
            config={{ responsive: true }}
            style={{ width: '100%', height: '80vh' }}
          />
        ) : (
          <div className="flex justify-center items-center h-96">
            {loading ? 'Generating matrix...' : 'No data to display.'}
          </div>
        )}
      </div>
    </div>
  );
} 