'use client';

import React, { useEffect, useState } from 'react';
import DashboardCard from '../components/DashboardCard';

interface DashboardMetrics {
  closed_positions: number;
  winning_factor: string;
  performance_factor: number;
  max_drawdown: number;
  open_positions: number;
  pnl: number;
  equity: number;
  balance: number;
}

export default function Home() {
  const [metrics, setMetrics] = useState<DashboardMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    fetch('http://127.0.0.1:8001/api/dashboard-metrics')
      .then((res) => {
        if (!res.ok) throw new Error('Failed to fetch metrics');
        return res.json();
      })
      .then((data) => {
        setMetrics(data);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  return (
    <div className="min-h-screen p-8 pb-20 bg-[#181A20]">
      <h1 className="text-2xl font-bold text-white mb-8 text-center">Live Performance Metrics</h1>
      {loading ? (
        <div className="text-center text-gray-400 text-lg">Loading metrics...</div>
      ) : error ? (
        <div className="text-center text-red-400 text-lg">{error}</div>
      ) : metrics ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 max-w-6xl mx-auto">
          <DashboardCard title="Closed Positions" value={metrics.closed_positions} subtitle="Number Of Closed Trades" />
          <DashboardCard title="Winning Factor" value={<span className="text-green-400">{metrics.winning_factor}</span>} subtitle="Winning Factor Ratio" color="green" />
          <DashboardCard title="Performance Factor" value={<span className="text-red-400">{metrics.performance_factor}</span>} subtitle="Performance Factor Of Historical Trades" color="red" />
          <DashboardCard title="Max Drawdown" value={<span className="text-orange-400">{metrics.max_drawdown}%</span>} subtitle="Maximum Drawdown Of Historical Trades" color="orange" />
          <DashboardCard title="Open Positions" value={metrics.open_positions} subtitle="Number Of Active Trades" />
          <DashboardCard title="P&L" value={<span className="text-red-400">$ {metrics.pnl}</span>} subtitle="Floating Profit & Loss" color="red" />
          <DashboardCard title="Equity" value={<span className="text-red-400">$ {metrics.equity}</span>} subtitle="Floating Equity" color="red" />
          <DashboardCard title="Balance" value={<span className="text-red-400">$ {metrics.balance}</span>} subtitle="Closed Balance" color="red" />
        </div>
      ) : null}
    </div>
  );
}
