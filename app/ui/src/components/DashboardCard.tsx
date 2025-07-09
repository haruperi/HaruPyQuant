import React from 'react';

interface DashboardCardProps {
  title: string;
  value: React.ReactNode;
  subtitle?: string;
  color?: 'default' | 'green' | 'red' | 'orange';
}

const colorMap = {
  default: 'text-blue-300',
  green: 'text-green-400',
  red: 'text-red-400',
  orange: 'text-orange-400',
};

const DashboardCard: React.FC<DashboardCardProps> = ({ title, value, subtitle, color = 'default' }) => (
  <div className="bg-[#23272F] rounded-xl p-6 min-w-[200px] min-h-[120px] flex flex-col justify-between shadow-md border border-[#23272F] hover:border-blue-500 transition-all">
    <div>
      <div className="text-lg font-semibold text-white mb-1">{title}</div>
      {subtitle && <div className="text-xs text-gray-400 mb-2">{subtitle}</div>}
    </div>
    <div className={`text-3xl font-bold mt-2 ${colorMap[color]}`}>{value}</div>
  </div>
);

export default DashboardCard; 