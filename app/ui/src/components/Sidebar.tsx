'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { MdDashboard, MdShowChart, MdAutoGraph, MdHistory, MdSettings, MdCalendarToday, MdBuild, MdExpandMore, MdExpandLess } from 'react-icons/md';
import { FaProjectDiagram, FaBalanceScale, FaChartPie, FaExclamationTriangle } from 'react-icons/fa';

const navItems = [
  { name: 'Dashboard', href: '/', icon: <MdDashboard size={22} /> },
  { name: 'Charts', href: '/charts', icon: <MdShowChart size={22} /> },
  { name: 'Technicals', href: '/technicals', icon: <MdAutoGraph size={22} /> },
  { name: 'Calendar', href: '/calendar', icon: <MdCalendarToday size={22} /> },
  { name: 'Strategies', href: '/strategies', icon: <MdAutoGraph size={22} /> },
  { name: 'Backtesting', href: '/backtesting', icon: <MdHistory size={22} /> },
  // Trading Tools will be handled separately
  { name: 'Settings', href: '/settings', icon: <MdSettings size={22} /> }
];

const tradingTools = [
  { name: 'Correlation', href: '/tools/correlation', icon: <FaProjectDiagram size={18} /> },
  { name: 'Currency Index', href: '/tools/currency-index', icon: <FaBalanceScale size={18} /> },
  { name: 'Position Sizing', href: '/tools/position-sizing', icon: <FaChartPie size={18} /> },
  { name: 'Value At Risk', href: '/tools/value-at-risk', icon: <FaExclamationTriangle size={18} /> },
];

const Sidebar = () => {
  const [toolsOpen, setToolsOpen] = useState(false);
  const pathname = usePathname();

  return (
    <aside className="h-screen w-56 bg-[#181A20] text-white flex flex-col p-4 shadow-lg">
      <div className="text-2xl font-bold mb-8 text-blue-400">HaruPyQuant</div>
      <nav className="flex-1">
        <ul className="space-y-2">
          {navItems.map((item) => {
            const isActive = pathname === item.href;
            return (
              <li key={item.name}>
                <Link 
                  href={item.href} 
                  className={`flex items-center px-3 py-2 rounded transition-colors ${
                    isActive 
                      ? 'bg-blue-600 text-white shadow-lg' 
                      : 'hover:bg-[#23272F] text-gray-300'
                  }`}
                >
                  <span className={`mr-3 ${isActive ? 'text-white' : 'text-blue-300'}`}>
                    {item.icon}
                  </span>
                  <span>{item.name}</span>
                </Link>
              </li>
            );
          })}
          {/* Trading Tools with submenu */}
          <li>
            <button
              className={`flex items-center w-full px-3 py-2 rounded transition-colors focus:outline-none ${
                tradingTools.some(tool => pathname === tool.href)
                  ? 'bg-blue-600 text-white shadow-lg'
                  : 'hover:bg-[#23272F] text-gray-300'
              }`}
              onClick={() => setToolsOpen((open) => !open)}
              aria-expanded={toolsOpen}
            >
              <span className={`mr-3 ${tradingTools.some(tool => pathname === tool.href) ? 'text-white' : 'text-blue-300'}`}>
                <MdBuild size={22} />
              </span>
              <span className="flex-1 text-left">Trading Tools</span>
              <span>{toolsOpen ? <MdExpandLess size={20} /> : <MdExpandMore size={20} />}</span>
            </button>
            {toolsOpen && (
              <ul className="ml-8 mt-1 space-y-1">
                {tradingTools.map((tool) => {
                  const isActive = pathname === tool.href;
                  return (
                    <li key={tool.name}>
                      <Link 
                        href={tool.href} 
                        className={`flex items-center px-2 py-1 rounded transition-colors text-sm ${
                          isActive 
                            ? 'bg-blue-600 text-white shadow-lg' 
                            : 'hover:bg-[#23272F] text-gray-300'
                        }`}
                      >
                        <span className={`mr-2 ${isActive ? 'text-white' : 'text-blue-200'}`}>
                          {tool.icon}
                        </span>
                        <span>{tool.name}</span>
                      </Link>
                    </li>
                  );
                })}
              </ul>
            )}
          </li>
        </ul>
      </nav>
      <div className="mt-8 text-xs text-gray-500">System Status<br /><span className="inline-block mt-1 px-2 py-0.5 bg-green-600 rounded-full text-white">Connected</span></div>
    </aside>
  );
};

export default Sidebar; 