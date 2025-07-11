'use client';

import dynamic from 'next/dynamic';

const EconomicCalendarWidget = dynamic(
  () => import('@/components/EconomicCalendarWidget'),
  {
    ssr: false,
    loading: () => <p className="text-center text-gray-400">Loading Calendar...</p>,
  }
);

export default function CalendarView() {
  return <EconomicCalendarWidget />;
} 