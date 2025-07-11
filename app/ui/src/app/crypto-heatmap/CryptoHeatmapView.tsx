'use client';

import dynamic from 'next/dynamic';

const CryptoHeatmapWidget = dynamic(
  () => import('@/components/CryptoHeatmapWidget'),
  {
    ssr: false,
    loading: () => <p className="text-center text-gray-400">Loading Crypto Heatmap...</p>,
  }
);

export default function CryptoHeatmapView() {
  return <CryptoHeatmapWidget />;
} 