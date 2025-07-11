'use client';

import dynamic from 'next/dynamic';

const NewsWidget = dynamic(
  () => import('@/components/NewsWidget'),
  {
    ssr: false,
    loading: () => <p className="text-center text-gray-400">Loading News...</p>,
  }
);

export default function NewsView() {
  return <NewsWidget />;
} 