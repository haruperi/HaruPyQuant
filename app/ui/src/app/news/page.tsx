import NewsView from './NewsView';

export default function NewsPage() {
  return (
    <div className="h-full flex flex-col p-6">
      <h1 className="text-3xl font-bold text-white mb-6">Market News</h1>
      <div className="bg-[#23272F] rounded-lg p-6 flex-1">
        <NewsView />
      </div>
    </div>
  );
} 