import CryptoHeatmapView from './CryptoHeatmapView';

export default function CryptoHeatmapPage() {
  return (
    <div className="h-full flex flex-col p-6">
      <h1 className="text-3xl font-bold text-white mb-6">Crypto Heatmap</h1>
      <div className="bg-[#23272F] rounded-lg p-6 flex-1">
        <CryptoHeatmapView />
      </div>
    </div>
  );
} 