"use client";

import { useEffect, useRef, useState, useCallback } from "react";
// TODO: Load ALL_SYMBOLS from backend API. For now, fallback to a hardcoded array.
const ALL_SYMBOLS = [
  "AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD", "AUDUSD",
  "CADCHF", "CADJPY", "CHFJPY",
  "EURAUD", "EURCAD", "EURCHF", "EURGBP", "EURJPY", "EURNZD", "EURUSD",
  "GBPAUD", "GBPCAD", "GBPCHF", "GBPJPY", "GBPNZD", "GBPUSD",
  "NZDCAD", "NZDCHF", "NZDJPY", "NZDUSD",
  "USDCHF", "USDCAD", "USDJPY",
  "XAUUSD", "XAUEUR", "XAUGBP", "XAUJPY", "XAUAUD", "XAUCHF", "XAGUSD",
  "US500", "US30", "UK100", "GER40", "NAS100", "USDX", "EURX"
];

const SYMBOLS = (ALL_SYMBOLS || []).map((s) => {
  // Map to TradingView format: 'EURUSD' -> 'PEPPERSTONE:EURAUD', 'USDJPY' -> 'PEPPERSTONE:USDJPY', etc. FX_IDC:EURUSD → ICE feed (default)
  const tvSymbol = s.replace("/", "");
  return {
    label: s,
    value: `PEPPERSTONE:${tvSymbol}`,
  };
});
const TIMEFRAMES = [
  { label: "1 minute", value: "1" },
  { label: "5 minutes", value: "5" },
  { label: "15 minutes", value: "15" },
  { label: "1 hour", value: "60" },
  { label: "1 day", value: "D" },
];

export default function ChartsPage() {
  const chartContainer = useRef<HTMLDivElement>(null);
  const [symbol, setSymbol] = useState(SYMBOLS[0].value);
  const [interval, setInterval] = useState(TIMEFRAMES[1].value); // default to 5 min
  const [isFullscreen, setIsFullscreen] = useState(false);

  // Helper to (re)load TradingView widget
  const loadWidget = useCallback(() => {
    if (chartContainer.current) chartContainer.current.innerHTML = "";
    const script = document.createElement("script");
    script.src = "https://s3.tradingview.com/tv.js";
    script.async = true;
    script.onload = () => {
      // @ts-expect-error - window.TradingView is not typed
      if (window.TradingView) {
        // @ts-expect-error - window.TradingView is not typed
        new window.TradingView.widget({
          autosize: true,
          symbol,
          interval,
          timezone: "Etc/UTC",
          theme: "dark",
          style: "1",
          locale: "en",
          toolbar_bg: "#23272F",
          enable_publishing: false,
          hide_top_toolbar: false,
          hide_legend: false,
          container_id: "tradingview_chart",
        });
      }
    };
    chartContainer.current?.appendChild(script);
  }, [symbol, interval]);

  useEffect(() => {
    loadWidget();
  }, [loadWidget]);

  // Fullscreen logic
  const chartAreaRef = useRef<HTMLDivElement>(null);
  const handleFullscreen = () => {
    const el = chartAreaRef.current;
    if (!el) return;
    const elem = el as HTMLElement & { webkitRequestFullscreen?: () => void; msRequestFullscreen?: () => void };
    const doc = document as Document & { webkitExitFullscreen?: () => void; msExitFullscreen?: () => void };
    if (!isFullscreen) {
      if (el.requestFullscreen) el.requestFullscreen();
      else if (elem.webkitRequestFullscreen) elem.webkitRequestFullscreen();
      else if (elem.msRequestFullscreen) elem.msRequestFullscreen();
      setIsFullscreen(true);
    } else {
      if (document.exitFullscreen) document.exitFullscreen();
      else if (doc.webkitExitFullscreen) doc.webkitExitFullscreen();
      else if (doc.msExitFullscreen) doc.msExitFullscreen();
      setIsFullscreen(false);
    }
  };
  // Listen for fullscreen change to update state
  useEffect(() => {
    const handler = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };
    document.addEventListener("fullscreenchange", handler);
    return () => document.removeEventListener("fullscreenchange", handler);
  }, []);

  // Placeholder handlers
  const handleSave = () => {
    alert("Save functionality coming soon!");
  };
  const handleSettings = () => {
    alert("Settings functionality coming soon!");
  };

  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold text-white mb-6">Charts</h1>
      <div
        className={`bg-[#23272F] rounded-lg p-0 flex flex-col${isFullscreen ? " fixed inset-0 z-50" : ""}`}
        style={{ height: isFullscreen ? "100vh" : "calc(100vh - 120px)" }}
        ref={chartAreaRef}
      >
        {/* Toolbar */}
        <div className="flex items-center px-6 py-3 border-b border-gray-700 bg-[#23272F]">
          <button
            className="text-xs bg-blue-600 text-white px-3 py-1 rounded mr-4 font-semibold"
            onClick={handleFullscreen}
          >
            {isFullscreen ? "Exit Full Screen" : "Full Screen"}
          </button>
          <select
            className="bg-[#181A20] text-white px-2 py-1 rounded mr-2 border border-gray-700 text-sm"
            value={symbol}
            onChange={e => setSymbol(e.target.value)}
          >
            {SYMBOLS.map((s) => (
              <option key={s.value} value={s.value}>{s.label}</option>
            ))}
          </select>
          <select
            className="bg-[#181A20] text-white px-2 py-1 rounded mr-2 border border-gray-700 text-sm"
            value={interval}
            onChange={e => setInterval(e.target.value)}
          >
            {TIMEFRAMES.map((tf) => (
              <option key={tf.value} value={tf.value}>{tf.label}</option>
            ))}
          </select>
          <button
            className="bg-[#23272F] text-white px-2 py-1 rounded border border-gray-700 text-sm mr-2"
            onClick={handleSave}
          >
            Save
          </button>
          <div className="flex-1" />
          <button
            className="bg-[#23272F] text-white px-2 py-1 rounded border border-gray-700 text-sm mr-2"
            onClick={handleSettings}
          >
            <span role="img" aria-label="settings">⚙️</span>
          </button>
        </div>
        {/* Chart Area */}
        <div style={{ flex: 1, width: "100%", display: "flex", height: "100%" }}>
          <div ref={chartContainer} id="tradingview_chart" style={{ width: "100%", height: "100%", flex: 1 }} />
        </div>
      </div>
    </div>
  );
} 