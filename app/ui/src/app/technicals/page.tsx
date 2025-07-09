"use client";

import { useEffect, useRef, useState } from "react";
import { createChart, IChartApi, CandlestickSeries, LineSeries } from "lightweight-charts";

export default function TechnicalsPage() {
  const chartContainer = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const chartAreaRef = useRef<HTMLDivElement>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
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
  const TIMEFRAMES = [
    { label: "1 minute", value: "M1" },
    { label: "2 minutes", value: "M2" },
    { label: "3 minutes", value: "M3" },
    { label: "4 minutes", value: "M4" },
    { label: "5 minutes", value: "M5" },
    { label: "6 minutes", value: "M6" },
    { label: "10 minutes", value: "M10" },
    { label: "12 minutes", value: "M12" },
    { label: "15 minutes", value: "M15" },
    { label: "20 minutes", value: "M20" },
    { label: "30 minutes", value: "M30" },
    { label: "1 hour", value: "H1" },
    { label: "2 hours", value: "H2" },
    { label: "3 hours", value: "H3" },
    { label: "4 hours", value: "H4" },
    { label: "6 hours", value: "H6" },
    { label: "8 hours", value: "H8" },
    { label: "12 hours", value: "H12" },
    { label: "1 day", value: "D1" },
    { label: "1 week", value: "W1" },
    { label: "1 month", value: "MN1" },
  ];
  const BAR_COUNTS = [100, 250, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000];
  const RANGE_MODES = [
    { label: "Number of Bars", value: "bars" },
    { label: "Date Range", value: "date" },
  ];
  const today = new Date().toISOString().slice(0, 10);
  const weekAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString().slice(0, 10);
  const [rangeMode, setRangeMode] = useState("bars");
  const [barCount, setBarCount] = useState(BAR_COUNTS[0]);
  const [startDate, setStartDate] = useState(weekAgo);
  const [endDate, setEndDate] = useState(today);
  const [symbol, setSymbol] = useState(ALL_SYMBOLS[0]);
  const [interval, setInterval] = useState(TIMEFRAMES[4].value); // default to 5 min
  const [loading, setLoading] = useState(false);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [selectedIndicators, setSelectedIndicators] = useState<string[]>(['EMA20']);

  useEffect(() => {
    if (!chartContainer.current) return;
    // Clean up previous chart instance
    if (chartRef.current) {
      chartRef.current.remove();
      chartRef.current = null;
    }
    const chart = createChart(chartContainer.current, {
      width: chartContainer.current.clientWidth,
      height: chartContainer.current.clientHeight,
      layout: {
        background: { color: '#23272F' },
        textColor: '#D1D5DB',
      },
      grid: {
        vertLines: { color: '#363C4E' },
        horzLines: { color: '#363C4E' },
      },
      crosshair: {
        mode: 0,
      },
      rightPriceScale: {
        borderColor: '#485158',
        visible: true,
      },
      timeScale: {
        borderColor: '#485158',
        timeVisible: true,         // 👈 important for intraday
        secondsVisible: true,      // 👈 useful for <1H data
        visible: true,
      },
    });
    chartRef.current = chart;
    // Example candlestick data (placeholder, could be dynamic based on symbol/interval)
    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#26a69a',
      downColor: '#ef5350',
      borderVisible: false,
      wickUpColor: '#26a69a',
      wickDownColor: '#ef5350',
      priceFormat: {
        type: 'price',
        precision: 5,
        minMove: 0.00001,
      },
    });
    candleSeries.setData([
      { time: '2024-07-01', open: 1.1, high: 1.2, low: 1.05, close: 1.15 },
      { time: '2024-07-02', open: 1.15, high: 1.18, low: 1.1, close: 1.12 },
      { time: '2024-07-03', open: 1.12, high: 1.16, low: 1.1, close: 1.14 },
      { time: '2024-07-04', open: 1.14, high: 1.19, low: 1.13, close: 1.18 },
      { time: '2024-07-05', open: 1.18, high: 1.22, low: 1.17, close: 1.2 },
    ]);
    // ResizeObserver for responsive chart
    let resizeObserver: ResizeObserver | null = null;
    if (chartContainer.current) {
      resizeObserver = new ResizeObserver(entries => {
        for (const entry of entries) {
          if (chartRef.current && entry.contentRect) {
            chartRef.current.resize(entry.contentRect.width, entry.contentRect.height);
          }
        }
      });
      resizeObserver.observe(chartContainer.current);
    }
    return () => {
      if (resizeObserver && chartContainer.current) {
        resizeObserver.unobserve(chartContainer.current);
      }
      if (chartRef.current) {
        chartRef.current.remove();
        chartRef.current = null;
      }
    };
  }, [isFullscreen, symbol, interval]);

  // Fetch MT5 data and update chart
  useEffect(() => {
    if (!chartRef.current) return;
    setLoading(true);
    setFetchError(null);
    
    const paramsObj: Record<string, string> = {
      symbol,
      timeframe: interval,
      mode: rangeMode,
    };
    if (rangeMode === "bars") {
      paramsObj.bars = String(barCount);
    }
    if (rangeMode === "date") {
      paramsObj.start_date = startDate ? startDate.slice(0, 10) : '';
      paramsObj.end_date = endDate ? endDate.slice(0, 10) : '';
    }
    
    const params = new URLSearchParams(paramsObj);
    
    // Fetch price data
    fetch(`http://127.0.0.1:8001/api/mt5-data?${params.toString()}`)
      .then(res => res.json())
      .then(json => {
        if (json.data && chartRef.current) {
          // Remove all series and add a new one
          chartRef.current.remove();
          const chart = createChart(chartContainer.current!, {
            width: chartContainer.current!.clientWidth,
            height: chartContainer.current!.clientHeight,
            layout: {
              background: { color: '#23272F' },
              textColor: '#D1D5DB',
            },
            grid: {
              vertLines: { color: '#363C4E' },
              horzLines: { color: '#363C4E' },
            },
            crosshair: { mode: 0 },
            rightPriceScale: {
                borderColor: '#485158',
                visible: true,
              },
              timeScale: {
                borderColor: '#485158',
                timeVisible: true,         // 👈 important for intraday
                secondsVisible: true,      // 👈 useful for <1H data
                visible: true,
              },
          });
          chartRef.current = chart;
          
          // Add candlestick series
          const candleSeries = chart.addSeries(CandlestickSeries, {
            upColor: '#26a69a',
            downColor: '#ef5350',
            borderVisible: false,
            wickUpColor: '#26a69a',
            wickDownColor: '#ef5350',
            priceFormat: {
              type: 'price',
              precision: 5,
              minMove: 0.00001,
            },
          });
          candleSeries.setData(json.data);
          
          // Fetch and add indicators if any are selected
          if (selectedIndicators.length > 0) {
            const indicatorParams = new URLSearchParams({
              ...paramsObj,
              indicators: selectedIndicators.join(',')
            });
            
            fetch(`http://127.0.0.1:8001/api/technical-indicators?${indicatorParams.toString()}`)
              .then(res => res.json())
                             .then(indicatorJson => {
                 if (indicatorJson.indicators && chartRef.current) {
                   // Add line series for each indicator
                   Object.entries(indicatorJson.indicators).forEach(([, values]) => {
                     if (values && Array.isArray(values)) {
                       const lineData = json.data.map((candle: { time: string | number }, index: number) => ({
                         time: candle.time,
                         value: values[index] || null
                       })).filter((item: { value: number | null }) => item.value !== null);
                       
                       if (lineData.length > 0) {
                         const lineSeries = chartRef.current!.addSeries(LineSeries, {
                           color: getIndicatorColor(),
                           lineWidth: 2,
                           priceFormat: {
                             type: 'price',
                             precision: 5,
                             minMove: 0.00001,
                           },
                         });
                         lineSeries.setData(lineData);
                       }
                     }
                   });
                }
              })
              .catch(e => console.error('Error fetching indicators:', e));
          }
        } else {
          setFetchError(json.error || 'No data');
        }
      })
      .catch(e => setFetchError(e.message))
      .finally(() => setLoading(false));
  }, [symbol, interval, rangeMode, barCount, startDate, endDate, selectedIndicators]);
  
  // Helper function to get indicator colors
  const getIndicatorColor = (): string => {
    return '#FF6B6B'; // Red color for EMA20
  };

  // Fullscreen logic
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

  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold text-white mb-6">Technicals</h1>
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
            {ALL_SYMBOLS.map((s) => (
              <option key={s} value={s}>{s}</option>
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
          <select
            className="bg-[#181A20] text-white px-2 py-1 rounded mr-2 border border-gray-700 text-sm"
            value={rangeMode}
            onChange={e => setRangeMode(e.target.value)}
          >
            {RANGE_MODES.map((mode) => (
              <option key={mode.value} value={mode.value}>{mode.label}</option>
            ))}
          </select>
          {rangeMode === "bars" ? (
            <select
              className="bg-[#181A20] text-white px-2 py-1 rounded mr-2 border border-gray-700 text-sm"
              value={barCount}
              onChange={e => setBarCount(Number(e.target.value))}
            >
              {BAR_COUNTS.map((n) => (
                <option key={n} value={n}>{n} Bars</option>
              ))}
            </select>
          ) : (
            <>
              <input
                type="date"
                className="bg-[#181A20] text-white px-2 py-1 rounded mr-2 border border-gray-700 text-sm"
                value={startDate}
                onChange={e => setStartDate(e.target.value)}
                max={endDate}
              />
              <input
                type="date"
                className="bg-[#181A20] text-white px-2 py-1 rounded mr-2 border border-gray-700 text-sm"
                value={endDate}
                onChange={e => setEndDate(e.target.value)}
                min={startDate}
                max={today}
              />
            </>
          )}
          
          {/* Indicator Controls */}
          <div className="ml-4 flex items-center">
            <span className="text-white text-sm mr-2">EMA20:</span>
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={selectedIndicators.includes('EMA20')}
                onChange={(e) => {
                  if (e.target.checked) {
                    setSelectedIndicators(['EMA20']);
                  } else {
                    setSelectedIndicators([]);
                  }
                }}
                className="mr-1"
              />
              <span className="text-white text-xs">Show</span>
            </label>
          </div>
        </div>
        {loading && <div className="text-blue-400 px-6 py-2">Loading data...</div>}
        {fetchError && <div className="text-red-400 px-6 py-2">Error: {fetchError}</div>}
        {/* Chart Area */}
        <div className="chart-container" style={{ flex: 1, width: "100%", display: "flex", height: "100%" }}>
          <div ref={chartContainer} style={{ width: '100%', height: '100%', flex: 1 }} />
          <style jsx>{`
            .chart-container {
              font-family: 'Arial', sans-serif;
            }
          `}</style>
        </div>
      </div>
    </div>
  );
} 