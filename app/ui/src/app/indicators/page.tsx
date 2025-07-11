"use client";

import { useEffect, useRef, useState } from "react";
import Plot from "../../components/PlotlyNoSSR";

interface OHLCVBar {
  time: string | number;
  open: number;
  high: number;
  low: number;
  close: number;
}

// EMA calculation function
const calculateEMA = (data: number[], period: number): number[] => {
  const ema: number[] = [];
  const multiplier = 2 / (period + 1);
  
  // First EMA value is SMA of first 'period' values
  let sum = 0;
  for (let i = 0; i < period && i < data.length; i++) {
    sum += data[i];
  }
  ema[period - 1] = sum / period;
  
  // Calculate EMA for remaining values
  for (let i = period; i < data.length; i++) {
    ema[i] = (data[i] * multiplier) + (ema[i - 1] * (1 - multiplier));
  }
  
  return ema;
};



export default function IndicatorsPage() {
  // =================================================================================
  // Component State and Refs
  // =================================================================================
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
    { label: "Today", value: "today" },
  ];
  const today = new Date().toISOString().slice(0, 10);
  const weekAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString().slice(0, 10);
  const [rangeMode, setRangeMode] = useState("today"); // default to today
  const [barCount, setBarCount] = useState(BAR_COUNTS[0]);
  const [startDate, setStartDate] = useState(weekAgo);
  const [endDate, setEndDate] = useState(today);
  const [symbol, setSymbol] = useState(ALL_SYMBOLS[0]);
  const [interval, setInterval] = useState(TIMEFRAMES[4].value); // default to 5 min
  const [selectedIndicators, setSelectedIndicators] = useState<string[]>([]);
  const [selectedSMC, setSelectedSMC] = useState<string[]>([]);
  const [selectedSignals, setSelectedSignals] = useState<string[]>([]);
  const [emaPeriod, setEmaPeriod] = useState(20);

  const SMC_OPTIONS = [
    { label: 'Swingline', value: 'Swingline' },
    { label: 'Swing Point', value: 'SwingPoint' },
    { label: 'S/R Lines', value: 'SRLines' },
    { label: 'BOS', value: 'BOS' },
    { label: 'CHoCH', value: 'CHoCH' },
    { label: 'OrderBlocks', value: 'OrderBlocks' },
    { label: 'Fib Signals', value: 'Fibonacci Signal' },
    { label: 'Retest Signal', value: 'Retest Signal' },
  ];
  const [smcDropdownOpen, setSmcDropdownOpen] = useState(false);
  const smcDropdownRef = useRef<HTMLDivElement>(null);

  // Close dropdown on outside click
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (smcDropdownRef.current && !smcDropdownRef.current.contains(event.target as Node)) {
        setSmcDropdownOpen(false);
      }
    }
    if (smcDropdownOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    } else {
      document.removeEventListener('mousedown', handleClickOutside);
    }
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [smcDropdownOpen]);

  // =================================================================================
  // Real Data State
  // =================================================================================
  const [ohlcv, setOhlcv] = useState<OHLCVBar[]>([]);
  const [smcData, setSmcData] = useState<{ 
    swingline: number[]; 
    swingvalue: number[]; 
    swingpoint: number[]; 
    Resistance: number[]; 
    Support: number[]; 
    BOS: number[]; 
    CHoCH: number[];
    Bullish_Order_Block_Top: number[];
    Bullish_Order_Block_Bottom: number[];
    Bullish_Order_Block_Mitigated: number[];
    Bearish_Order_Block_Top: number[];
    Bearish_Order_Block_Bottom: number[];
    Bearish_Order_Block_Mitigated: number[];
    fib_signal?: number[];
    retest_signal?: number[];
  } | null>(null);
  const [loading, setLoading] = useState(false);
  const [fetchError, setFetchError] = useState<string | null>(null);

  // =================================================================================
  // Fetch OHLCV Data from Flask API
  // =================================================================================
  useEffect(() => {
    setLoading(true);
    setFetchError(null);
    setOhlcv([]);
    setSmcData(null);
    
    // Auto-set dates for "today" mode
    if (rangeMode === "today") {
      setStartDate(today);
      setEndDate(today);
    }
    
    const paramsObj: Record<string, string> = {
      symbol,
      timeframe: interval,
      mode: rangeMode,
    };
    if (rangeMode === "bars") {
      paramsObj.bars = String(barCount);
    }
    if (rangeMode === "date" || rangeMode === "today") {
      paramsObj.start_date = startDate ? startDate.slice(0, 10) : '';
      paramsObj.end_date = endDate ? endDate.slice(0, 10) : '';
    }
    const params = new URLSearchParams(paramsObj);
    
    // Fetch OHLCV data
    fetch(`http://127.0.0.1:8001/api/mt5-data?${params.toString()}`)
      .then(res => res.json())
      .then(json => {
        if (json.data) {
          setOhlcv(json.data);
        } else {
          setFetchError(json.error || 'No data');
        }
      })
      .catch(e => setFetchError(e.message));
    
    // Fetch SMC data
    fetch(`http://127.0.0.1:8001/api/smc-data?${params.toString()}`)
      .then(res => res.json())
      .then(json => {
        if (json.smc_data) {
          setSmcData(json.smc_data);
        }
      })
      .catch(e => console.error('SMC data fetch error:', e))
      .finally(() => setLoading(false));
  }, [symbol, interval, rangeMode, barCount, startDate, endDate, today]);

  // =================================================================================
  // Fullscreen Logic
  // =================================================================================
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
  useEffect(() => {
    const handler = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };
    document.addEventListener("fullscreenchange", handler);
    return () => document.removeEventListener("fullscreenchange", handler);
  }, []);

  // =================================================================================
  // Calculate Indicators
  // =================================================================================
  const calculateIndicators = () => {
    const closePrices = ohlcv.map(d => d.close);
    const times = ohlcv.map(d => d.time);
    
    const indicators: Array<
      | {
          x: (string | number)[];
          y: number[];
          type: string;
          mode: string;
          name: string;
          line: { color: string; width: number; dash?: string };
          showlegend: boolean;
          line_shape?: 'hv';
          connectgaps?: boolean;
        }
      | {
          x: (string | number)[];
          y: number[];
          type: string;
          mode: string;
          name: string;
          marker: { color: string[]; size: number; symbol: string | string[]; line: { width: number; color: string } };
          showlegend: boolean;
        }
    > = [];
    
    // Calculate EMA
    if (selectedIndicators.includes('EMA') && closePrices.length >= emaPeriod) {
      const ema = calculateEMA(closePrices, emaPeriod);
      // Only show EMA values from index (emaPeriod-1) onwards (where we have valid EMA)
      const validEma = ema.slice(emaPeriod - 1);
      const validTimes = times.slice(emaPeriod - 1);
      
      indicators.push({
        x: validTimes,
        y: validEma,
        type: 'scatter',
        mode: 'lines',
        name: `EMA ${emaPeriod}`,
        line: {
          color: '#FF6B6B',
          width: 2
        },
        showlegend: true
      });
    }
    
    // Calculate Swinglines using server SMC data
    if (selectedSMC.includes('Swingline') && smcData) {
      const { swingline, swingvalue } = smcData;
      // Build segments for consecutive same nonzero swingline values
      let segmentX: (string | number)[] = [];
      let segmentY: number[] = [];
      let currentType: number | null = null;
      for (let i = 0; i < swingline.length; i++) {
        if (swingline[i] === 1 || swingline[i] === -1) {
          if (currentType === swingline[i]) {
            // Continue current segment
            segmentX.push(times[i]);
            segmentY.push(swingvalue[i]);
          } else {
            // End previous segment if exists
            if (segmentX.length > 1) {
              indicators.push({
                x: segmentX,
                y: segmentY,
                type: 'scatter',
                mode: 'lines',
                name: currentType === 1 ? 'Swingline Up' : 'Swingline Down',
                line: {
                  color: currentType === 1 ? 'rgb(38,166,154)' : 'rgb(239,83,80)',
                  width: 2
                },
                showlegend: false
              });
            }
            // Start new segment
            currentType = swingline[i];
            segmentX = [times[i]];
            segmentY = [swingvalue[i]];
          }
        } else {
          // End current segment if exists
          if (segmentX.length > 1) {
            indicators.push({
              x: segmentX,
              y: segmentY,
              type: 'scatter',
              mode: 'lines',
              name: currentType === 1 ? 'Swingline Up' : 'Swingline Down',
              line: {
                color: currentType === 1 ? 'rgb(38,166,154)' : 'rgb(239,83,80)',
                width: 2
              },
              showlegend: false
            });
          }
          currentType = null;
          segmentX = [];
          segmentY = [];
        }
      }
      // Push last segment if needed
      if (segmentX.length > 1) {
        indicators.push({
          x: segmentX,
          y: segmentY,
          type: 'scatter',
          mode: 'lines',
          name: currentType === 1 ? 'Swingline Up' : 'Swingline Down',
          line: {
            color: currentType === 1 ? 'rgb(38,166,154)' : 'rgb(239,83,80)',
            width: 2
          },
          showlegend: false
        });
      }
    }
    
    // Draw Swing Points using server SMC data
    if (selectedSMC.includes('SwingPoint') && smcData && smcData.swingpoint) {
      const swingPointsX: (string | number)[] = [];
      const swingPointsY: number[] = [];
      const swingPointsColor: string[] = [];
      for (let i = 0; i < smcData.swingpoint.length; i++) {
        if (smcData.swingpoint[i] === 1) {
          swingPointsX.push(times[i]);
          swingPointsY.push(ohlcv[i]?.high ?? null);
          swingPointsColor.push('rgb(10,30,80)'); // dark blue
        } else if (smcData.swingpoint[i] === -1) {
          swingPointsX.push(times[i]);
          swingPointsY.push(ohlcv[i]?.low ?? null);
          swingPointsColor.push('rgb(120,10,30)'); // dark red
        }
      }
      if (swingPointsX.length > 0) {
        indicators.push({
          x: swingPointsX,
          y: swingPointsY,
          type: 'scatter',
          mode: 'markers',
          name: 'Swing Point',
          marker: {
            color: swingPointsColor,
            size: 14,
            symbol: 'circle',
            line: { width: 1, color: '#fff' }
          },
          showlegend: false
        });
      }
    }

    // Draw Support and Resistance stepwise lines
    if (selectedSMC.includes('SRLines') && smcData && smcData.Resistance && smcData.Resistance.length === times.length) {
      indicators.push({
        x: times,
        y: smcData.Resistance,
        type: 'scatter',
        mode: 'lines',
        name: 'Resistance',
        line: { color: 'blue', width: 2 },
        showlegend: true,
        line_shape: 'hv',
        connectgaps: false
      });
    }
    if (selectedSMC.includes('SRLines') && smcData && smcData.Support && smcData.Support.length === times.length) {
      indicators.push({
        x: times,
        y: smcData.Support,
        type: 'scatter',
        mode: 'lines',
        name: 'Support',
        line: { color: 'red', width: 2 },
        showlegend: true,
        line_shape: 'hv',
        connectgaps: false
      });
    }
    
    // Draw BOS markers using server SMC data
    if (selectedSMC.includes('BOS') && smcData && smcData.BOS) {
      const bosX: (string | number)[] = [];
      const bosY: number[] = [];
      const bosColor: string[] = [];
      const bosSymbol: string[] = [];
      for (let i = 0; i < smcData.BOS.length; i++) {
        if (smcData.BOS[i] === 1) {
          bosX.push(times[i]);
          bosY.push(ohlcv[i]?.high ?? null);
          bosColor.push('rgb(10,30,80)'); // dark blue
          bosSymbol.push('triangle-up');
        } else if (smcData.BOS[i] === -1) {
          bosX.push(times[i]);
          bosY.push(ohlcv[i]?.low ?? null);
          bosColor.push('rgb(120,10,30)'); // dark red
          bosSymbol.push('triangle-down');
        }
      }
      if (bosX.length > 0) {
        indicators.push({
          x: bosX,
          y: bosY,
          type: 'scatter',
          mode: 'markers',
          name: 'BOS',
          marker: {
            color: bosColor,
            size: 18,
            symbol: bosSymbol,
            line: { width: 1, color: '#fff' }
          },
          showlegend: false
        });
      }
    }
    
    // Draw CHoCH markers using server SMC data
    if (selectedSMC.includes('CHoCH') && smcData && smcData.CHoCH) {
      const chochX: (string | number)[] = [];
      const chochY: number[] = [];
      const chochColor: string[] = [];
      const chochSymbol: string[] = [];
      for (let i = 0; i < smcData.CHoCH.length; i++) {
        if (smcData.CHoCH[i] === 1) {
          chochX.push(times[i]);
          chochY.push(ohlcv[i]?.high ?? null);
          chochColor.push('rgb(10,30,80)'); // dark blue
          chochSymbol.push('hexagram');
        } else if (smcData.CHoCH[i] === -1) {
          chochX.push(times[i]);
          chochY.push(ohlcv[i]?.low ?? null);
          chochColor.push('rgb(120,10,30)'); // dark red
          chochSymbol.push('hexagram');
        }
      }
      if (chochX.length > 0) {
        indicators.push({
          x: chochX,
          y: chochY,
          type: 'scatter',
          mode: 'markers',
          name: 'CHoCH',
          marker: {
            color: chochColor,
            size: 18,
            symbol: chochSymbol,
            line: { width: 1, color: '#fff' }
          },
          showlegend: false
        });
      }
    }
    
    // Draw Order Blocks using server SMC data
    if (selectedSMC.includes('OrderBlocks') && smcData) {
      // Draw last contiguous Bullish Order Block region as a rectangle
      if (smcData.Bullish_Order_Block_Top && smcData.Bullish_Order_Block_Bottom) {
        let start = -1, end = -1;
        for (let i = smcData.Bullish_Order_Block_Top.length - 1; i >= 0; i--) {
          const top = smcData.Bullish_Order_Block_Top[i];
          const bottom = smcData.Bullish_Order_Block_Bottom[i];
          if (top !== null && bottom !== null && top !== undefined && bottom !== undefined) {
            if (end === -1) end = i;
            start = i;
          } else if (end !== -1) {
            break;
          }
        }
        if (start !== -1 && end !== -1 && end > start) {
          const top = smcData.Bullish_Order_Block_Top[start];
          const bottom = smcData.Bullish_Order_Block_Bottom[start];
          const mitigated = smcData.Bullish_Order_Block_Mitigated?.[end] || 0;
          indicators.push({
            x: [times[start], times[end], times[end], times[start], times[start]],
            y: [bottom, bottom, top, top, bottom],
            type: 'scatter',
            mode: 'lines',
            name: `Bullish Order Block${mitigated ? ' (Mitigated)' : ''}`,
            line: {
              color: 'rgba(0, 255, 255, 0.25)', // Cyan with 25% opacity
              width: 2
            },
            showlegend: true
          });
        }
      }
      // Draw last contiguous Bearish Order Block region as a rectangle
      if (smcData.Bearish_Order_Block_Top && smcData.Bearish_Order_Block_Bottom) {
        let start = -1, end = -1;
        for (let i = smcData.Bearish_Order_Block_Top.length - 1; i >= 0; i--) {
          const top = smcData.Bearish_Order_Block_Top[i];
          const bottom = smcData.Bearish_Order_Block_Bottom[i];
          if (top !== null && bottom !== null && top !== undefined && bottom !== undefined) {
            if (end === -1) end = i;
            start = i;
          } else if (end !== -1) {
            break;
          }
        }
        if (start !== -1 && end !== -1 && end > start) {
          const top = smcData.Bearish_Order_Block_Top[start];
          const bottom = smcData.Bearish_Order_Block_Bottom[start];
          const mitigated = smcData.Bearish_Order_Block_Mitigated?.[end] || 0;
          indicators.push({
            x: [times[start], times[end], times[end], times[start], times[start]],
            y: [bottom, bottom, top, top, bottom],
            type: 'scatter',
            mode: 'lines',
            name: `Bearish Order Block${mitigated ? ' (Mitigated)' : ''}`,
            line: {
              color: 'rgba(255, 0, 255, 0.25)', // Magenta with 25% opacity
              width: 2
            },
            showlegend: true
          });
        }
      }
    }
    
    // Draw Signal arrows using server SMC data
    if (selectedSignals.includes('Fibonacci Signal') && smcData && smcData.fib_signal) {
      const signalX: (string | number)[] = [];
      const signalY: number[] = [];
      const signalColor: string[] = [];
      const signalSymbol: string[] = [];
      for (let i = 0; i < smcData.fib_signal.length; i++) {
        if (smcData.fib_signal[i] === 1) {
          signalX.push(times[i]);
          signalY.push(ohlcv[i]?.low ?? null);
          signalColor.push('rgb(0, 150, 255)'); // Blue
          signalSymbol.push('star-triangle-up');
        } else if (smcData.fib_signal[i] === -1) {
          signalX.push(times[i]);
          signalY.push(ohlcv[i]?.high ?? null);
          signalColor.push('rgb(255, 50, 50)'); // Red
          signalSymbol.push('star-triangle-down');
        }
      }
      if (signalX.length > 0) {
        indicators.push({
          x: signalX,
          y: signalY,
          type: 'scatter',
          mode: 'markers',
          name: 'Fibonacci Signal',
          marker: {
            color: signalColor,
            size: 20,
            symbol: signalSymbol,
            line: { width: 2, color: '#fff' }
          },
          showlegend: true
        });
      }
    }
    
    // Draw Retest Signal arrows using server SMC data
    if (selectedSignals.includes('Retest Signal') && smcData && smcData.retest_signal) {
      const retestX: (string | number)[] = [];
      const retestY: number[] = [];
      const retestColor: string[] = [];
      const retestSymbol: string[] = [];
      for (let i = 0; i < smcData.retest_signal.length; i++) {
        if (smcData.retest_signal[i] === 1) {
          retestX.push(times[i]);
          retestY.push(ohlcv[i]?.low ?? null);
          retestColor.push('rgb(0, 255, 180)'); // Teal for buy
          retestSymbol.push('star-triangle-up');
        } else if (smcData.retest_signal[i] === -1) {
          retestX.push(times[i]);
          retestY.push(ohlcv[i]?.high ?? null);
          retestColor.push('rgb(255, 180, 0)'); // Orange for sell
          retestSymbol.push('star-triangle-down');
        }
      }
      if (retestX.length > 0) {
        indicators.push({
          x: retestX,
          y: retestY,
          type: 'scatter',
          mode: 'markers',
          name: 'Retest Signal',
          marker: {
            color: retestColor,
            size: 20,
            symbol: retestSymbol,
            line: { width: 2, color: '#fff' }
          },
          showlegend: true
        });
      }
    }
    
    return indicators;
  };

  // =================================================================================
  // Plotly Data and Layout
  // =================================================================================
  const plotlyData = [
    {
      x: ohlcv.map(d => d.time), // Use original ISO strings
      open: ohlcv.map(d => d.open),
      high: ohlcv.map(d => d.high),
      low: ohlcv.map(d => d.low),
      close: ohlcv.map(d => d.close),
      type: 'candlestick',
      name: symbol,
      increasing: { line: { color: '#26a69a' } },
      decreasing: { line: { color: '#ef5350' } },
    },
    ...calculateIndicators()
  ];
  const [showCrosshair, setShowCrosshair] = useState(true);

  const plotlyLayout = {
    dragmode: 'zoom',
    margin: { t: 30, r: 40, b: 40, l: 60 },
    paper_bgcolor: '#23272F',
    plot_bgcolor: '#23272F',
    font: { color: '#D1D5DB' },
    xaxis: {
      type: 'date',
      rangeslider: { visible: false },
      gridcolor: '#363C4E',
      tickfont: { color: '#D1D5DB' },
      rangebreaks: [
        { pattern: 'day of week', bounds: [6, 1] }, // Hide weekends
      ],
      showspikes: showCrosshair,
      spikecolor: '#D1D5DB',
      spikethickness: 1,
      spikedash: 'solid',
      spikemode: 'across',
      spikesnap: 'cursor',
    },
    yaxis: {
      gridcolor: '#363C4E',
      tickfont: { color: '#D1D5DB' },
      showspikes: showCrosshair,
      spikecolor: '#D1D5DB',
      spikethickness: 1,
      spikedash: 'solid',
      spikemode: 'across',
      spikesnap: 'cursor',
    },
    showlegend: true,
    legend: {
      x: 0.02,
      y: 0.98,
      bgcolor: 'rgba(35, 39, 47, 0.8)',
      bordercolor: '#363C4E',
      borderwidth: 1,
      font: { color: '#D1D5DB' }
    },
    autosize: true,
    hovermode: 'x unified',
  };

  // =================================================================================
  // Render Logic
  // =================================================================================
  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold text-white mb-6">Indicators Analysis</h1>
      <div
        className={`bg-[#23272F] rounded-lg p-0 flex flex-col${isFullscreen ? " fixed inset-0 z-50" : ""}`}
        style={{ height: isFullscreen ? "100vh" : "calc(100vh - 120px)" }}
        ref={chartAreaRef}
      >
        {/* Toolbar - now two rows */}
        <div className="flex flex-col px-6 py-3 border-b border-gray-700 bg-[#23272F]">
          {/* Row 1: Main controls */}
          <div className="flex flex-wrap items-center mb-2 gap-2">
            <button
              className="text-xs bg-blue-600 text-white px-3 py-1 rounded font-semibold"
              onClick={handleFullscreen}
            >
              {isFullscreen ? "Exit Full Screen" : "Full Screen"}
            </button>
            <button
              className={`text-xs px-3 py-1 rounded font-semibold ${
                showCrosshair 
                  ? 'bg-green-600 text-white' 
                  : 'bg-gray-600 text-gray-300'
              }`}
              onClick={() => setShowCrosshair(!showCrosshair)}
              title="Toggle Crosshair"
            >
              {showCrosshair ? "Crosshair ON" : "Crosshair OFF"}
            </button>
            <select
              className="bg-[#181A20] text-white px-2 py-1 rounded border border-gray-700 text-sm"
              value={symbol}
              onChange={e => setSymbol(e.target.value)}
            >
              {ALL_SYMBOLS.map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
            <select
              className="bg-[#181A20] text-white px-2 py-1 rounded border border-gray-700 text-sm"
              value={interval}
              onChange={e => setInterval(e.target.value)}
            >
              {TIMEFRAMES.map((tf) => (
                <option key={tf.value} value={tf.value}>{tf.label}</option>
              ))}
            </select>
            <select
              className="bg-[#181A20] text-white px-2 py-1 rounded border border-gray-700 text-sm"
              value={rangeMode}
              onChange={e => setRangeMode(e.target.value)}
            >
              {RANGE_MODES.map((mode) => (
                <option key={mode.value} value={mode.value}>{mode.label}</option>
              ))}
            </select>
            {rangeMode === "bars" ? (
              <select
                className="bg-[#181A20] text-white px-2 py-1 rounded border border-gray-700 text-sm"
                value={barCount}
                onChange={e => setBarCount(Number(e.target.value))}
              >
                {BAR_COUNTS.map((n) => (
                  <option key={n} value={n}>{n} Bars</option>
                ))}
              </select>
            ) : rangeMode === "today" ? (
              <span className="text-white text-sm">Today: {today}</span>
            ) : (
              <>
                <input
                  type="date"
                  className="bg-[#181A20] text-white px-2 py-1 rounded border border-gray-700 text-sm"
                  value={startDate}
                  onChange={e => setStartDate(e.target.value)}
                  max={endDate}
                />
                <input
                  type="date"
                  className="bg-[#181A20] text-white px-2 py-1 rounded border border-gray-700 text-sm"
                  value={endDate}
                  onChange={e => setEndDate(e.target.value)}
                  min={startDate}
                  max={today}
                />
              </>
            )}
          </div>
          {/* Row 2: Indicator controls */}
          <div className="flex flex-wrap items-center gap-4">
            {/* Indicator Controls */}
            <div className="flex items-center">
              <span className="text-white text-sm mr-2">EMA:</span>
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={selectedIndicators.includes('EMA')}
                  onChange={(e) => {
                    if (e.target.checked) {
                      setSelectedIndicators(['EMA']);
                    } else {
                      setSelectedIndicators([]);
                    }
                  }}
                  className="mr-1"
                />
                <span className="text-white text-xs">Show</span>
              </label>
              <select
                className="bg-[#181A20] text-white px-2 py-1 rounded ml-2 border border-gray-700 text-sm"
                value={emaPeriod}
                onChange={e => setEmaPeriod(Number(e.target.value))}
                disabled={!selectedIndicators.includes('EMA')}
              >
                <option value={5}>5</option>
                <option value={10}>10</option>
                <option value={12}>12</option>
                <option value={20}>20</option>
                <option value={50}>50</option>
                <option value={100}>100</option>
                <option value={144}>144</option>
                <option value={200}>200</option>
                <option value={288}>288</option>
              </select>
            </div>
            {/* SMC Controls - custom dropdown */}
            <div className="flex items-center relative" ref={smcDropdownRef}>
              <button
                className="bg-[#181A20] text-white px-3 py-1 rounded border border-gray-700 text-sm flex items-center min-w-[120px]"
                onClick={() => setSmcDropdownOpen((open) => !open)}
                type="button"
              >
                {selectedSMC.length > 0
                  ? SMC_OPTIONS.filter(opt => selectedSMC.includes(opt.value)).map(opt => opt.label).join(', ')
                  : 'SMC Tools'}
                <svg className="ml-2 w-3 h-3" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" /></svg>
              </button>
              {smcDropdownOpen && (
                <div className="absolute left-0 mt-2 w-48 bg-[#23272F] border border-gray-700 rounded shadow-lg z-50 p-2">
                  {SMC_OPTIONS.map(opt => (
                    <label key={opt.value} className="flex items-center px-2 py-1 hover:bg-[#181A20] rounded cursor-pointer">
                      <input
                        type="checkbox"
                        checked={selectedSMC.includes(opt.value)}
                        onChange={e => {
                          if (e.target.checked) {
                            setSelectedSMC(prev => [...prev.filter(item => item !== opt.value), opt.value]);
                            if (opt.value === 'Fibonacci Signal') {
                              setSelectedSignals(prev => [...prev.filter(item => item !== 'Fibonacci Signal'), 'Fibonacci Signal']);
                            }
                            if (opt.value === 'Retest Signal') {
                              setSelectedSignals(prev => [...prev.filter(item => item !== 'Retest Signal'), 'Retest Signal']);
                            }
                          } else {
                            setSelectedSMC(prev => prev.filter(item => item !== opt.value));
                            if (opt.value === 'Fibonacci Signal') {
                              setSelectedSignals(prev => prev.filter(item => item !== 'Fibonacci Signal'));
                            }
                            if (opt.value === 'Retest Signal') {
                              setSelectedSignals(prev => prev.filter(item => item !== 'Retest Signal'));
                            }
                          }
                        }}
                        className="mr-2"
                      />
                      <span className="text-white text-xs">{opt.label}</span>
                    </label>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
        {loading && <div className="text-blue-400 px-6 py-2">Loading data...</div>}
        {fetchError && <div className="text-red-400 px-6 py-2">Error: {fetchError}</div>}
        {/* Chart Area */}
        <div className="chart-container" style={{ flex: 1, width: "100%", display: "flex", height: "100%", position: "relative" }}>
          <Plot
            data={plotlyData}
            layout={{ ...plotlyLayout, width: undefined, height: undefined, autosize: true }}
            useResizeHandler={true}
            style={{ width: '100%', height: '100%' }}
            config={{ displayModeBar: true, responsive: true }}
          />
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