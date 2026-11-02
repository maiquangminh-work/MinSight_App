import { useEffect, useRef, useState } from 'react';
import {
  createChart,
  CandlestickSeries,
  HistogramSeries,
  LineSeries,
  AreaSeries,
  createSeriesMarkers,
} from 'lightweight-charts';
import { formatVND } from '../../utils/formatting';

/**
 * Generates subsequent business days (Mon-Fri) in YYYY-MM-DD format using UTC
 * to prevent any timezone shifting across client environments.
 */
function getNextBusinessDays(dateStr, count) {
  if (!dateStr || count <= 0) return [];
  const parts = dateStr.split('-');
  if (parts.length < 3) return [];
  const y = Number(parts[0]);
  const m = Number(parts[1]);
  const d = Number(parts[2]);

  const dates = [];
  const curr = new Date(Date.UTC(y, m - 1, d));
  while (dates.length < count) {
    curr.setUTCDate(curr.getUTCDate() + 1);
    const day = curr.getUTCDay();
    if (day !== 0 && day !== 6) {
      const year = curr.getUTCFullYear();
      const month = String(curr.getUTCMonth() + 1).padStart(2, '0');
      const date = String(curr.getUTCDate()).padStart(2, '0');
      dates.push(`${year}-${month}-${date}`);
    }
  }
  return dates;
}

const TIMEFRAME_BAR_COUNT = {
  '1M': 22,
  '3M': 66,
  '6M': 132,
  '1Y': 252,
  All: 99999,
};

export default function TradingViewChart({
  chartData = [],
  predictions = [],
  attentionWeights = [],
  activeIndicators = new Set(),
  forecastEnabled = true,
  confidenceBandEnabled = true,
  indicatorData = {},
  isLightTheme = false,
  timeframe = '1Y',
}) {
  const chartContainerRef = useRef(null);
  const subContainerRef = useRef(null);

  // Core chart instances
  const mainChartRef = useRef(null);
  const subChartRef = useRef(null);
  const isDisposedRef = useRef(false);
  const isSyncingRef = useRef(false);

  // Series refs
  const candleSeriesRef = useRef(null);
  const volumeSeriesRef = useRef(null);
  const attnSeriesRef = useRef(null);
  const ema12SeriesRef = useRef(null);
  const ema26SeriesRef = useRef(null);
  const bollUpperSeriesRef = useRef(null);
  const bollLowerSeriesRef = useRef(null);
  const predLineSeriesRef = useRef(null);
  const upperBandSeriesRef = useRef(null);
  const lowerBandSeriesRef = useRef(null);

  // Markers wrapper refs
  const forecastMarkersRef = useRef(null);
  const attnMarkersRef = useRef(null);

  // Tooltip state
  const [hoverData, setHoverData] = useState(null);

  // =========================================================================
  // 1. Initial Mount: Create Main Chart & Sub Chart ONCE
  // =========================================================================
  useEffect(() => {
    const mainContainer = chartContainerRef.current;
    const subContainer = subContainerRef.current;
    if (!mainContainer || !subContainer) return;

    isDisposedRef.current = false;
    isSyncingRef.current = false;

    const bg = isLightTheme ? '#ffffff' : '#07090e';
    const textColor = isLightTheme ? '#94a3b8' : '#94a3b8';
    const gridColor = isLightTheme ? 'rgba(0,0,0,0.04)' : 'rgba(255,255,255,0.03)';
    const borderColor = isLightTheme ? 'rgba(0,0,0,0.1)' : 'rgba(255,255,255,0.08)';

    const chartWidth = mainContainer.clientWidth || 800;
    const chartHeight = mainContainer.clientHeight || 450;
    const subHeight = subContainer.clientHeight || 120;

    let mainChart = null;
    let subChart = null;

    try {
      // Main Candlestick Chart
      mainChart = createChart(mainContainer, {
        width: chartWidth,
        height: chartHeight,
        layout: { background: { type: 'solid', color: bg }, textColor },
        grid: { vertLines: { color: gridColor }, horzLines: { color: gridColor } },
        crosshair: { mode: 1 },
        timeScale: {
          borderColor,
          rightOffset: 5,
          fixRightEdge: false,
        },
      });

      // Sub Chart (Volume & Attention)
      subChart = createChart(subContainer, {
        width: chartWidth,
        height: subHeight,
        layout: { background: { type: 'solid', color: bg }, textColor },
        grid: { vertLines: { color: gridColor, visible: false }, horzLines: { color: gridColor, visible: false } },
        crosshair: { mode: 1 },
        timeScale: {
          borderColor,
          rightOffset: 5,
          fixRightEdge: false,
        },
      });

      mainChartRef.current = mainChart;
      subChartRef.current = subChart;

      // Series 1: Candles
      candleSeriesRef.current = mainChart.addSeries(CandlestickSeries, {
        upColor: '#0ecb81',
        downColor: '#f6465d',
        borderVisible: false,
        wickUpColor: '#0ecb81',
        wickDownColor: '#f6465d',
      });

      // Series 2: Volume Histogram
      volumeSeriesRef.current = subChart.addSeries(HistogramSeries, {
        priceFormat: { type: 'volume' },
        priceScaleId: '',
      });

      // Series 3: Attention Overlay
      attnSeriesRef.current = subChart.addSeries(LineSeries, {
        color: '#f59e0b',
        lineWidth: 2,
        priceScaleId: 'right',
        title: 'Attention %',
      });

      // Series 4: Overlays (EMA, Bollinger)
      ema12SeriesRef.current = mainChart.addSeries(LineSeries, {
        color: '#38bdf8',
        lineWidth: 1.5,
        title: 'EMA12',
        visible: false,
      });

      ema26SeriesRef.current = mainChart.addSeries(LineSeries, {
        color: '#e879f9',
        lineWidth: 1.5,
        title: 'EMA26',
        visible: false,
      });

      bollUpperSeriesRef.current = mainChart.addSeries(LineSeries, {
        color: '#a855f7',
        lineWidth: 1,
        title: 'Upper BB',
        visible: false,
      });

      bollLowerSeriesRef.current = mainChart.addSeries(LineSeries, {
        color: '#a855f7',
        lineWidth: 1,
        title: 'Lower BB',
        visible: false,
      });

      // Series 5: Uncertainty Confidence Band
      upperBandSeriesRef.current = mainChart.addSeries(AreaSeries, {
        lineColor: 'transparent',
        topColor: 'rgba(56, 189, 248, 0.15)',
        bottomColor: 'rgba(56, 189, 248, 0.02)',
        lineWidth: 0,
        visible: false,
      });

      lowerBandSeriesRef.current = mainChart.addSeries(AreaSeries, {
        lineColor: 'transparent',
        topColor: 'rgba(56, 189, 248, 0.0)',
        bottomColor: 'rgba(56, 189, 248, 0.12)',
        lineWidth: 0,
        visible: false,
      });

      // Series 6: AI Forecast Line
      predLineSeriesRef.current = mainChart.addSeries(LineSeries, {
        color: '#38bdf8',
        lineWidth: 2.5,
        lineStyle: 1, // Dashed
        title: 'AI Forecast',
        visible: false,
      });

      // Initialize persistent marker wrappers once
      try {
        if (predLineSeriesRef.current) {
          forecastMarkersRef.current = createSeriesMarkers(predLineSeriesRef.current, []);
        }
      } catch (me) {
        console.warn('Could not init forecastMarkers primitive:', me);
      }

      try {
        if (candleSeriesRef.current) {
          attnMarkersRef.current = createSeriesMarkers(candleSeriesRef.current, []);
        }
      } catch (me) {
        console.warn('Could not init attnMarkers primitive:', me);
      }
    } catch (err) {
      console.error('LightweightCharts initialization error:', err);
      return;
    }

    // Crosshair Sync between Main Chart and Sub Chart (with loop prevention)
    const onMainRangeChange = (range) => {
      if (isDisposedRef.current || isSyncingRef.current || !range || !subChartRef.current) return;
      isSyncingRef.current = true;
      try {
        subChartRef.current.timeScale().setVisibleLogicalRange(range);
      } catch (_) {}
      isSyncingRef.current = false;
    };

    const onSubRangeChange = (range) => {
      if (isDisposedRef.current || isSyncingRef.current || !range || !mainChartRef.current) return;
      isSyncingRef.current = true;
      try {
        mainChartRef.current.timeScale().setVisibleLogicalRange(range);
      } catch (_) {}
      isSyncingRef.current = false;
    };

    try {
      mainChart.timeScale().subscribeVisibleLogicalRangeChange(onMainRangeChange);
      subChart.timeScale().subscribeVisibleLogicalRangeChange(onSubRangeChange);
    } catch (_) {}

    // Crosshair Hover Tooltip
    const onCrosshairMove = (param) => {
      if (isDisposedRef.current) return;
      if (!param.point || !param.time || !candleSeriesRef.current) {
        setHoverData(null);
        return;
      }
      try {
        const candle = param.seriesData.get(candleSeriesRef.current);
        if (candle) {
          setHoverData({
            time: param.time,
            open: candle.open,
            high: candle.high,
            low: candle.low,
            close: candle.close,
          });
        }
      } catch (_) {}
    };

    try {
      mainChart.subscribeCrosshairMove(onCrosshairMove);
    } catch (_) {}

    // Resize observer with safe boundary checks
    const resizeObserver = new ResizeObserver((entries) => {
      if (isDisposedRef.current) return;
      for (const entry of entries) {
        if (entry.target === mainContainer && mainChartRef.current) {
          const w = mainContainer.clientWidth;
          const h = mainContainer.clientHeight;
          if (w > 0 && h > 0) {
            try {
              mainChartRef.current.resize(w, h);
            } catch (_) {}
          }
        }
        if (entry.target === subContainer && subChartRef.current) {
          const w = subContainer.clientWidth;
          const h = subContainer.clientHeight;
          if (w > 0 && h > 0) {
            try {
              subChartRef.current.resize(w, h);
            } catch (_) {}
          }
        }
      }
    });

    try {
      resizeObserver.observe(mainContainer);
      resizeObserver.observe(subContainer);
    } catch (_) {}

    return () => {
      isDisposedRef.current = true;
      try {
        resizeObserver.disconnect();
      } catch (_) {}

      // Safely unsubscribe handlers before removal
      try {
        mainChart?.timeScale().unsubscribeVisibleLogicalRangeChange(onMainRangeChange);
      } catch (_) {}
      try {
        subChart?.timeScale().unsubscribeVisibleLogicalRangeChange(onSubRangeChange);
      } catch (_) {}
      try {
        mainChart?.unsubscribeCrosshairMove(onCrosshairMove);
      } catch (_) {}

      // Detach marker primitives safely
      if (forecastMarkersRef.current) {
        try {
          forecastMarkersRef.current.detach();
        } catch (_) {}
        forecastMarkersRef.current = null;
      }
      if (attnMarkersRef.current) {
        try {
          attnMarkersRef.current.detach();
        } catch (_) {}
        attnMarkersRef.current = null;
      }

      // Null out series refs before removing charts
      candleSeriesRef.current = null;
      volumeSeriesRef.current = null;
      attnSeriesRef.current = null;
      ema12SeriesRef.current = null;
      ema26SeriesRef.current = null;
      bollUpperSeriesRef.current = null;
      bollLowerSeriesRef.current = null;
      predLineSeriesRef.current = null;
      upperBandSeriesRef.current = null;
      lowerBandSeriesRef.current = null;

      const m = mainChartRef.current;
      const s = subChartRef.current;
      mainChartRef.current = null;
      subChartRef.current = null;

      try {
        if (m) m.remove();
      } catch (_) {}
      try {
        if (s) s.remove();
      } catch (_) {}
    };
  }, []);

  // =========================================================================
  // 2. Data Updates (Candles, Volume & Time Scale Right Offset)
  // =========================================================================
  useEffect(() => {
    if (isDisposedRef.current || !candleSeriesRef.current || !volumeSeriesRef.current) return;
    try {
      if (!chartData || chartData.length === 0) {
        candleSeriesRef.current?.setData([]);
        volumeSeriesRef.current?.setData([]);
        return;
      }

      candleSeriesRef.current.setData(chartData);

      const volData = chartData.map((d) => ({
        time: d.time,
        value: d.volume,
        color: d.close >= d.open ? 'rgba(14, 203, 129, 0.45)' : 'rgba(246, 70, 93, 0.45)',
      }));
      volumeSeriesRef.current.setData(volData);

      const rightOffset = forecastEnabled && predictions.length ? predictions.length + 3 : 2;
      mainChartRef.current?.timeScale().applyOptions({ rightOffset });
      subChartRef.current?.timeScale().applyOptions({ rightOffset });
    } catch (err) {
      console.warn('Guarded chart data update error:', err);
    }
  }, [chartData, forecastEnabled, predictions.length]);

  // =========================================================================
  // 2b. Timeframe Range Zooming (TradingView style: zooms visible window, keeps all history)
  // =========================================================================
  useEffect(() => {
    if (isDisposedRef.current || !mainChartRef.current || !chartData || chartData.length === 0) return;
    const rafId = requestAnimationFrame(() => {
      try {
        if (isDisposedRef.current || !mainChartRef.current) return;
        const total = chartData.length;
        const count = TIMEFRAME_BAR_COUNT[timeframe] || 252;
        const rightOffset = forecastEnabled && predictions.length ? predictions.length + 3 : 2;

        if (timeframe === 'All' || count >= total) {
          mainChartRef.current.timeScale().fitContent();
          subChartRef.current?.timeScale().fitContent();
        } else {
          const from = Math.max(0, total - count);
          const to = total + rightOffset;
          mainChartRef.current.timeScale().setVisibleLogicalRange({ from, to });
          subChartRef.current?.timeScale().setVisibleLogicalRange({ from, to });
        }
      } catch (err) {
        console.warn('Guarded timeframe zoom error:', err);
      }
    });
    return () => cancelAnimationFrame(rafId);
  }, [timeframe, chartData.length, forecastEnabled, predictions.length]);

  // =========================================================================
  // 3. Technical Indicator Overlays (EMA, Bollinger Bands)
  // =========================================================================
  useEffect(() => {
    if (isDisposedRef.current) return;
    try {
      // EMA12
      if (ema12SeriesRef.current) {
        const active = activeIndicators.has('ema12') && indicatorData?.ema12?.length > 0;
        if (active) {
          ema12SeriesRef.current.setData(indicatorData.ema12);
          ema12SeriesRef.current.applyOptions({ visible: true });
        } else {
          ema12SeriesRef.current.applyOptions({ visible: false });
        }
      }

      // EMA26
      if (ema26SeriesRef.current) {
        const active = activeIndicators.has('ema26') && indicatorData?.ema26?.length > 0;
        if (active) {
          ema26SeriesRef.current.setData(indicatorData.ema26);
          ema26SeriesRef.current.applyOptions({ visible: true });
        } else {
          ema26SeriesRef.current.applyOptions({ visible: false });
        }
      }

      // Bollinger Bands
      if (bollUpperSeriesRef.current && bollLowerSeriesRef.current) {
        const active =
          activeIndicators.has('boll') &&
          indicatorData?.bollUpper?.length > 0 &&
          indicatorData?.bollLower?.length > 0;
        if (active) {
          bollUpperSeriesRef.current.setData(indicatorData.bollUpper);
          bollLowerSeriesRef.current.setData(indicatorData.bollLower);
          bollUpperSeriesRef.current.applyOptions({ visible: true });
          bollLowerSeriesRef.current.applyOptions({ visible: true });
        } else {
          bollUpperSeriesRef.current.applyOptions({ visible: false });
          bollLowerSeriesRef.current.applyOptions({ visible: false });
        }
      }
    } catch (err) {
      console.warn('Guarded indicator update error:', err);
    }
  }, [activeIndicators, indicatorData]);

  // =========================================================================
  // 4. AI Forecast & Uncertainty Band Overlays
  // =========================================================================
  useEffect(() => {
    if (
      isDisposedRef.current ||
      !predLineSeriesRef.current ||
      !upperBandSeriesRef.current ||
      !lowerBandSeriesRef.current
    ) {
      return;
    }

    try {
      const lastCandle = chartData && chartData.length > 0 ? chartData[chartData.length - 1] : null;

      if (forecastEnabled && predictions.length > 0 && lastCandle) {
        const futureDates = getNextBusinessDays(lastCandle.time, predictions.length);
        const predPoints = [
          { time: lastCandle.time, value: lastCandle.close },
          ...predictions.map((p, i) => ({
            time: futureDates[i],
            value: Number(p.predicted_price) * 1000,
          })),
        ];

        predLineSeriesRef.current.setData(predPoints);
        predLineSeriesRef.current.applyOptions({ visible: true });

        if (confidenceBandEnabled) {
          const baseBandPct = 0.012; // 1.2% per sqrt(day)
          const upperPoints = [
            { time: lastCandle.time, value: lastCandle.close },
            ...predictions.map((p, i) => ({
              time: futureDates[i],
              value: Number(p.predicted_price) * 1000 * (1 + baseBandPct * Math.sqrt(i + 1)),
            })),
          ];
          const lowerPoints = [
            { time: lastCandle.time, value: lastCandle.close },
            ...predictions.map((p, i) => ({
              time: futureDates[i],
              value: Number(p.predicted_price) * 1000 * (1 - baseBandPct * Math.sqrt(i + 1)),
            })),
          ];
          upperBandSeriesRef.current.setData(upperPoints);
          lowerBandSeriesRef.current.setData(lowerPoints);
          upperBandSeriesRef.current.applyOptions({ visible: true });
          lowerBandSeriesRef.current.applyOptions({ visible: true });
        } else {
          upperBandSeriesRef.current.applyOptions({ visible: false });
          lowerBandSeriesRef.current.applyOptions({ visible: false });
        }

        // Forecast Horizon node markers (T+1, T+3, T+5)
        const forecastMarkers = [];
        [0, 2, 4].forEach((idx) => {
          if (idx < predictions.length && futureDates[idx]) {
            const pred = predictions[idx];
            const pVal = Number(pred.predicted_price) * 1000;
            const diffPct = ((pVal - lastCandle.close) / lastCandle.close) * 100;
            const sign = diffPct >= 0 ? '+' : '';
            const color = diffPct >= 0 ? '#0ecb81' : '#f6465d';
            forecastMarkers.push({
              time: futureDates[idx],
              position: diffPct >= 0 ? 'aboveBar' : 'belowBar',
              color,
              shape: diffPct >= 0 ? 'arrowUp' : 'arrowDown',
              text: `${pred.day} (${sign}${diffPct.toFixed(1)}%)`,
              size: 1,
            });
          }
        });

        if (forecastMarkersRef.current) {
          try {
            forecastMarkersRef.current.setMarkers(forecastMarkers);
          } catch (_) {}
        }
      } else {
        predLineSeriesRef.current?.applyOptions({ visible: false });
        upperBandSeriesRef.current?.applyOptions({ visible: false });
        lowerBandSeriesRef.current?.applyOptions({ visible: false });
        if (forecastMarkersRef.current) {
          try {
            forecastMarkersRef.current.setMarkers([]);
          } catch (_) {}
        }
      }
    } catch (err) {
      console.warn('Guarded forecast update error:', err);
    }
  }, [predictions, forecastEnabled, confidenceBandEnabled, chartData]);

  // =========================================================================
  // 5. Attention Overlay & Peak Markers
  // =========================================================================
  useEffect(() => {
    if (isDisposedRef.current || !attnSeriesRef.current) return;

    try {
      if (attentionWeights && attentionWeights.length > 0) {
        attnSeriesRef.current.setData(
          attentionWeights.map((a) => ({
            time: a.time,
            value: parseFloat((a.weight * 100).toFixed(2)),
          }))
        );
        attnSeriesRef.current.applyOptions({ visible: true });

        if (candleSeriesRef.current && chartData && chartData.length > 0 && attnMarkersRef.current) {
          const sortedAttn = [...attentionWeights].sort((a, b) => b.weight - a.weight);
          const top3Dates = new Set(sortedAttn.slice(0, 3).map((a) => a.time));
          const attnMarkers = chartData
            .filter((d) => top3Dates.has(d.time))
            .map((d) => ({
              time: d.time,
              position: 'aboveBar',
              color: '#f59e0b',
              shape: 'circle',
              text: 'Attn Peak',
              size: 1,
            }));

          try {
            attnMarkersRef.current.setMarkers(attnMarkers);
          } catch (_) {}
        }
      } else {
        attnSeriesRef.current?.setData([]);
        if (attnMarkersRef.current) {
          try {
            attnMarkersRef.current.setMarkers([]);
          } catch (_) {}
        }
      }
    } catch (err) {
      console.warn('Guarded attention update error:', err);
    }
  }, [attentionWeights, chartData]);

  // =========================================================================
  // 6. Dynamic Theme Application (No teardown needed)
  // =========================================================================
  useEffect(() => {
    if (isDisposedRef.current || !mainChartRef.current || !subChartRef.current) return;
    try {
      const bg = isLightTheme ? '#ffffff' : '#07090e';
      const textColor = isLightTheme ? '#1e293b' : '#94a3b8';
      const gridColor = isLightTheme ? 'rgba(0,0,0,0.04)' : 'rgba(255,255,255,0.03)';
      const borderColor = isLightTheme ? 'rgba(0,0,0,0.1)' : 'rgba(255,255,255,0.08)';

      mainChartRef.current.applyOptions({
        layout: { background: { type: 'solid', color: bg }, textColor },
        grid: { vertLines: { color: gridColor }, horzLines: { color: gridColor } },
        timeScale: { borderColor },
      });

      subChartRef.current.applyOptions({
        layout: { background: { type: 'solid', color: bg }, textColor },
        grid: { vertLines: { color: gridColor, visible: false }, horzLines: { color: gridColor, visible: false } },
        timeScale: { borderColor },
      });
    } catch (err) {
      console.warn('Guarded theme update error:', err);
    }
  }, [isLightTheme]);

  return (
    <div className="terminal-chart-area">
      {/* Floating Corner Tooltip */}
      {hoverData && (
        <div className="terminal-floating-tooltip">
          <div className="terminal-tooltip-item">
            <span>Ngày:</span> <strong>{hoverData.time}</strong>
          </div>
          <div className="terminal-tooltip-item">
            <span>Mở:</span> <strong>{formatVND(hoverData.open)}</strong>
          </div>
          <div className="terminal-tooltip-item">
            <span>Cao:</span> <strong>{formatVND(hoverData.high)}</strong>
          </div>
          <div className="terminal-tooltip-item">
            <span>Thấp:</span> <strong>{formatVND(hoverData.low)}</strong>
          </div>
          <div className="terminal-tooltip-item">
            <span>Đóng:</span>{' '}
            <strong style={{ color: hoverData.close >= hoverData.open ? '#0ecb81' : '#f6465d' }}>
              {formatVND(hoverData.close)}
            </strong>
          </div>
        </div>
      )}

      {/* Main Candlestick Container */}
      <div className="terminal-chart-container" style={{ position: 'relative' }}>
        <div ref={chartContainerRef} style={{ width: '100%', height: '100%' }} />

        {(!chartData || chartData.length === 0) && (
          <div
            style={{
              position: 'absolute',
              inset: 0,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'var(--terminal-text-muted)',
              fontSize: '13px',
              background: isLightTheme ? 'rgba(255,255,255,0.85)' : 'rgba(7,9,14,0.85)',
              zIndex: 4,
            }}
          >
            Đang khởi tạo biểu đồ và đồng bộ dữ liệu...
          </div>
        )}
      </div>

      {/* Synchronized Volume & Attention Sub-chart */}
      <div className="terminal-subchart-container">
        <div ref={subContainerRef} style={{ width: '100%', height: '100%' }} />
      </div>
    </div>
  );
}
