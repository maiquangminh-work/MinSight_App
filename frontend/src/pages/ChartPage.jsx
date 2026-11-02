import { useEffect, useMemo, useState, Component } from 'react';
import { useMarketData } from '../contexts/MarketDataContext';
import { useTheme } from '../contexts/ThemeContext';
import { calcEMA, calcBollingerBands, calcPivotPoints, calcATR } from '../utils/indicators';
import { fetchSignalHistory } from '../api/performance';
import { fetchProfile } from '../api/profile';
import { fetchNews } from '../api/market';
import { exportChartDataCSV } from '../utils/export';
import OfflineFallbackPanel from '../components/ui/OfflineFallbackPanel';

// Modern Quant Terminal Components
import '../styles/terminal.css';
import TerminalTopBar from '../components/chart/TerminalTopBar';
import TradingViewChart from '../components/chart/TradingViewChart';
import QuantDecisionPanel from '../components/chart/QuantDecisionPanel';
import DrawerPanel from '../components/chart/DrawerPanel';

class ChartErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }
  static getDerivedStateFromError() {
    return { hasError: true };
  }
  componentDidCatch(err) {
    console.warn('Chart error caught by ChartErrorBoundary:', err);
  }
  render() {
    if (this.state.hasError) {
      return (
        <div
          className="terminal-chart-area"
          style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#94a3b8',
            gap: '12px',
            background: 'var(--terminal-bg, #07090e)',
            borderRadius: '12px',
            padding: '24px',
          }}
        >
          <div style={{ fontSize: '14px', fontWeight: 600 }}>Biểu đồ đang được tự động khôi phục</div>
          <button
            type="button"
            onClick={() => this.setState({ hasError: false })}
            style={{
              padding: '8px 16px',
              background: '#38bdf8',
              color: '#07090e',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer',
              fontWeight: 600,
            }}
          >
            Tải lại biểu đồ
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

const BANK_NAMES = {
  VCB: 'Ngân hàng TMCP Ngoại Thương Việt Nam (Vietcombank)',
  BID: 'Ngân hàng TMCP Đầu Tư & Phát Triển Việt Nam (BIDV)',
  CTG: 'Ngân hàng TMCP Công Thương Việt Nam (VietinBank)',
  MBB: 'Ngân hàng TMCP Quân Đội (MB Bank)',
  TCB: 'Ngân hàng TMCP Kỹ Thương Việt Nam (Techcombank)',
  VPB: 'Ngân hàng TMCP Việt Nam Thịnh Vượng (VPBank)',
  ACB: 'Ngân hàng TMCP Á Châu (ACB)',
  HDB: 'Ngân hàng TMCP Phát Triển TP.HCM (HDBank)',
  SHB: 'Ngân hàng TMCP Sài Gòn - Hà Nội (SHB)',
  VIB: 'Ngân hàng TMCP Quốc Tế Việt Nam (VIB)',
};

export default function ChartPage() {
  const { ticker, data, loading, error } = useMarketData();
  const { isLightTheme } = useTheme();

  // Toolbar & Display States
  const [timeframe, setTimeframe] = useState('1Y');
  const [activeIndicators, setActiveIndicators] = useState(new Set(['ema12']));
  const [forecastEnabled, setForecastEnabled] = useState(true);
  const [bandEnabled, setBandEnabled] = useState(true);
  const [isChartExpanded, setIsChartExpanded] = useState(false);
  const [isDrawerOpen, setIsDrawerOpen] = useState(true);
  const [drawerTab, setDrawerTab] = useState('signals');

  // Async Data for Bottom Drawer & Panels
  const [signalHistory, setSignalHistory] = useState([]);
  const [profileData, setProfileData] = useState(null);
  const [newsData, setNewsData] = useState([]);

  useEffect(() => {
    let cancelled = false;

    fetchSignalHistory(ticker, 90)
      .then((res) => {
        if (!cancelled && res?.signals) {
          setSignalHistory(res.signals);
        }
      })
      .catch(() => {
        if (!cancelled) setSignalHistory([]);
      });

    fetchProfile(ticker)
      .then((res) => {
        if (!cancelled && res) {
          setProfileData(res);
        }
      })
      .catch(() => {});

    fetchNews(40)
      .then((res) => {
        if (!cancelled && Array.isArray(res)) {
          setNewsData(res);
        }
      })
      .catch(() => {});

    return () => {
      cancelled = true;
    };
  }, [ticker]);

  // Indicator toggle handler
  const handleToggleIndicator = (id) => {
    setActiveIndicators((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  // 1. Process Complete Historical Chart Data (never discarded)
  const sortedFullData = useMemo(() => {
    const raw = Array.isArray(data?.chart_data) ? data.chart_data : [];
    const uniqueMap = new Map();
    raw.forEach((item) => {
      const t = String(item.time || item.Time || '').split(' ')[0];
      if (t && t.length >= 8) {
        uniqueMap.set(t, {
          time: t,
          open: (Number(item.open) || 0) * 1000,
          high: (Number(item.high) || 0) * 1000,
          low: (Number(item.low) || 0) * 1000,
          close: (Number(item.close_winsorized || item.close) || 0) * 1000,
          volume: Number(item.volume) || 0,
          rsi_14: Number(item.rsi_14) || 50,
        });
      }
    });
    return Array.from(uniqueMap.values()).sort((a, b) => a.time.localeCompare(b.time));
  }, [data]);

  // 2. Compute Technical Indicators & Risk Boundaries on full dataset
  const indicatorData = useMemo(() => {
    if (!sortedFullData.length) return {};
    const ema12 = calcEMA(sortedFullData, 12);
    const ema26 = calcEMA(sortedFullData, 26);
    const boll = calcBollingerBands(sortedFullData, 20, 2);

    return {
      ema12,
      ema26,
      bollUpper: boll.upper,
      bollLower: boll.lower,
    };
  }, [sortedFullData]);

  const pivotPoints = useMemo(() => calcPivotPoints(sortedFullData), [sortedFullData]);
  const atr14 = useMemo(() => calcATR(sortedFullData, 14), [sortedFullData]);

  // 3. Compute Metrics for Topbar & Decision Panel
  const latestCandle = sortedFullData[sortedFullData.length - 1];
  const prevCandle = sortedFullData[sortedFullData.length - 2];
  const currentPrice = latestCandle ? latestCandle.close : Number(data?.current_price || 0) * 1000;
  const prevPrice = prevCandle ? prevCandle.close : currentPrice;
  const dailyChange = currentPrice - prevPrice;
  const dailyChangePct = prevPrice > 0 ? (dailyChange / prevPrice) * 100 : 0;

  // Average volume (20 bars)
  const recentVolumeSlice = sortedFullData.slice(-20);
  const averageRecentVolume = recentVolumeSlice.length
    ? recentVolumeSlice.reduce((sum, item) => sum + (Number(item.volume) || 0), 0) / recentVolumeSlice.length
    : 0;

  // Predictions array
  const predictions = useMemo(() => {
    return Array.isArray(data?.predictions) ? data.predictions : [];
  }, [data]);

  // Attention weights
  const attentionWeights = useMemo(() => {
    return Array.isArray(data?.attention_weights) ? data.attention_weights : [];
  }, [data]);

  // Error boundary fallback
  if (error) {
    return (
      <div className="terminal-workspace" style={{ alignItems: 'center', justifyContent: 'center' }}>
        <OfflineFallbackPanel
          ticker={ticker}
          errorMessage={error}
          onRetry={() => window.location.reload()}
          lastUpdatedTime={data?.latest_data_time}
        />
      </div>
    );
  }

  return (
    <div className="terminal-workspace">
      {/* 1. Topbar: Ticker, Live Prices, Timeframes, Overlays & Fullscreen Toggle */}
      <TerminalTopBar
        ticker={ticker}
        bankName={BANK_NAMES[ticker] || ''}
        currentPrice={currentPrice}
        dailyChange={dailyChange}
        dailyChangePct={dailyChangePct}
        timeframe={timeframe}
        onSelectTimeframe={setTimeframe}
        activeIndicators={activeIndicators}
        onToggleIndicator={handleToggleIndicator}
        forecastEnabled={forecastEnabled}
        onToggleForecast={() => setForecastEnabled((prev) => !prev)}
        bandEnabled={bandEnabled}
        onToggleBand={() => setBandEnabled((prev) => !prev)}
        isChartExpanded={isChartExpanded}
        onToggleExpand={() => setIsChartExpanded((prev) => !prev)}
      />

      {/* 2. Main Terminal Grid: Chart + Decision Panel */}
      <div className={`terminal-main-grid ${isChartExpanded ? 'expanded' : ''}`}>
        {/* Left Column: Candlestick Chart + Synced Volume & Attention Sub-pane */}
        <ChartErrorBoundary key={ticker}>
          <TradingViewChart
            chartData={sortedFullData}
            predictions={predictions}
            attentionWeights={attentionWeights}
            activeIndicators={activeIndicators}
            forecastEnabled={forecastEnabled}
            confidenceBandEnabled={bandEnabled}
            indicatorData={indicatorData}
            isLightTheme={isLightTheme}
            timeframe={timeframe}
          />
        </ChartErrorBoundary>

        {/* Right Column: Hero Decision Badge, Calibrated Confidence, Gate Status, Horizon Table */}
        {!isChartExpanded && (
          <QuantDecisionPanel
            ticker={ticker}
            currentPrice={currentPrice}
            recommendation={data?.recommendation || 'TRUNG LẬP'}
            confidenceScore={data?.recommendation_confidence_score ?? 75}
            confidenceLabel={data?.recommendation_confidence_label || 'Khá'}
            confidenceGate={data?.confidence_gate}
            predictions={predictions}
            rsi={Number(data?.rsi_14 || latestCandle?.rsi_14 || 50)}
            volume={latestCandle?.volume || 0}
            averageVolume={averageRecentVolume}
            attentionXai={data?.attention_xai}
            pivotPoints={pivotPoints}
            atr={atr14}
            onOpenDrawer={() => {
              setIsDrawerOpen(true);
              setDrawerTab('signals');
            }}
            onOpenXai={() => {
              setIsDrawerOpen(true);
              setDrawerTab('xai');
            }}
          />
        )}
      </div>

      {/* 3. Bottom Multi-Panel Analytics Dock (Open by default, 5 Tabs) */}
      <DrawerPanel
        isOpen={isDrawerOpen}
        onToggle={() => setIsDrawerOpen((prev) => !prev)}
        ticker={ticker}
        attentionWeights={attentionWeights}
        attentionXai={data?.attention_xai}
        signalHistory={signalHistory}
        chartData={sortedFullData}
        profileData={profileData || data?.profile_data}
        newsData={newsData}
        currentPrice={currentPrice}
        activeTab={drawerTab}
        onSelectTab={setDrawerTab}
        onExportCSV={() => exportChartDataCSV(ticker, sortedFullData)}
      />
    </div>
  );
}
