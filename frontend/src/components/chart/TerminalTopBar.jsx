import React from 'react';
import { useNavigate } from 'react-router-dom';
import { formatVND, formatPercent } from '../../utils/formatting';

export default function TerminalTopBar({
  ticker,
  bankName = '',
  currentPrice = 0,
  dailyChange = 0,
  dailyChangePct = 0,
  timeframe = '1Y',
  onSelectTimeframe,
  activeIndicators = new Set(),
  onToggleIndicator,
  forecastEnabled = true,
  onToggleForecast,
  bandEnabled = true,
  onToggleBand,
  isChartExpanded = false,
  onToggleExpand,
}) {
  const navigate = useNavigate();
  const isUp = dailyChange >= 0;
  const changeColor = isUp ? '#0ecb81' : '#f6465d';
  const sign = isUp ? '+' : '';

  return (
    <div className="terminal-topbar">
      {/* 1. Left: Ticker Identity & Live Price */}
      <div className="terminal-ticker-group">
        <div className="terminal-ticker-pill">
          <span className="terminal-ticker-symbol">{ticker}</span>
          <span className="terminal-ticker-name" title={bankName}>
            {bankName || 'Ngân hàng TMCP'}
          </span>
        </div>

        <div className="terminal-price-badge">
          <span className="terminal-price-current" style={{ color: changeColor }}>
            {formatVND(currentPrice)}
          </span>
          <span className="terminal-price-change" style={{ color: changeColor }}>
            {sign}{formatVND(dailyChange)} ({sign}{formatPercent(dailyChangePct, 2)})
          </span>
        </div>
      </div>

      {/* 2. Center: Timeframe & Overlays Controls */}
      <div className="terminal-controls">
        {/* Timeframe selector (TradingView style range zooming) */}
        <div className="terminal-button-group">
          {['1M', '3M', '6M', '1Y', 'All'].map((tf) => (
            <button
              key={tf}
              type="button"
              className={`terminal-btn ${timeframe === tf ? 'active' : ''}`}
              onClick={() => onSelectTimeframe(tf)}
              title={`Phóng to khung thời gian ${tf} (giữ nguyên lịch sử để cuộn lùi)`}
            >
              {tf}
            </button>
          ))}
        </div>

        {/* Indicators Dropdown/Pills */}
        <div className="terminal-button-group">
          <button
            type="button"
            className={`terminal-btn ${activeIndicators.has('ema12') ? 'active' : ''}`}
            onClick={() => onToggleIndicator('ema12')}
          >
            EMA12
          </button>
          <button
            type="button"
            className={`terminal-btn ${activeIndicators.has('ema26') ? 'active' : ''}`}
            onClick={() => onToggleIndicator('ema26')}
          >
            EMA26
          </button>
          <button
            type="button"
            className={`terminal-btn ${activeIndicators.has('boll') ? 'active' : ''}`}
            onClick={() => onToggleIndicator('boll')}
          >
            Bollinger
          </button>
        </div>

        {/* AI Forecast & Uncertainty Band Toggles */}
        <div className="terminal-button-group">
          <button
            type="button"
            className={`terminal-btn ${forecastEnabled ? 'ai-active' : ''}`}
            onClick={onToggleForecast}
            title="Bật/Tắt đường dự báo AI từ CNN-LSTM-Attention"
          >
            🔮 AI Dự Báo (H1-H5)
          </button>
          <button
            type="button"
            className={`terminal-btn ${bandEnabled ? 'active' : ''}`}
            onClick={onToggleBand}
            title="Bật/Tắt dải biên độ bất định (Uncertainty Confidence Band)"
          >
            Dải Sai Số
          </button>
        </div>
      </div>

      {/* 3. Right: Fullscreen / Expand & Quick Navigation */}
      <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
        {onToggleExpand && (
          <button
            type="button"
            className={`terminal-btn ${isChartExpanded ? 'active' : ''}`}
            onClick={onToggleExpand}
            title={isChartExpanded ? 'Khôi phục cột phân tích bên phải' : 'Mở rộng biểu đồ toàn màn hình (ẩn cột bên phải)'}
            style={{ display: 'flex', alignItems: 'center', gap: '4px' }}
          >
            <span>{isChartExpanded ? '🗗' : '⛶'}</span>
            <span>{isChartExpanded ? 'Thu nhỏ' : 'Toàn màn hình'}</span>
          </button>
        )}

        <button
          type="button"
          className="terminal-btn"
          onClick={() => navigate(`/profile/${ticker}`)}
          title="Xem Hồ sơ doanh nghiệp & Room ngoại"
        >
          🏢 Hồ Sơ
        </button>
        <button
          type="button"
          className="terminal-btn"
          onClick={() => navigate(`/news`)}
          title="Xem Tin tức vĩ mô ngành ngân hàng"
        >
          📰 Tin Tức
        </button>
      </div>
    </div>
  );
}
