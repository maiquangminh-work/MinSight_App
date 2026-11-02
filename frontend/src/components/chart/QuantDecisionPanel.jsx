import React from 'react';
import { formatVND, formatPercent } from '../../utils/formatting';

export default function QuantDecisionPanel({
  ticker,
  currentPrice = 0,
  recommendation = 'TRUNG LẬP',
  confidenceScore = 75,
  confidenceLabel = 'Cao',
  confidenceGate = null,
  predictions = [],
  rsi = 50,
  volume = 0,
  averageVolume = 0,
  attentionXai = null,
  onOpenDrawer,
  onOpenXai,
}) {
  const isBuy = recommendation === 'MUA' || recommendation === 'KHẢ QUAN';
  const isSell = recommendation === 'BÁN' || recommendation === 'KÉM KHẢ QUAN';
  const statusClass = isBuy ? 'buy' : isSell ? 'sell' : 'hold';

  const gatePassed = confidenceGate?.passed ?? true;
  const gateScore = confidenceGate?.confidence_score ? confidenceGate.confidence_score.toFixed(2) : '1.12';
  const gateThreshold = confidenceGate?.threshold ? confidenceGate.threshold.toFixed(2) : '0.40';

  // Horizon table rows (T+1 to T+5)
  const horizonRows = predictions.slice(0, 5).map((p, i) => {
    const pPrice = Number(p.predicted_price) * 1000;
    const diffPct = currentPrice > 0 ? ((pPrice - currentPrice) / currentPrice) * 100 : 0;
    const sign = diffPct >= 0 ? '+' : '';
    const color = diffPct >= 0 ? '#0ecb81' : '#f6465d';
    return {
      day: p.day || `T+${i + 1}`,
      price: pPrice,
      diffPct,
      sign,
      color,
    };
  });

  const rsiLabel = rsi >= 70 ? 'Quá mua' : rsi <= 30 ? 'Quá bán' : 'Cân bằng';
  const rsiColor = rsi >= 70 ? '#f6465d' : rsi <= 30 ? '#0ecb81' : '#848e9c';

  const topDriver = attentionXai?.peak_sessions?.[0] || null;

  return (
    <div className="terminal-decision-panel">
      {/* 1. Hero Decision Card */}
      <div className={`terminal-panel-card signal-hero-wrapper ${statusClass}`}>
        <div className="signal-action-header">
          <span className="signal-action-title">Khuyến Nghị Định Lượng</span>
          <span className={`signal-badge-pill ${statusClass}`}>{recommendation}</span>
        </div>

        {/* Confidence Gauge Bar */}
        <div className="confidence-meter mt-2">
          <div className="confidence-meter-header">
            <span style={{ color: 'var(--terminal-text-muted)' }}>Độ Tin Cậy (Calibrated):</span>
            <strong style={{ color: isBuy ? '#0ecb81' : isSell ? '#f6465d' : '#fcd535' }}>
              {Math.round(confidenceScore)}% ({confidenceLabel})
            </strong>
          </div>
          <div className="confidence-track">
            <div
              className="confidence-fill"
              style={{
                width: `${Math.min(100, Math.max(5, confidenceScore))}%`,
                backgroundColor: isBuy ? '#0ecb81' : isSell ? '#f6465d' : '#fcd535',
              }}
            />
          </div>
        </div>

        {/* Hybrid Gate Pill */}
        <div className={`hybrid-gate-pill ${gatePassed ? 'passed' : 'filtered'} mt-1`}>
          <span>{gatePassed ? '🛡️' : '⚠️'}</span>
          <div>
            <strong>{gatePassed ? 'Gate Đạt Chuẩn' : 'Gate Chờ Xác Nhận'}</strong>
            <div style={{ fontSize: '11px', color: 'var(--terminal-text-muted)' }}>
              {gatePassed
                ? `Biên độ ${gateScore}x SD vượt ngưỡng ${gateThreshold}x`
                : 'Biên độ dự báo thấp, hệ thống giữ vị thế an toàn'}
            </div>
          </div>
        </div>
      </div>

      {/* 2. Multi-Horizon Forecast Matrix (T+1 -> T+5) */}
      <div className="terminal-panel-card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span className="signal-action-title">Dự Báo Mục Tiêu (H=1..5)</span>
          <span style={{ fontSize: '11px', color: 'var(--terminal-accent-blue)' }}>CNN-LSTM</span>
        </div>

        <table className="horizon-table">
          <thead>
            <tr>
              <th>Phiên</th>
              <th style={{ textAlign: 'right' }}>Giá Dự Báo</th>
              <th style={{ textAlign: 'right' }}>Kỳ Vọng</th>
            </tr>
          </thead>
          <tbody>
            {horizonRows.length > 0 ? (
              horizonRows.map((row) => (
                <tr key={row.day}>
                  <td style={{ color: '#fff', fontWeight: 600 }}>{row.day}</td>
                  <td style={{ textAlign: 'right' }}>{formatVND(row.price)}</td>
                  <td style={{ textAlign: 'right', color: row.color, fontWeight: 600 }}>
                    {row.sign}
                    {formatPercent(row.diffPct, 2)}
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan="3" style={{ textAlign: 'center', color: 'var(--terminal-text-muted)' }}>
                  Đang cập nhật chuỗi dự báo...
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {/* 3. Core Indicators Snapshot */}
      <div className="terminal-panel-card">
        <span className="signal-action-title">Tín Hiệu Kỹ Thuật Nhanh</span>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', marginTop: '8px' }}>
          <div style={{ background: 'rgba(0,0,0,0.2)', padding: '6px 8px', borderRadius: '4px' }}>
            <div style={{ fontSize: '11px', color: 'var(--terminal-text-muted)' }}>RSI(14)</div>
            <strong style={{ color: rsiColor }}>
              {rsi.toFixed(1)} <span style={{ fontSize: '11px' }}>({rsiLabel})</span>
            </strong>
          </div>
          <div style={{ background: 'rgba(0,0,0,0.2)', padding: '6px 8px', borderRadius: '4px' }}>
            <div style={{ fontSize: '11px', color: 'var(--terminal-text-muted)' }}>Thanh Khoản / MA20</div>
            <strong>
              {averageVolume > 0 ? `${(volume / averageVolume).toFixed(1)}x` : '1.0x'}
            </strong>
          </div>
        </div>
      </div>

      {/* 4. Explainable AI (XAI) Insight Teaser */}
      {topDriver && (
        <div
          className="terminal-panel-card"
          style={{
            background: 'rgba(245, 158, 11, 0.06)',
            border: '1px solid rgba(245, 158, 11, 0.28)',
            cursor: 'pointer',
            transition: 'border-color 0.2s ease',
          }}
          onClick={onOpenXai || onOpenDrawer}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '11px', color: '#f59e0b', fontWeight: 800 }}>
              🧠 TÂM ĐIỂM ATTENTION (XAI)
            </span>
            <span
              style={{
                fontSize: '10px',
                background: 'rgba(245, 158, 11, 0.25)',
                color: '#fcd535',
                padding: '1px 6px',
                borderRadius: '4px',
                fontWeight: 700,
              }}
            >
              {topDriver.weight_pct}%
            </span>
          </div>
          <div style={{ fontSize: '12px', fontWeight: 700, color: '#f8fafc', margin: '4px 0 2px 0' }}>
            Phiên {topDriver.date}
          </div>
          <div style={{ fontSize: '11px', color: 'var(--terminal-text-muted)', lineHeight: 1.35 }}>
            {topDriver.primary_driver}
          </div>
          <div style={{ fontSize: '10px', color: 'var(--terminal-accent-blue)', marginTop: '4px', fontWeight: 600 }}>
            &rarr; Bấm để mở giải trình chi tiết 4 Heads
          </div>
        </div>
      )}

      {/* 5. Quick Action Button to Open Drawer */}
      <button
        type="button"
        className="terminal-btn active"
        style={{
          width: '100%',
          padding: '10px',
          background: 'rgba(56, 189, 248, 0.1)',
          border: '1px solid rgba(56, 189, 248, 0.3)',
          color: 'var(--terminal-accent-blue)',
          fontWeight: 600,
          marginTop: 'auto',
          borderRadius: '6px',
        }}
        onClick={onOpenDrawer}
      >
        💡 Mở Bảng Lịch Sử &amp; Giải Trình AI
      </button>
    </div>
  );
}
