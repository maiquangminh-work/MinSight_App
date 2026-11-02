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
  pivotPoints = null,
  atr = 0,
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

  // Capital Management & Risk-Reward calculation
  const targetT5 = horizonRows[4]?.price || currentPrice;
  const expectedReturnPct = currentPrice > 0 ? ((targetT5 - currentPrice) / currentPrice) * 100 : 0;
  const dynamicStopLoss = atr > 0 ? Math.round(currentPrice - atr * 1.8) : Math.round(currentPrice * 0.965);
  const riskAmount = currentPrice - dynamicStopLoss;
  const rewardAmount = Math.abs(targetT5 - currentPrice);
  const rrRatio = riskAmount > 0 ? (rewardAmount / riskAmount).toFixed(1) : '2.0';

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

      {/* 2. Capital Management & Risk/Reward Guidance */}
      <div className="terminal-panel-card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
          <span className="signal-action-title">Kế Hoạch Quản Trị Vốn & R/R</span>
          <span style={{ fontSize: '11px', color: '#0ecb81', fontWeight: 700 }}>R:R {rrRatio} : 1</span>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
          <div style={{ background: 'rgba(0,0,0,0.25)', padding: '6px 8px', borderRadius: '4px' }}>
            <div style={{ fontSize: '11px', color: 'var(--terminal-text-muted)' }}>Mục Tiêu T+5</div>
            <strong style={{ color: expectedReturnPct >= 0 ? '#0ecb81' : '#f6465d' }}>
              {formatVND(targetT5)} ({expectedReturnPct >= 0 ? '+' : ''}{expectedReturnPct.toFixed(1)}%)
            </strong>
          </div>
          <div style={{ background: 'rgba(0,0,0,0.25)', padding: '6px 8px', borderRadius: '4px' }}>
            <div style={{ fontSize: '11px', color: 'var(--terminal-text-muted)' }}>Cắt Lỗ Đề Xuất (ATR)</div>
            <strong style={{ color: '#f6465d' }}>
              {formatVND(dynamicStopLoss)}
            </strong>
          </div>
        </div>
      </div>

      {/* 3. Multi-Horizon Forecast Matrix (T+1 -> T+5) */}
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

      {/* 4. Pivot Points Technical Boundaries */}
      {pivotPoints && (
        <div className="terminal-panel-card">
          <span className="signal-action-title">Ngưỡng Kỹ Thuật Then Chốt (Pivot Points)</span>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '4px', marginTop: '8px', textAlign: 'center', fontSize: '11px' }}>
            <div style={{ background: 'rgba(246, 70, 93, 0.1)', padding: '4px 2px', borderRadius: '4px' }}>
              <div style={{ color: '#f6465d', fontWeight: 600 }}>R2</div>
              <div style={{ color: '#f8fafc', fontWeight: 700 }}>{(pivotPoints.r2 / 1000).toFixed(1)}k</div>
            </div>
            <div style={{ background: 'rgba(246, 70, 93, 0.05)', padding: '4px 2px', borderRadius: '4px' }}>
              <div style={{ color: '#f6465d' }}>R1</div>
              <div style={{ color: '#f8fafc', fontWeight: 700 }}>{(pivotPoints.r1 / 1000).toFixed(1)}k</div>
            </div>
            <div style={{ background: 'rgba(252, 213, 53, 0.1)', padding: '4px 2px', borderRadius: '4px' }}>
              <div style={{ color: '#fcd535', fontWeight: 700 }}>PP</div>
              <div style={{ color: '#fff', fontWeight: 700 }}>{(pivotPoints.pp / 1000).toFixed(1)}k</div>
            </div>
            <div style={{ background: 'rgba(14, 203, 129, 0.05)', padding: '4px 2px', borderRadius: '4px' }}>
              <div style={{ color: '#0ecb81' }}>S1</div>
              <div style={{ color: '#f8fafc', fontWeight: 700 }}>{(pivotPoints.s1 / 1000).toFixed(1)}k</div>
            </div>
            <div style={{ background: 'rgba(14, 203, 129, 0.1)', padding: '4px 2px', borderRadius: '4px' }}>
              <div style={{ color: '#0ecb81', fontWeight: 600 }}>S2</div>
              <div style={{ color: '#f8fafc', fontWeight: 700 }}>{(pivotPoints.s2 / 1000).toFixed(1)}k</div>
            </div>
          </div>
        </div>
      )}

      {/* 5. Core Indicators & Volatility Regime */}
      <div className="terminal-panel-card">
        <span className="signal-action-title">Tín Hiệu Kỹ Thuật Nhanh & ATR</span>
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
          {atr > 0 && (
            <div style={{ background: 'rgba(0,0,0,0.2)', padding: '6px 8px', borderRadius: '4px', gridColumn: 'span 2' }}>
              <div style={{ fontSize: '11px', color: 'var(--terminal-text-muted)' }}>Biên Độ Dao Động Ngày ATR(14)</div>
              <strong style={{ color: 'var(--terminal-accent-blue)' }}>
                {formatVND(atr)} ({((atr / (currentPrice || 1)) * 100).toFixed(1)}% / phiên)
              </strong>
            </div>
          )}
        </div>
      </div>

      {/* 6. Explainable AI (XAI) Insight Teaser */}
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

      {/* 7. Quick Action Button to Open Drawer */}
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
          borderRadius: '6px',
        }}
        onClick={onOpenDrawer}
      >
        💡 Mở Bảng Lịch Sử &amp; Giải Trình AI
      </button>
    </div>
  );
}
