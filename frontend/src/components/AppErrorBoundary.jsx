import React from 'react';

class AppErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('AppErrorBoundary caught an error:', error, errorInfo);
    this.setState({ errorInfo });
  }

  render() {
    if (this.state.hasError) {
      return (
        <div
          style={{
            minHeight: '100vh',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '24px',
            background: '#0b0e11',
            color: '#eaecef',
            fontFamily: '"Segoe UI Variable Text", "Segoe UI Variable", "Segoe UI", "Inter", "Noto Sans", Arial, sans-serif',
          }}
        >
          <div
            style={{
              width: 'min(760px, 100%)',
              background: '#161a1e',
              border: '1px solid #2b3139',
              borderRadius: '16px',
              padding: '24px',
              boxShadow: '0 18px 40px rgba(0, 0, 0, 0.22)',
            }}
          >
            <div
              style={{
                color: '#fcd535',
                fontSize: '12px',
                fontWeight: 700,
                letterSpacing: '1px',
                textTransform: 'uppercase',
                marginBottom: '10px',
              }}
            >
              Frontend Error
            </div>
            <h1 style={{ margin: 0, fontSize: '28px', marginBottom: '12px' }}>
              Website gặp lỗi khi render giao diện
            </h1>
            <p style={{ color: '#aab5c4', lineHeight: 1.7, marginBottom: '16px' }}>
              Hệ thống đã chặn lỗi để bảo vệ ứng dụng. Phần dưới là chi tiết lỗi kỹ thuật:
            </p>
            <div
              style={{
                background: '#0f1318',
                border: '1px solid #2b3139',
                borderRadius: '12px',
                padding: '14px 16px',
                color: '#f8fafc',
                fontFamily: 'Consolas, monospace',
                fontSize: '13px',
                lineHeight: 1.7,
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
                marginBottom: '20px',
                maxHeight: '360px',
                overflowY: 'auto',
              }}
            >
              <div style={{ color: '#f87171', fontWeight: 700, marginBottom: '8px' }}>
                {String(this.state.error?.message || this.state.error || 'Unknown render error')}
              </div>
              {this.state.error?.stack && (
                <div style={{ color: '#94a3b8', fontSize: '12px', marginBottom: '8px' }}>
                  {this.state.error.stack}
                </div>
              )}
              {this.state.errorInfo?.componentStack && (
                <div style={{ color: '#64748b', fontSize: '11px', borderTop: '1px solid #1e293b', paddingTop: '8px' }}>
                  {this.state.errorInfo.componentStack}
                </div>
              )}
            </div>

            <div style={{ display: 'flex', gap: '12px' }}>
              <button
                type="button"
                onClick={() => {
                  this.setState({ hasError: false, error: null });
                  window.location.reload();
                }}
                style={{
                  background: '#fcd535',
                  color: '#0b0e11',
                  border: 'none',
                  borderRadius: '8px',
                  padding: '10px 20px',
                  fontSize: '14px',
                  fontWeight: 600,
                  cursor: 'pointer',
                }}
              >
                🔄 Tải Lại Trang
              </button>
              <button
                type="button"
                onClick={() => {
                  this.setState({ hasError: false, error: null });
                  window.location.href = '/';
                }}
                style={{
                  background: 'rgba(255, 255, 255, 0.08)',
                  color: '#f8fafc',
                  border: '1px solid #2b3139',
                  borderRadius: '8px',
                  padding: '10px 20px',
                  fontSize: '14px',
                  fontWeight: 500,
                  cursor: 'pointer',
                }}
              >
                Trang Chủ
              </button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default AppErrorBoundary;
