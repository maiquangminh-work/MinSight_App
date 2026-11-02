"""
Setup Data Script — Thu thập dữ liệu cổ phiếu ngân hàng đầy đủ từ vnstock (2018 - 2026).
Nguồn dữ liệu độc quyền qua thư viện vnstock chuẩn.

Chạy trực tiếp:
    python scripts/setup_data.py
"""
import os
import sys
import datetime
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Trỏ đường dẫn gốc
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from src.backend.database import SessionLocal, StockPrice, Base, engine
from src.data_pipeline.preprocess import process_ticker

TICKERS = ['VCB', 'BID', 'CTG', 'MBB', 'TCB', 'VPB', 'ACB', 'HDB', 'SHB', 'VIB']

def _normalize_df(df):
    df.columns = [col.lower() for col in df.columns]
    rows = []
    for _, row in df.iterrows():
        t_val = row['time']
        if isinstance(t_val, str):
            trade_date = datetime.datetime.strptime(t_val[:10], "%Y-%m-%d").date()
        else:
            trade_date = t_val.date()
        c = float(row.get('close', 0))
        if c <= 0:
            continue
        o = float(row.get('open', c))
        h = float(row.get('high', max(o, c)))
        l = float(row.get('low', min(o, c)))
        v = int(row.get('volume', 0))
        scale = 1000.0 if c > 1000 else 1.0
        rows.append({
            "time": trade_date,
            "open": round(o / scale, 2),
            "high": round(h / scale, 2),
            "low": round(l / scale, 2),
            "close": round(c / scale, 2),
            "volume": v,
        })
    return pd.DataFrame(rows)

def fetch_data_vnstock(ticker):
    """
    Thu thập dữ liệu giá OHLCV qua vnstock theo từng năm (2018 - 2026)
    để đảm bảo đầy đủ lịch sử không bị giới hạn 365 ngày.
    """
    from vnstock import Quote
    q = Quote(symbol=ticker)
    now_year = datetime.datetime.now().year
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    
    dfs = []
    for yr in range(2018, now_year + 1):
        s = f"{yr}-01-01"
        e = f"{yr}-12-31" if yr < now_year else today_str
        try:
            part = q.history(start=s, end=e, interval='1D')
            if part is not None and not part.empty:
                dfs.append(part)
        except Exception:
            pass
            
    if not dfs:
        return None
        
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=['time'])
    print(f"  --> [vnstock] Tải thành công {len(combined)} phiên giao dịch (2018 - {now_year})!")
    return _normalize_df(combined)

def process_bank(ticker):
    print(f"\n---> Bắt đầu nạp dữ liệu cho {ticker}...")
    df = fetch_data_vnstock(ticker)
    
    if df is None or df.empty:
        print(f"  [THẤT BẠI] Không thể tải dữ liệu {ticker} qua vnstock.")
        return False
        
    db = SessionLocal()
    for _, row in df.iterrows():
        t = row['time']
        rec = db.query(StockPrice).filter(StockPrice.ticker == ticker, StockPrice.time == t).first()
        if rec is None:
            rec = StockPrice(ticker=ticker, time=t)
            db.add(rec)
        rec.open = row['open']
        rec.high = row['high']
        rec.low = row['low']
        rec.close = row['close']
        rec.volume = row['volume']
        
    db.commit()
    db.close()
    print(f"  [1/2 SQLite] Đã lưu {len(df)} phiên vào database.")
    
    # Tiền xử lý chỉ báo kỹ thuật (SMA, RSI, Winsorized Close)
    ok = process_ticker(ticker)
    if ok:
        print(f"  [2/2 Features] Đã tính toán chỉ báo và lưu CSV cho {ticker} thành công!")
        return True
    return False

def main():
    print("=" * 65)
    print("MINSIGHT BANKING AI — CẬP NHẬT DỮ LIỆU ĐỘC QUYỀN VNSTOCK (2018-2026)")
    print("Nguồn dữ liệu: vnstock")
    print("=" * 65)
    Base.metadata.create_all(bind=engine)
    
    success = 0
    for i, t in enumerate(TICKERS, 1):
        print(f"\n[{i}/{len(TICKERS)}] Xử lý mã ngân hàng {t}...")
        if process_bank(t):
            success += 1
            
    print("\n" + "=" * 65)
    print(f"KẾT QUẢ: {success}/{len(TICKERS)} mã ngân hàng đã được nạp dữ liệu từ vnstock.")
    print("=" * 65)

if __name__ == "__main__":
    main()
