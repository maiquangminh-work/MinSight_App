import sys
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
import schedule
import time
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import or_
# Nạp class cốt lõi của thế hệ thư viện mới
try:
    from src.data_pipeline.vnstock_compat import Vnstock
except Exception:
    Vnstock = None 
import sys
import os

# Trỏ đường dẫn để Python nhận diện được thư mục src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.backend.database import SessionLocal, StockPrice

TICKERS = ['VCB', 'BID', 'CTG', 'MBB', 'TCB', 'VPB', 'ACB', 'HDB', 'SHB', 'VIB']
OPTIONAL_FOREIGN_COLUMNS = ['foreign_buy_volume', 'foreign_sell_volume', 'foreign_net_volume']
BACKFILL_FOREIGN_HISTORY = os.getenv('BACKFILL_FOREIGN_HISTORY', '0').strip().lower() in {'1', 'true', 'yes', 'on'}

# Hàm phụ để xử lý giá trị có thể thiếu của các cột foreign volume
def _optional_int(row, column_name):
    if column_name not in row.index:
        return None
    value = row.get(column_name)
    if pd.isna(value):
        return None
    try:
        return int(round(float(value)))
    except Exception:
        return None

# Hàm phụ để tìm bản ghi đầu tiên thiếu dữ liệu foreign volume (nếu có)
def _first_missing_foreign_record(db, ticker):
    return (
        db.query(StockPrice)
        .filter(StockPrice.ticker == ticker)
        .filter(
            or_(
                StockPrice.foreign_buy_volume.is_(None),
                StockPrice.foreign_sell_volume.is_(None),
            )
        )
        .order_by(StockPrice.time.asc())
        .first()
    )

# Hàm phụ để trích xuất lịch sử giá với khả năng xử lý linh hoạt các cột 
# foreign volume tùy theo nguồn dữ liệu
def _fetch_history_with_optional_foreign(stock, start_date, end_date):
    try:
        return stock.quote.history(
            start=start_date,
            end=end_date,
            interval='1D',
            get_all=True,
        )
    except TypeError:
        return stock.quote.history(start=start_date, end=end_date, interval='1D')

def _fetch_from_tcbs(ticker, start_date, end_date):
    """Nguồn dự phòng tự động từ Yahoo Finance và CafeF."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    # 1. Thử Yahoo Finance
    try:
        import requests
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}.VN"
        params = {"range": "10y", "interval": "1d"}
        resp = requests.get(url, params=params, headers=headers, timeout=12)
        if resp.status_code == 200:
            results = resp.json().get("chart", {}).get("result", [])
            if results:
                timestamps = results[0].get("timestamp", [])
                quotes = results[0].get("indicators", {}).get("quote", [{}])[0]
                rows = []
                for t, o, h, l, c, v in zip(timestamps, quotes.get("open", []), quotes.get("high", []), quotes.get("low", []), quotes.get("close", []), quotes.get("volume", [])):
                    if None in (t, o, h, l, c) or o <= 0:
                        continue
                    trade_date = datetime.fromtimestamp(t).strftime('%Y-%m-%d')
                    scale = 1000.0 if o > 1000 else 1.0
                    rows.append({
                        "time": trade_date,
                        "open": float(o) / scale,
                        "high": float(h) / scale,
                        "low": float(l) / scale,
                        "close": float(c) / scale,
                        "volume": int(v) if v else 1000000,
                    })
                if len(rows) > 50:
                    print(f"  [Yahoo Finance] Tải thành công {len(rows)} phiên cho {ticker}")
                    return pd.DataFrame(rows)
    except Exception:
        pass

    # 2. Thử CafeF
    try:
        import requests
        url = "https://s.cafef.vn/Ajax/PageNew/DataHistory/PriceHistory.ashx"
        params = {"Symbol": ticker, "StartDate": "01/01/2016", "EndDate": datetime.now().strftime("%d/%m/%Y"), "PageIndex": 1, "PageSize": 3000}
        resp = requests.get(url, params=params, headers=headers, timeout=12)
        if resp.status_code == 200:
            items = resp.json().get("Data", {}).get("Data", [])
            if items:
                rows = []
                for item in items:
                    ngay = item.get("Ngay", "")
                    try:
                        trade_date = datetime.strptime(ngay, "%d/%m/%Y").strftime('%Y-%m-%d')
                    except Exception:
                        continue
                    c = float(item.get("GiaDongCua", 0))
                    if c <= 0:
                        continue
                    scale = 1000.0 if c > 1000 else 1.0
                    rows.append({
                        "time": trade_date,
                        "open": float(item.get("GiaMoCua", c)) / scale,
                        "high": float(item.get("GiaCaoNhat", c)) / scale,
                        "low": float(item.get("GiaThapNhat", c)) / scale,
                        "close": c / scale,
                        "volume": int(item.get("KhoiLuongKhopLenh", 0)),
                    })
                if len(rows) > 50:
                    rows.sort(key=lambda x: x["time"])
                    print(f"  [CafeF] Tải thành công {len(rows)} phiên cho {ticker}")
                    return pd.DataFrame(rows)
    except Exception:
        pass

    return None

# Hàm chính để cập nhật dữ liệu từ API và lưu vào Database
def update_database():
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Bắt đầu tiến trình cập nhật dữ liệu...")
    db = SessionLocal()
    
    today_str = datetime.now().strftime('%Y-%m-%d')
    
    for ticker in TICKERS:
        try:
            # Tìm ngày gần nhất đã lưu trong Database
            last_record = db.query(StockPrice).filter(StockPrice.ticker == ticker).order_by(StockPrice.time.desc()).first()
            
            if last_record:
                # Nếu đã có data, chỉ kéo từ ngày tiếp theo
                next_price_date = (last_record.time + timedelta(days=1)).strftime('%Y-%m-%d')
            else:
                # Nếu database trống, kéo từ 2016
                next_price_date = '2016-01-01'
            missing_foreign_record = _first_missing_foreign_record(db, ticker) if BACKFILL_FOREIGN_HISTORY else None
            if BACKFILL_FOREIGN_HISTORY and missing_foreign_record:
                start_date = min(next_price_date, missing_foreign_record.time.strftime('%Y-%m-%d'))
            else:
                start_date = next_price_date

            if start_date > today_str:
                print(f"{ticker}: Dữ liệu đã được cập nhật mới nhất.")
                continue
                
            # Gọi vnstock API lấy dữ liệu giá
            print(f"Đang kéo dữ liệu {ticker} từ {start_date} đến {today_str}...")
            
            # Khởi tạo đối tượng mã chứng khoán (KBS là nguồn ổn định nhất hiện tại)
            stock = Vnstock().stock(symbol=ticker, source='KBS') if Vnstock is not None else None
            
            # Trích xuất dữ liệu giá lịch sử
            df = _fetch_history_with_optional_foreign(stock, start_date, today_str) if stock is not None else None
            if df is None or df.empty:
                df = _fetch_from_tcbs(ticker, start_date, today_str)
            
            if df is None or df.empty:
                print(f" {ticker}: Không có dữ liệu giao dịch mới (có thể là ngày nghỉ).")
                continue
                
            # Chuẩn hóa tên cột về chữ thường để tránh lỗi viết hoa/thường (Time vs time)
            df.columns = [col.lower() for col in df.columns]
            has_foreign_history = any(column in df.columns for column in OPTIONAL_FOREIGN_COLUMNS)
            if 'foreign_net_volume' not in df.columns and {'foreign_buy_volume', 'foreign_sell_volume'}.issubset(df.columns):
                df['foreign_net_volume'] = df['foreign_buy_volume'] - df['foreign_sell_volume']
                
            # Ghi dữ liệu mới vào Database
            inserted_count = 0
            updated_count = 0
            foreign_count = 0
            for _, row in df.iterrows():
                # Xử lý linh hoạt định dạng thời gian
                time_val = row['time']
                if isinstance(time_val, str):
                    trade_date = datetime.strptime(time_val[:10], '%Y-%m-%d').date()
                else:
                    trade_date = time_val.date() # Nếu đã là object datetime
                
                record = (
                    db.query(StockPrice)
                    .filter(StockPrice.ticker == ticker, StockPrice.time == trade_date)
                    .first()
                )
                if record is None:
                    record = StockPrice(ticker=ticker, time=trade_date)
                    db.add(record)
                    inserted_count += 1
                else:
                    updated_count += 1

                record.open = row['open']
                record.high = row['high']
                record.low = row['low']
                record.close = row['close']
                record.volume = row['volume']

                foreign_buy_volume = _optional_int(row, 'foreign_buy_volume')
                foreign_sell_volume = _optional_int(row, 'foreign_sell_volume')
                foreign_net_volume = _optional_int(row, 'foreign_net_volume')
                if foreign_net_volume is None and foreign_buy_volume is not None and foreign_sell_volume is not None:
                    foreign_net_volume = foreign_buy_volume - foreign_sell_volume

                if foreign_buy_volume is not None or foreign_sell_volume is not None or foreign_net_volume is not None:
                    record.foreign_buy_volume = foreign_buy_volume
                    record.foreign_sell_volume = foreign_sell_volume
                    record.foreign_net_volume = foreign_net_volume
                    record.foreign_data_source = 'KBS'
                    foreign_count += 1
                
            db.commit()
            if has_foreign_history:
                print(f"{ticker}: Them {inserted_count}, cap nhat {updated_count}, DTNN {foreign_count} dong.")
            else:
                print(f"{ticker}: Them {inserted_count}, cap nhat {updated_count}. Nguon khong tra DTNN lich su.")
            records = [None] * inserted_count
            print(f"{ticker}: Đã chèn thêm {len(records)} dòng vào Database.")
            
        except Exception as e:
            print(f"Lỗi khi cập nhật {ticker}: {e}")
            db.rollback()
            
    db.close()
    print("Hoàn tất tiến trình cập nhật!\n")

# Lập lịch tự động chạy vào 01:00 AM mỗi sáng
schedule.every().day.at("01:00").do(update_database)

if __name__ == "__main__":
    print("Hệ thống Tự động cập nhật Dữ liệu đã khởi động!")
    print("Đang chạy thử lần đầu tiên để đồng bộ Database...")
    update_database() 
    
    print("Đang chờ đến lịch chạy tiếp theo (01:00 AM hàng ngày)... Nhấn Ctrl+C để thoát.")
    while True:
        schedule.run_pending()
        time.sleep(60)
