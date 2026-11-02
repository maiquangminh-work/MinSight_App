"""
MinSight Banking AI - Batch Training for All 10 Listed Vietnamese Commercial Banks.
Trains the Full System (Enhanced Multi-Task + 30 Banking Domain Features) for:
- Core Research Focus: VCB, BID, CTG
- Demo Expansion: MBB, TCB, VPB, ACB, HDB, SHB, VIB
"""

import os
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.banking_features import TICKERS
from src.model.ablation import evaluate_model_m3

def train_all_ten_banks(horizon_days=5, epochs=45):
    print("\n" + "=" * 80)
    print(f"MINSIGHT BANKING AI - BATCH TRAINING 10 NGÂN HÀNG VIỆT NAM (H={horizon_days})")
    print(f"Danh sách mã: {', '.join(TICKERS)}")
    print("=" * 80)

    summary = []
    total_start = time.time()

    for idx, ticker in enumerate(TICKERS, start=1):
        print(f"\n[{idx}/10] >>> Bắt đầu huấn luyện {ticker} (H={horizon_days})...")
        t0 = time.time()
        try:
            res = evaluate_model_m3(ticker, horizon_days=horizon_days, epochs=epochs, save_weights=True)
            elapsed = time.time() - t0
            print(f"  ✓ {ticker}: DA_bin={res['binary_da']:.2f}% | DA_3c={res['multi_class_da']:.2f}% | MAE={res['mae']:.4f} | Expectancy={res['expectancy']:+.2f}% | {elapsed:.1f}s")
            summary.append({
                "ticker": ticker,
                "binary_da": res["binary_da"],
                "multi_class_da": res["multi_class_da"],
                "mae": res["mae"],
                "expectancy": res["expectancy"],
                "win_rate": res["win_rate"],
                "status": "SUCCESS",
            })
        except Exception as e:
            print(f"  ✗ Lỗi khi huấn luyện {ticker}: {e}")
            summary.append({
                "ticker": ticker,
                "status": f"FAILED: {e}",
            })

    total_time = time.time() - total_start
    print("\n" + "=" * 80)
    print(f"TỔNG KẾT BATCH TRAINING 10 NGÂN HÀNG (Tổng thời gian: {total_time:.1f}s):")
    print("=" * 80)
    print(f"{'Mã CK':<8} | {'Binary DA':<12} | {'3-Class DA':<12} | {'MAE':<10} | {'Expectancy':<14} | {'Win Rate':<10}")
    print("-" * 75)
    for s in summary:
        if s["status"] == "SUCCESS":
            print(f"{s['ticker']:<8} | {s['binary_da']:>10.2f}% | {s['multi_class_da']:>10.2f}% | {s['mae']:>10.4f} | {s['expectancy']:>+12.2f}% | {s['win_rate']:>8.1f}%")
        else:
            print(f"{s['ticker']:<8} | {s['status']}")
    print("=" * 80)

if __name__ == "__main__":
    h = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    ep = int(sys.argv[2]) if len(sys.argv) > 2 else 45
    train_all_ten_banks(horizon_days=h, epochs=ep)
