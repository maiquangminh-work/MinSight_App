"""
Huấn luyện và Đánh giá Mô hình Nâng cao Enhanced Multi-Task CNN-BiLSTM-MHTA (Phase 1).

Đặc tính:
1. Target kép (Multi-Task):
   - Regression Head: Log-return magnitude (đo độ lớn biến động)
   - Direction Head: 3 classes [0: Down, 1: Flat, 2: Up] (tối ưu Directional Accuracy)
2. Loss Functions:
   - AsymmetricVarianceMatchingLoss (chống flat prediction + phạt nặng False Buy)
   - DirectionalFocalLoss (tập trung học các điểm đảo chiều khó)
3. Architecture:
   - Inception Multi-Scale Conv1D (kernels 2, 3, 5)
   - Bidirectional LSTM (2 x 32 units)
   - Multi-Head Temporal Attention (4 heads)
   - Residual Skip Connection
4. Native XAI:
   - Tự động trích xuất Attention weights 30 phiên trực tiếp từ đồ thị mạng.
"""
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.model.architecture import build_enhanced_multitask_model
from src.model.losses import AsymmetricVarianceMatchingLoss, DirectionalFocalLoss
from src.model.train import REGRESSION_FEATURE_COLUMNS, _augment_regression_features
from src.model.statistical_tests import (
    pesaran_timmermann,
    diebold_mariano,
    binomial_da_test,
)

SEED = 42
WINDOW_SIZE = 30
DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models/multitask")


def set_seed(seed=SEED):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def create_multitask_sequences(features, reg_target, dir_target, window_size=WINDOW_SIZE):
    X, y_reg, y_dir = [], [], []
    for i in range(len(features) - window_size):
        X.append(features[i : i + window_size])
        y_reg.append(reg_target[i + window_size])
        y_dir.append(dir_target[i + window_size])
    return (
        np.array(X, dtype=np.float32),
        np.array(y_reg, dtype=np.float32),
        np.array(y_dir, dtype=np.int32),
    )


def train_multitask_ticker(ticker: str = "VCB",
                           horizon_days: int = 1,
                           epochs: int = 80,
                           batch_size: int = 16,
                           direction_threshold: float = 0.003):
    """Huấn luyện mô hình Multi-Task cho một mã ngân hàng."""
    ticker = ticker.upper()
    set_seed(SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = DATA_DIR / f"{ticker}_features.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file đặc trưng: {csv_path}")

    print(f"\n=======================================================")
    print(f"[PHASE 1] Huấn luyện Enhanced Multi-Task: {ticker} (H={horizon_days})")
    print(f"=======================================================")

    df = pd.read_csv(csv_path)
    df = _augment_regression_features(df)

    features = list(REGRESSION_FEATURE_COLUMNS)

    # 1. Tạo biến mục tiêu
    if horizon_days == 1:
        raw_log_ret = np.log(df['close_winsorized'] / df['close_winsorized'].shift(1))
    else:
        raw_log_ret = np.log(df['close_winsorized'].shift(-horizon_days) / df['close_winsorized'])

    df['target_log_ret'] = raw_log_ret

    # Phân loại hướng: 0: Down (r < -th), 1: Flat (-th <= r <= th), 2: Up (r > th)
    th = float(direction_threshold)
    dir_labels = np.where(raw_log_ret < -th, 0, np.where(raw_log_ret > th, 2, 1))
    df['target_dir'] = dir_labels

    df.dropna(subset=features + ['target_log_ret', 'target_dir'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    data_values = df[features].values
    reg_values = df[['target_log_ret']].values
    dir_values = df['target_dir'].values

    n = len(df)
    train_end = int(n * 0.8)
    val_start = max(0, train_end - WINDOW_SIZE)

    train_data = data_values[:train_end]
    val_data = data_values[val_start:]

    train_reg = reg_values[:train_end]
    val_reg = reg_values[val_start:]

    train_dir = dir_values[:train_end]
    val_dir = dir_values[val_start:]

    # Scale
    f_scaler = MinMaxScaler(feature_range=(0, 1))
    t_scaler = MaxAbsScaler()

    scaled_train_data = f_scaler.fit_transform(train_data)
    scaled_train_reg = t_scaler.fit_transform(train_reg)

    scaled_val_data = f_scaler.transform(val_data)
    scaled_val_reg = t_scaler.transform(val_reg)

    # Tạo sequences
    X_train, y_reg_train, y_dir_train = create_multitask_sequences(
        scaled_train_data, scaled_train_reg, train_dir, WINDOW_SIZE
    )
    X_val, y_reg_val, y_dir_val = create_multitask_sequences(
        scaled_val_data, scaled_val_reg, val_dir, WINDOW_SIZE
    )

    print(f"  Tập Train: X={X_train.shape}, y_reg={y_reg_train.shape}, y_dir={y_dir_train.shape}")
    print(f"  Tập Val:   X={X_val.shape}, y_reg={y_reg_val.shape}, y_dir={y_dir_val.shape}")

    # Build mô hình Enhanced Multi-Task
    model = build_enhanced_multitask_model(
        input_shape=(WINDOW_SIZE, len(features)),
        num_direction_classes=3,
        asymmetric_penalty=1.75,
        learning_rate=0.001
    )

    checkpoint_path = str(MODELS_DIR / f"enhanced_mhta_{ticker.lower()}_h{horizon_days}.keras")
    callbacks = [
        EarlyStopping(monitor="val_direction_output_accuracy", patience=22, mode="max", restore_best_weights=True, verbose=1),
        ModelCheckpoint(checkpoint_path, monitor="val_direction_output_accuracy", mode="max", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-5, verbose=1),
    ]

    print("\n Bắt đầu huấn luyện Multi-Task CNN-BiLSTM-MHTA...")
    history = model.fit(
        X_train,
        {
            "direction_output": y_dir_train,
            "regression_output": y_reg_train,
        },
        validation_data=(
            X_val,
            {
                "direction_output": y_dir_val,
                "regression_output": y_reg_val,
            },
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Đánh giá trên tập Validation
    print("\n-------------------------------------------------------")
    print(f"[ĐÁNH GIÁ ĐỊNH LƯỢNG HỌC THUẬT: {ticker}]")
    print("-------------------------------------------------------")
    val_preds = model.predict(X_val, verbose=0)
    pred_dir_probs = val_preds["direction_output"]  # (n_val, 3)
    pred_dir_classes = np.argmax(pred_dir_probs, axis=-1)
    pred_reg_scaled = val_preds["regression_output"]
    pred_reg_unscaled = t_scaler.inverse_transform(pred_reg_scaled).flatten()
    actual_reg_unscaled = t_scaler.inverse_transform(y_reg_val).flatten()
    actual_dir = y_dir_val

    # 1. Directional Accuracy (DA)
    # Binary DA: Xét các phiên có xu hướng rõ ràng (loại flat hoặc sign match)
    sign_pred = np.sign(pred_reg_unscaled)
    sign_actual = np.sign(actual_reg_unscaled)
    valid_sign_mask = (sign_actual != 0)
    binary_da = np.mean(sign_pred[valid_sign_mask] == sign_actual[valid_sign_mask]) * 100

    # 3-class Directional Accuracy
    multi_class_da = np.mean(pred_dir_classes == actual_dir) * 100

    # 2. Pesaran-Timmermann Test
    pt_result = pesaran_timmermann(actual_reg_unscaled, pred_reg_unscaled)
    pt_stat = pt_result.get("pt_stat", 0.0)
    pt_p_val = pt_result.get("p_value", 1.0)

    # 3. Diebold-Mariano Test vs Naive (Predict 0)
    errors_model = actual_reg_unscaled - pred_reg_unscaled
    errors_naive = actual_reg_unscaled - 0.0
    dm_res = diebold_mariano(errors_model, errors_naive, h=horizon_days)
    dm_stat = dm_res.get("dm_stat", 0.0)
    dm_p_val = dm_res.get("p_value", 1.0)

    # 4. MAE & RMSE
    mae = np.mean(np.abs(actual_reg_unscaled - pred_reg_unscaled))
    rmse = np.sqrt(np.mean((actual_reg_unscaled - pred_reg_unscaled) ** 2))

    print(f"✓ Binary Directional Accuracy (DA):    {binary_da:.2f}% (Mục tiêu: >= 57%)")
    print(f"✓ Multi-Class Direction Accuracy:     {multi_class_da:.2f}%")
    print(f"✓ Pesaran-Timmermann Test Stat:       {pt_stat:.3f} (p-value: {pt_p_val:.4f})")
    print(f"✓ Diebold-Mariano vs Naive Baseline:  DM={dm_stat:.3f} (p-value: {dm_p_val:.4f})")
    print(f"✓ Regression MAE:                     {mae:.5f}")
    print(f"✓ Regression RMSE:                    {rmse:.5f}")

    # Lưu Scaler & Config
    config_payload = {
        "ticker": ticker,
        "prediction_mode": "enhanced_multitask",
        "horizon_days": int(horizon_days),
        "window_size": int(WINDOW_SIZE),
        "feature_columns": list(features),
        "target_type": "log_return",
        "direction_classes": ["Down", "Flat", "Up"],
        "metrics": {
            "directional_accuracy_binary": float(binary_da),
            "directional_accuracy_multiclass": float(multi_class_da),
            "pt_stat": float(pt_stat),
            "pt_p_value": float(pt_p_val),
            "dm_stat": float(dm_stat),
            "dm_p_value": float(dm_p_val),
            "mae": float(mae),
            "rmse": float(rmse),
        }
    }

    with open(MODELS_DIR / f"{ticker.lower()}_f_scaler.pkl", "wb") as f:
        pickle.dump(f_scaler, f)
    with open(MODELS_DIR / f"{ticker.lower()}_t_scaler.pkl", "wb") as f:
        pickle.dump(t_scaler, f)
    with open(MODELS_DIR / f"{ticker.lower()}_multitask_config.pkl", "wb") as f:
        pickle.dump(config_payload, f)

    print(f"✓ Đã lưu gói mô hình thành công vào: {MODELS_DIR}")
    return config_payload


if __name__ == "__main__":
    ticker_arg = sys.argv[1] if len(sys.argv) > 1 else "VCB"
    horizon_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    train_multitask_ticker(ticker_arg, horizon_days=horizon_arg, epochs=50)
