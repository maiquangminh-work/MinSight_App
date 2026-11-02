"""
Custom loss functions cho bài toán dự báo định lượng và phân loại hướng cổ phiếu ngân hàng.

Bao gồm:
1. VarianceMatchingMSE: Giữ nguyên từ baseline luận văn (chống flat prediction).
2. AsymmetricVarianceMatchingLoss: Nâng cấp với Asymmetric Direction Penalty,
   phạt nặng hơn đối với lỗi False Buy (hô Tăng nhưng giá Giảm, gây rủi ro vốn)
   so với False Sell (bỏ lỡ cơ hội).
3. DirectionalFocalLoss: Dành cho nhánh phân loại hướng Multi-Task (Down / Flat / Up),
   tập trung gradient vào các điểm đảo chiều khó (turning points) và giảm trọng số
   của các phiên sideway nhiễu.
"""
import tensorflow as tf


class VarianceMatchingMSE(tf.keras.losses.Loss):
    """Composite loss = MSE + variance_gap_penalty + direction_mismatch_penalty

    L(y, ŷ) = MSE(y, ŷ)
            + λ_var · ReLU(Std(y) - Std(ŷ)) / Std(y)       # chỉ phạt khi pred "phẳng"
            + λ_dir · mean[(tanh(T·y) - tanh(T·ŷ))²]       # phạt khi sai hướng
    """

    def __init__(self,
                 variance_weight: float = 0.3,
                 direction_weight: float = 0.1,
                 direction_temp: float = 20.0,
                 name: str = "variance_matching_mse",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.variance_weight = float(variance_weight)
        self.direction_weight = float(direction_weight)
        self.direction_temp = float(direction_temp)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # 1. MSE cơ bản
        mse = tf.reduce_mean(tf.square(y_true - y_pred))

        # 2. Std matching
        std_true = tf.sqrt(tf.math.reduce_variance(y_true) + 1e-8)
        std_pred = tf.sqrt(tf.math.reduce_variance(y_pred) + 1e-8)
        var_gap = tf.nn.relu(std_true - std_pred) / (std_true + 1e-8)

        # 3. Direction mismatch (smooth sign)
        tt = tf.tanh(y_true * self.direction_temp)
        tp = tf.tanh(y_pred * self.direction_temp)
        direction_mismatch = tf.reduce_mean(tf.square(tt - tp))

        return (mse
                + self.variance_weight * var_gap
                + self.direction_weight * direction_mismatch)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "variance_weight": self.variance_weight,
            "direction_weight": self.direction_weight,
            "direction_temp": self.direction_temp,
        })
        return cfg


class AsymmetricVarianceMatchingLoss(tf.keras.losses.Loss):
    """Nâng cấp bất đối xứng (Asymmetric Penalty) cho Regression Head.

    L_asym(y, ŷ) = MSE(y, ŷ)
                 + λ_var · ReLU(Std(y) - Std(ŷ)) / Std(y)
                 + λ_dir · mean[ W_asym(y, ŷ) · (tanh(T·y) - tanh(T·ŷ))² ]

    Trong đó W_asym:
      - False Buy (ŷ > 0 và y < 0): nhân trọng số asymmetric_penalty (mặc định 1.75).
        Lý do tài chính: Mua nhầm cổ phiếu lao dốc gây drawdown vốn nặng nề,
        bắt buộc mô hình phải thận trọng cao nhất trước khi báo Tăng.
      - Các trường hợp khác: nhân trọng số 1.0.
    """

    def __init__(self,
                 variance_weight: float = 0.4,
                 direction_weight: float = 0.2,
                 asymmetric_penalty: float = 1.75,
                 direction_temp: float = 20.0,
                 name: str = "asymmetric_variance_matching_loss",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.variance_weight = float(variance_weight)
        self.direction_weight = float(direction_weight)
        self.asymmetric_penalty = float(asymmetric_penalty)
        self.direction_temp = float(direction_temp)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # 1. MSE cơ bản
        mse = tf.reduce_mean(tf.square(y_true - y_pred))

        # 2. Khớp phương sai (Chống flat prediction)
        std_true = tf.sqrt(tf.math.reduce_variance(y_true) + 1e-8)
        std_pred = tf.sqrt(tf.math.reduce_variance(y_pred) + 1e-8)
        var_gap = tf.nn.relu(std_true - std_pred) / (std_true + 1e-8)

        # 3. Phạt sai hướng bất đối xứng
        tt = tf.tanh(y_true * self.direction_temp)
        tp = tf.tanh(y_pred * self.direction_temp)
        sq_mismatch = tf.square(tt - tp)

        # Phát hiện False Buy: pred > 0 nhưng true < 0
        is_false_buy = tf.cast(tf.logical_and(y_pred > 0.0, y_true < 0.0), tf.float32)
        asym_weights = 1.0 + (self.asymmetric_penalty - 1.0) * is_false_buy
        asym_direction_mismatch = tf.reduce_mean(asym_weights * sq_mismatch)

        return (mse
                + self.variance_weight * var_gap
                + self.direction_weight * asym_direction_mismatch)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "variance_weight": self.variance_weight,
            "direction_weight": self.direction_weight,
            "asymmetric_penalty": self.asymmetric_penalty,
            "direction_temp": self.direction_temp,
        })
        return cfg


class DirectionalFocalLoss(tf.keras.losses.Loss):
    """Focal Cross-Entropy Loss có phạt khoảng cách nhãn cho Direction Classification.

    FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)
    Kèm ma trận phạt khoảng cách: đoán Up (2) mà thực tế Down (0) bị phạt gấp đôi
    so với đoán Flat (1).
    """

    def __init__(self,
                 gamma: float = 2.0,
                 alpha: list | None = None,
                 name: str = "directional_focal_loss",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = float(gamma)
        # Trọng số cho 3 lớp [Down, Flat, Up] — tăng trọng số cho Down và Up
        self.alpha = alpha or [1.2, 0.8, 1.2]

    def call(self, y_true, y_pred):
        """
        Args:
            y_true: Tensor shape (batch_size,) chứa index int [0, 1, 2]
            y_pred: Tensor shape (batch_size, 3) softmax probabilities
        """
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_true_one_hot = tf.one_hot(y_true, depth=3)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # Cross entropy cơ bản theo từng mẫu
        ce = -tf.reduce_sum(y_true_one_hot * tf.math.log(y_pred), axis=-1)

        # Xác suất dự báo đúng lớp mục tiêu p_t
        p_t = tf.reduce_sum(y_true_one_hot * y_pred, axis=-1)

        # Focal modulating factor (1 - p_t)^gamma
        focal_weight = tf.pow(1.0 - p_t, self.gamma)

        # Alpha class balancing
        alpha_tensor = tf.constant(self.alpha, dtype=tf.float32)
        sample_alpha = tf.gather(alpha_tensor, y_true)

        # Khoảng cách sai lệch hướng (Severe Direction Error Penalty)
        # Nếu True=0 (Down) nhưng Pred argmax=2 (Up) hoặc ngược lại -> phạt thêm
        pred_labels = tf.cast(tf.argmax(y_pred, axis=-1), tf.float32)
        true_labels = tf.cast(y_true, tf.float32)
        dir_distance_penalty = 1.0 + 0.5 * tf.abs(pred_labels - true_labels)

        loss = sample_alpha * focal_weight * ce * dir_distance_penalty
        return tf.reduce_mean(loss)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "gamma": self.gamma,
            "alpha": self.alpha,
        })
        return cfg
