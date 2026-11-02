import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    LSTM,
    Bidirectional,
    Dense,
    Dropout,
    Flatten,
    Layer,
    Concatenate,
    LayerNormalization,
    GlobalAveragePooling1D,
    Add,
)
from tensorflow.keras.models import Model
from src.model.losses import AsymmetricVarianceMatchingLoss, DirectionalFocalLoss


class AttentionLayer(Layer):
    """Additive Single-Head Attention layer từ baseline của luận văn.
    Giữ nguyên tương thích ngược 100% với các checkpoint cũ.
    """
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

    def get_attention_weights(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        return tf.keras.backend.softmax(e, axis=1)


class MultiHeadTemporalAttention(Layer):
    """Multi-Head Temporal Attention (MHTA) Layer chuyên sâu cho chuỗi thời gian tài chính.

    Cơ chế:
      - Chia không gian biểu diễn thành H=4 heads độc lập:
        + Head 1: Nhận diện đà bứt phá động lượng (Momentum Breakout)
        + Head 2: Nhận diện bất thường khối lượng (Liquidity Spikes)
        + Head 3: Nhận diện biến động rủi ro và nến co thắt (Volatility Regime)
        + Head 4: Nhận diện điểm đảo chiều và hồi quy trung bình (Mean-Reversion)
      - Tính Scaled Dot-Product Attention: Softmax(Q·K^T / sqrt(d_k))·V
      - Trả về đồng thời context vector (cho dense head) và ma trận attention (cho XAI).
    """

    def __init__(self, num_heads: int = 4, key_dim: int = 16, **kwargs):
        super(MultiHeadTemporalAttention, self).__init__(**kwargs)
        self.num_heads = int(num_heads)
        self.key_dim = int(key_dim)
        self.total_dim = self.num_heads * self.key_dim

    def build(self, input_shape):
        feature_dim = input_shape[-1]
        self.q_proj = Dense(self.total_dim, name="q_proj", use_bias=False)
        self.k_proj = Dense(self.total_dim, name="k_proj", use_bias=False)
        self.v_proj = Dense(self.total_dim, name="v_proj", use_bias=False)
        self.out_proj = Dense(feature_dim, name="out_proj")
        super(MultiHeadTemporalAttention, self).build(input_shape)

    def _split_heads(self, x, batch_size):
        seq_len = tf.shape(x)[1]
        x = tf.reshape(x, (batch_size, seq_len, self.num_heads, self.key_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, x):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        q = self._split_heads(self.q_proj(x), batch_size)
        k = self._split_heads(self.k_proj(x), batch_size)
        v = self._split_heads(self.v_proj(x), batch_size)

        # Scaled dot-product: (batch, num_heads, seq_len, seq_len)
        scores = tf.matmul(q, k, transpose_b=True)
        scale = tf.math.rsqrt(tf.cast(self.key_dim, tf.float32))
        scores = scores * scale
        weights = tf.nn.softmax(scores, axis=-1)

        # Weighted value: (batch, num_heads, seq_len, key_dim)
        context = tf.matmul(weights, v)
        context = tf.transpose(context, perm=[0, 2, 1, 3])
        context = tf.reshape(context, (batch_size, seq_len, self.total_dim))

        projected = self.out_proj(context)
        context_vector = tf.reduce_mean(projected, axis=1)

        # Attention distribution: Lấy attention của step cuối cùng qua các heads
        last_step_weights = weights[:, :, -1, :]  # (batch, num_heads, seq_len)
        mean_attention_weights = tf.reduce_mean(last_step_weights, axis=1)  # (batch, seq_len)

        return context_vector, mean_attention_weights

    def get_config(self):
        cfg = super(MultiHeadTemporalAttention, self).get_config()
        cfg.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
        })
        return cfg


def _compile_model(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mae']
    )
    return model


def _apply_dense_head(x):
    x = Dense(units=32, activation='relu', name="Dense_Layer")(x)
    x = Dropout(0.2, name="Dropout_1")(x)
    return Dense(units=1, activation='linear', name="Output_Layer")(x)


# ==============================================================================
# BASELINE SINGLE-TASK ARCHITECTURES (Bảo toàn phục vụ Ablation Study)
# ==============================================================================

def build_lstm_only_model(input_shape):
    inputs = Input(shape=input_shape, name="Input_Layer")
    x = LSTM(units=50, name="LSTM_Layer")(inputs)
    outputs = _apply_dense_head(x)
    model = Model(inputs=inputs, outputs=outputs, name="LSTM_Only")
    return _compile_model(model)


def build_cnn_only_model(input_shape):
    inputs = Input(shape=input_shape, name="Input_Layer")
    x = Conv1D(filters=64, kernel_size=3, activation='relu', name="1D_CNN_Layer")(inputs)
    x = MaxPooling1D(pool_size=2, name="MaxPooling_Layer")(x)
    x = Flatten(name="Flatten_Layer")(x)
    outputs = _apply_dense_head(x)
    model = Model(inputs=inputs, outputs=outputs, name="CNN_Only")
    return _compile_model(model)


def build_attention_only_model(input_shape):
    inputs = Input(shape=input_shape, name="Input_Layer")
    x = AttentionLayer(name="Attention_Layer")(inputs)
    outputs = _apply_dense_head(x)
    model = Model(inputs=inputs, outputs=outputs, name="Attention_Only")
    return _compile_model(model)


def build_cnn_lstm_model(input_shape):
    inputs = Input(shape=input_shape, name="Input_Layer")
    x = Conv1D(filters=64, kernel_size=3, activation='relu', name="1D_CNN_Layer")(inputs)
    x = MaxPooling1D(pool_size=2, name="MaxPooling_Layer")(x)
    x = LSTM(units=50, name="LSTM_Layer")(x)
    outputs = _apply_dense_head(x)
    model = Model(inputs=inputs, outputs=outputs, name="CNN_LSTM")
    return _compile_model(model)


def build_lstm_attention_model(input_shape):
    inputs = Input(shape=input_shape, name="Input_Layer")
    x = LSTM(units=50, return_sequences=True, name="LSTM_Layer")(inputs)
    x = AttentionLayer(name="Attention_Layer")(x)
    outputs = _apply_dense_head(x)
    model = Model(inputs=inputs, outputs=outputs, name="LSTM_Attention")
    return _compile_model(model)


def build_cnn_attention_model(input_shape):
    inputs = Input(shape=input_shape, name="Input_Layer")
    x = Conv1D(filters=64, kernel_size=3, activation='relu', name="1D_CNN_Layer")(inputs)
    x = MaxPooling1D(pool_size=2, name="MaxPooling_Layer")(x)
    x = AttentionLayer(name="Attention_Layer")(x)
    outputs = _apply_dense_head(x)
    model = Model(inputs=inputs, outputs=outputs, name="CNN_Attention")
    return _compile_model(model)


def build_cnn_lstm_attention_model(input_shape):
    """Mô hình chuẩn Baseline CNN-LSTM-Attention của luận văn."""
    inputs = Input(shape=input_shape, name="Input_Layer")
    x = Conv1D(filters=64, kernel_size=3, activation='relu', name="1D_CNN_Layer")(inputs)
    x = MaxPooling1D(pool_size=2, name="MaxPooling_Layer")(x)
    x = LSTM(units=50, return_sequences=True, name="LSTM_Layer")(x)
    x = AttentionLayer(name="Attention_Layer")(x)
    outputs = _apply_dense_head(x)
    model = Model(inputs=inputs, outputs=outputs, name="CNN_LSTM_Attention")
    return _compile_model(model)


def build_cnn_lstm_attention_classifier_model(input_shape, num_classes=3):
    inputs = Input(shape=input_shape, name="Input_Layer")
    x = Conv1D(filters=64, kernel_size=3, activation='relu', name="1D_CNN_Layer")(inputs)
    x = MaxPooling1D(pool_size=2, name="MaxPooling_Layer")(x)
    x = LSTM(units=50, return_sequences=True, name="LSTM_Layer")(x)
    x = AttentionLayer(name="Attention_Layer")(x)
    x = Dense(units=32, activation='relu', name="Dense_Layer")(x)
    x = Dropout(0.2, name="Dropout_1")(x)
    outputs = Dense(units=num_classes, activation='softmax', name="Probability_Output_Layer")(x)
    model = Model(inputs=inputs, outputs=outputs, name="CNN_LSTM_Attention_Alpha_Classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ==============================================================================
# PHASE 1 ADVANCED UPGRADE: ENHANCED MULTI-TASK CNN-BILSTM-MHTA MODEL
# ==============================================================================

def build_enhanced_multitask_model(input_shape,
                                   num_direction_classes: int = 3,
                                   asymmetric_penalty: float = 1.75,
                                   learning_rate: float = 0.001):
    """Kiến trúc Multi-Task Tối Thượng (Phase 1).

    Cải tiến kỹ thuật:
      1. Multi-scale 1D-CNN Inception Block (kernels 2, 3, 5 with same padding)
         bảo toàn trọn vẹn 30 bước thời gian (không downsample).
      2. LayerNormalization ổn định phân phối đặc trưng.
      3. Bidirectional LSTM (BiLSTM, 2x32=64 units, return_sequences=True).
      4. Multi-Head Temporal Attention (MHTA, 4 heads) trích xuất đồng thời
         context vector và ma trận attention 30 phiên cho XAI.
      5. Residual Skip Connection từ CNN sang Attention.
      6. Ba đầu ra đồng bộ:
         - 'direction_output': 3 classes (Down, Flat, Up) với DirectionalFocalLoss.
         - 'regression_output': 1 unit linear với AsymmetricVarianceMatchingLoss.
         - 'attention_weights': 30 units (softmax over 30 days) cho XAI trực tiếp.
    """
    inputs = Input(shape=input_shape, name="Input_Sequence")

    # 1. Multi-scale 1D-CNN Inception Block (padding='same' giữ nguyên 30 timesteps)
    c2 = Conv1D(filters=32, kernel_size=2, padding="same", activation="relu", name="Conv_Scale2")(inputs)
    c3 = Conv1D(filters=32, kernel_size=3, padding="same", activation="relu", name="Conv_Scale3")(inputs)
    c5 = Conv1D(filters=32, kernel_size=5, padding="same", activation="relu", name="Conv_Scale5")(inputs)
    cnn_concat = Concatenate(axis=-1, name="CNN_MultiScale_Concat")([c2, c3, c5])
    cnn_norm = LayerNormalization(name="CNN_LayerNorm")(cnn_concat)
    cnn_drop = Dropout(0.15, name="CNN_Dropout")(cnn_norm)

    # 2. Bidirectional LSTM Layer (giữ nguyên sequence length 30)
    bilstm = Bidirectional(
        LSTM(units=32, return_sequences=True, dropout=0.1, recurrent_dropout=0.0),
        name="BiLSTM_Bidirectional"
    )(cnn_drop)
    bilstm_norm = LayerNormalization(name="BiLSTM_LayerNorm")(bilstm)

    # 3. Multi-Head Temporal Attention Layer (4 heads)
    mhta_layer = MultiHeadTemporalAttention(num_heads=4, key_dim=16, name="MultiHead_Attention")
    attn_context, attn_scores = mhta_layer(bilstm_norm)  # context: (batch, 64), scores: (batch, 30)

    # 4. Residual Skip Connection từ CNN sang Attention
    cnn_global = GlobalAveragePooling1D(name="CNN_GlobalSkip")(cnn_drop)
    skip_dim = int(attn_context.shape[-1]) if attn_context.shape[-1] is not None else 64
    cnn_skip_proj = Dense(skip_dim, activation="relu", name="Skip_Projection")(cnn_global)
    merged_representation = Add(name="Residual_Add")([attn_context, cnn_skip_proj])
    merged_norm = LayerNormalization(name="Merged_LayerNorm")(merged_representation)

    # 5. Shared Dense Core
    shared = Dense(units=64, activation="relu", name="Shared_Representation")(merged_norm)
    shared = Dropout(0.2, name="Shared_Dropout")(shared)

    # 6. Branch 1: Direction Classification Head (Focal Loss)
    dir_h = Dense(units=32, activation="relu", name="Dir_Dense_Hidden")(shared)
    dir_h = Dropout(0.1, name="Dir_Dropout")(dir_h)
    direction_output = Dense(
        units=num_direction_classes,
        activation="softmax",
        name="direction_output"
    )(dir_h)

    # 7. Branch 2: Magnitude Regression Head (Asymmetric Variance Matching Loss)
    reg_h = Dense(units=32, activation="relu", name="Reg_Dense_Hidden")(shared)
    reg_h = Dropout(0.1, name="Reg_Dropout")(reg_h)
    regression_output = Dense(
        units=1,
        activation="linear",
        name="regression_output"
    )(reg_h)

    # 8. Model Outputs: Hai nhánh huấn luyện chính (direction & regression)
    model = Model(
        inputs=inputs,
        outputs={
            "direction_output": direction_output,
            "regression_output": regression_output,
        },
        name="Enhanced_MultiTask_CNN_BiLSTM_MHTA"
    )

    reg_loss = AsymmetricVarianceMatchingLoss(
        variance_weight=0.4,
        direction_weight=0.2,
        asymmetric_penalty=asymmetric_penalty,
        direction_temp=20.0
    )
    dir_loss = DirectionalFocalLoss(gamma=2.0)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "direction_output": dir_loss,
            "regression_output": reg_loss,
        },
        loss_weights={
            "direction_output": 0.6,
            "regression_output": 0.4,
        },
        metrics={
            "direction_output": ["accuracy"],
            "regression_output": ["mae"],
        }
    )

    return model


def extract_attention_from_model(model: Model, X_input: np.ndarray) -> list:
    """Trích xuất Attention weights 100% từ layer MultiHead_Attention (XAI)."""
    try:
        mhta_layer = model.get_layer("MultiHead_Attention")
        extractor = Model(inputs=model.input, outputs=mhta_layer.output[1])
        scores = extractor.predict(X_input, verbose=0)
        return scores.tolist()
    except Exception as e:
        print(f"Warning: Không thể trích xuất MHTA attention: {e}")
        return []


MODEL_BUILDERS = {
    "lstm_only": build_lstm_only_model,
    "cnn_only": build_cnn_only_model,
    "attention_only": build_attention_only_model,
    "cnn_lstm": build_cnn_lstm_model,
    "lstm_attention": build_lstm_attention_model,
    "cnn_attention": build_cnn_attention_model,
    "cnn_lstm_attention": build_cnn_lstm_attention_model,
    "enhanced_multitask": build_enhanced_multitask_model,
}
