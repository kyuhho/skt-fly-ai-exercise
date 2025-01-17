import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import SimpleRNN

# Ref: https://wikidocs.net/22886

# 간단한 예시 데이터
train_X = [
    [0.1, 4.2, 1.5, 1.1, 2.8],
    [1.0, 3.1, 2.5, 0.7, 1.1],
    [0.3, 2.1, 1.5, 2.1, 0.1],
    [2.2, 1.4, 0.5, 0.9, 1.1]
]

# (4, 5)
print("Shape of train_X (list):", np.shape(train_X))

# RNN에 넣기 위해 차원을 (batch, timesteps, features) 형태로 만들어줌
train_X = [
    [
        [0.1, 4.2, 1.5, 1.1, 2.8],
        [1.0, 3.1, 2.5, 0.7, 1.1],
        [0.3, 2.1, 1.5, 2.1, 0.1],
        [2.2, 1.4, 0.5, 0.9, 1.1]
    ]
]

# 넘파이 배열로 변환 (float32 자료형)
train_X = np.array(train_X, dtype=np.float32)
print("Shape of train_X (ndarray):", train_X.shape)  # (1, 4, 5)

# ------------------------------
# 1) return_sequences=False, return_state=False (기본값)
# ------------------------------
rnn = SimpleRNN(3)
hidden_state = rnn(train_X)
print(f"Hidden state (SimpleRNN 기본): {hidden_state}, shape: {hidden_state.shape}")

# ------------------------------
# 2) return_sequences=True
# ------------------------------
rnn = SimpleRNN(3, return_sequences=True)
hidden_states = rnn(train_X)
print(f"Hidden states (return_sequences=True): {hidden_states}, shape: {hidden_states.shape}")
