import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# 1. 데이터 로드
data = pd.read_csv('/Users/kyuho/Codes/skt-fly-ai-exercise/Week2/Day6/judge_pred/train.csv')
texts = data['facts']
labels = data['first_party_winner']

# 2. 텍스트 전처리
# OOV는 dictionary에 없는 데이터
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)


for i, (word, index) in enumerate(tokenizer.word_index.items()):
    if i < 5:  # 상위 5개만 출력
        print(f"{word}: {index}")
    else:
        break
# <OOV>: 1
# the: 2
# of: 3
# to: 4
# and: 5

# text의 sequence가 길어지면 잘라줌
padded_sequences = pad_sequences(sequences, maxlen=200, truncating='post')

print(padded_sequences[0])

# 3. 데이터 분리
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# # 4. 모델 구성
model = Sequential([
    Embedding(input_dim=10000, output_dim=64, input_length=200),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dense(1, activation='sigmoid')
])

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# # 5. 학습
# history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# # 6. 평가
# results = model.evaluate(X_val, y_val)
# print("Validation Loss, Validation Accuracy:", results)
