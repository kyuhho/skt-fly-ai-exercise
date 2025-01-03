import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv("/Users/kyuho/Codes/skt-fly-ai-exercise/Week2/ai-prog-1/auto-mpg.csv", sep=",")


# '?'를 NaN으로 변환
df['horsepower'] = df['horsepower'].replace('?', np.nan)

# 숫자로 변환 (NaN 유지)
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')

# 결측값을 평균값으로 대체
mean_horsepower = df['horsepower'].mean()  # 평균값 계산
df['horsepower'] = df['horsepower'].fillna(mean_horsepower)  # 평균값으로 결측값 대체

# Linear Regression


X = df['weight'].values.reshape(-1, 1)
y = df['horsepower'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_train)

# 선형 회귀 모델 학습
model = LinearRegression()
model.fit(X_poly, y_train)

X_range = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)  # X 범위 내에서 500개의 점 생성
X_range_poly = poly.transform(X_range)  # 다항식 변환


X_test_poly = poly.transform(X_test)
print (model.score(X_test_poly, y_test))

y_pred = model.predict(X_test_poly)

# plt.scatter(X, y, color='blue', alpha=0.5, label="Generated data")
# plt.plot(X_range, model.predict(X_range_poly), color='red', label="Fitted line")
# plt.title("Linear Regression Model")
# plt.xlabel("Weight")
# plt.ylabel("Horsepower")
# plt.legend()

# plt.show()

plt.figure(figsize=(12, 8))
ax1 = sns.kdeplot(y_test, label='y_test')
ax2 = sns.kdeplot(y_pred, label='y_pred', ax=ax1)

plt.show()