import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

X = [-3, 31, -11, 4, 0, 22, -2, -5, -25, -14]
y = [-2, 32, -10, 5, 1, 23, -1, -4, -24, -13]

X_train = np.array(X).reshape(-1, 1)
y_train = np.array(y)

# print (X_train.shape, y_train.shape)

# plt.plot(X,y, 'o--')
# plt.show()

df = pd.DataFrame({'X': X, 'y': y})
# print (df.head())

# 모든 데이터를 사용하여 학습
X_train = df.loc[:, ['X']]
y_train = df.loc[:, ['y']]

# print(X_train, y_train)

lr = LinearRegression()

lr.fit(X_train,y_train)

# print(lr.coef_, lr.intercept_)

# X_new = np.array(11).reshape(1,1)
X_new = pd.DataFrame({'X': [11]})  # Feature 이름을 명시적으로 지정
# print(lr.predict(X_new)) # 12

X_test = np.arange(11,16,1).reshape(-1, 1)

# Result
# [[12.]
#  [13.]
#  [14.]
#  [15.]
#  [16.]]
# print(lr.predict(X_test))

x = 2 * np.random.rand(100,1) # [0, 1) 범위에서 균일한 분포 100 x 1 Array
y = 4 + 3*x + np.random.randn(100,1) # normal distribution(mu = 0, var=1)

# plt.scatter(x,y)
# plt.show()

x_b = np.c_[np.ones((100,1)), x] # 모든 sample index 0번에 1을 추가

theta_best = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y)
# print(theta_best)

x_new = np.array([[0], [2]])
x_new_b = np.c_[np.ones((2,1)), x_new]
y_predict = x_new_b.dot(theta_best)
# print(y_predict)

# plt.plot(x_new, y_predict, 'r-')
# plt.plot(x, y, 'b.')
# plt.axis([0,2,0,15])
# plt.show()

# Using Scikit-Learn
lin_reg = LinearRegression()
lin_reg.fit(x, y)
# print(lin_reg.intercept_, lin_reg.coef_)

# 경사하강법 구현
y = 4 + 3*np.random.rand(100,1)

learing_rate = 0.1
interations = 1000
m = x_b.shape[0] #100

theta = np.random.randn(2,1) # 2x1 크기의 평균 0, 분산 1 정규 분포

for iteration in range(interations):
    gradients = 2/m * x_b.T.dot(x_b.dot(theta) - y)
    theta = theta - learing_rate * gradients

print(theta)
# [[ 5.61952603]
#  [-0.17837781]]
