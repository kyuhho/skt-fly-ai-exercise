import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

data_num = 1000
x = 3 *np.random.rand(data_num, 1) -1
y = 0.2*(x**2) + np.random.randn(data_num, 1)

# plt.scatter(x, y)
# plt.show()

poly_features = PolynomialFeatures(degree=2, include_bias=False)

# fit 안해도 되나? => fit_transform
# x, x^2
x_poly = poly_features.fit_transform(x)

print(x[0])
print(x_poly[0])
# [-0.49085036]
# [-0.49085036  0.24093407]

lin_reg = LinearRegression()
lin_reg.fit(x_poly, y)
print(lin_reg.intercept_, lin_reg.coef_)


# 데이터 시각화
plt.scatter(x, y, color='blue', alpha=0.5, label="Generated data")
plt.title("Randomly Generated Linear Data with Noise")
plt.xlabel("x (Input Feature)")
plt.ylabel("y (Target Output)")
plt.legend()
plt.show()