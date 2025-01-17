import numpy as np

import pandas as pd

data = np.loadtxt('/Users/kyuho/Codes/skt-fly-ai-exercise/Week1/Day1/chocolate_rating.csv', delimiter=',')



# 카카오 함량 (3번째 열), 평점 (4번째 열) 추출
cocoa_percent = data[:, 2]
ratings = data[:, 3]

print ('차원', data.ndim)
print ('형태', data.shape)
print ("원소 수", data.size)


# 차원 2
# 형태 (1795, 4)
# 원소 수 7180
# [[1.000e+00 2.016e+03 6.300e-01 3.750e+00]
#  [2.000e+00 2.015e+03 7.000e-01 2.750e+00]
#  [3.000e+00 2.015e+03 7.000e-01 3.000e+00]
#  ...
#  [1.793e+03 2.011e+03 6.500e-01 3.500e+00]
#  [1.794e+03 2.011e+03 6.200e-01 3.250e+00]
#  [1.795e+03 2.010e+03 6.500e-01 3.000e+00]]

# 상위 10 퍼센트의 평점을 가진 초콜릿, 카카오 함량이 평균 언저리


# 상위 10% 평점 기준
threshold = np.percentile(ratings, 90)  # 상위 10%의 평점 계산
print(f"상위 10% 평점 기준: {threshold:.2f}")

# 상위 10% 초콜릿 인덱스 배열 추출
high_rating_indices = ratings >= threshold  # 상위 10%에 해당하는 인덱스

# 카카오 함량 특정 범위 기준
# 카카오 함량 특정 범위 기준
MIN_COCOA_THRESHOLD = 0.6  # 최소 카카오 함량 (60%)
MAX_COCOA_THRESHOLD = 0.8  # 최대 카카오 함량 (80%)

# 상위 10% 초콜릿 중 카카오 함량이 특정 범위에 해당하는 초콜릿 추출
high_rating_cocoa_indices = np.logical_and(high_rating_indices, np.logical_and(cocoa_percent >= MIN_COCOA_THRESHOLD, cocoa_percent <= MAX_COCOA_THRESHOLD))

print(f"상위 10% 평점을 받은 초콜릿 중 카카오 함량이 {MIN_COCOA_THRESHOLD*100}% 이상 {MAX_COCOA_THRESHOLD*100}% 이하인 초콜릿 개수: {np.sum(high_rating_cocoa_indices)}개")


columns = ['index','years','cacao','rating']

df = pd.read_csv('/Users/kyuho/Codes/skt-fly-ai-exercise/Week1/Day1/chocolate_rating.csv', header=None, sep=',', names=columns)

print(df)