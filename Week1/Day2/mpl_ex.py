import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns

# 한글 폰트 설정
# plt.rc('font', family='Malgun Gothic')  # Windows의 경우
plt.rc('font', family='AppleGothic')  # macOS의 경우

# 유니코드 마이너스 기호 깨짐 방지
plt.rc('axes', unicode_minus=False)

y = np.arange(0, 10)**3
plt.plot(y)
# plt.show()

x = np.linspace(0, 10, 100)
y = np.sin(x)
z = np.cos(x)
plt.plot(x, y)
plt.plot(x, z)
# plt.show()


x1 = np.linspace(0, 10, 30)
y1 = np.sin(x1)
z1 = np.cos(x1)

plt.style.use('ggplot')
plt.figure(figsize=(15, 4))
# 두 줄 두 칼럼
plt.subplot(1, 3, 1)
plt.plot(x1, y1, 'r-')
plt.subplot(1,3, 2)
plt.plot(x1, z1, 'g--')
plt.subplot(1,3, 3)
plt.plot(x1, z1, 'b-.')
# plt.show()

n= 50
x2 = np.random.rand(n)
y2 = np.random.rand(n)
area = (30 * np.random.rand(n))**2
colors = np.random.rand(n)
plt.scatter(x2, y2, s=area, c=colors, alpha=0.7)
# plt.show()


arr=np.random.standard_normal((30,40))
plt.matshow(arr)
#히트맵 범례 : colorbar(shrink=0.8, aspect=10)
plt.colorbar(shrink=0.9, aspect=10)
# plt.show()


df = pd.read_csv("./titanic.csv", sep=',')

# # Embarked와 Fare 관계 시각화
# plt.figure(figsize=(8, 6))
# df.groupby('Embarked')['Fare'].mean().plot(kind='bar', color='skyblue', edgecolor='black')

# # 그래프 꾸미기
# plt.title('Average Fare by Embarkation Location', fontsize=14)
# plt.xlabel('Embarked (Location)', fontsize=12)
# plt.ylabel('Average Fare', fontsize=12)
# plt.xticks(rotation=0)
# plt.grid(axis='y', linestyle='--', alpha=0.7)

# # 그래프 표시
# plt.tight_layout()
# plt.show()

# Embarked와 Fare 데이터 결측값 제거
df = df.dropna(subset=['Embarked', 'Fare'])

print (df.describe())

# Boxplot 생성
plt.figure(figsize=(8, 6))
df.boxplot(column='Fare', by='Embarked', grid=False, patch_artist=True, showmeans=True, 
           boxprops=dict(facecolor='lightblue', color='black'),
           meanprops=dict(marker='o', markerfacecolor='red', markeredgecolor='black'),
           medianprops=dict(color='green')
           )

# Y축 스케일 조정 (95% 범위로 제한)
fare_max = df['Fare'].quantile(0.95)  # 상위 95% 값을 계산
plt.ylim(0, fare_max + 50)  # 약간의 여유 공간 추가

# 그래프 꾸미기
plt.title('3조: Boxplot of Fare by Embarkation Location', fontsize=14)
plt.suptitle('')  # 자동으로 생성된 상단 제목 제거
plt.xlabel('Embarked (Location)', fontsize=12)
plt.ylabel('Fare', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 그래프 표시
plt.tight_layout()
# plt.show()

