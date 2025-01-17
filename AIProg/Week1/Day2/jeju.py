import pandas as pd

url = "https://raw.githubusercontent.com/Datamanim/pandas/main/Jeju.csv"
df = pd.read_csv(url, encoding="EUC-KR")

print(df)

ans = df.select_dtypes(include='object').describe()
print(ans)

ans1 = df.select_dtypes(exclude=object).columns
# Index(['id', '거주인구', '근무인구', '방문인구', '총 유동인구', '평균 속도', '평균 소요 시간', '평균 기온', '일강수량', '평균 풍속'], dtype='object')

ans2 = df.select_dtypes(include=object).columns
# Index(['일자', '시도명', '읍면동명'], dtype='object')

df.isnull().sum()

# id          0
# 일자          0
# 시도명         0
# 읍면동명        0
# 거주인구        0
# 근무인구        0
# 방문인구        0
# 총 유동인구      0
# 평균 속도       0
# 평균 소요 시간    0
# 평균 기온       0
# 일강수량        0
# 평균 풍속       0
# dtype: int64


print(df.describe())