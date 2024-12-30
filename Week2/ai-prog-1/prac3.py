import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

# 데이터 로드
data = pd.read_csv("/Users/kyuho/Codes/skt-fly-ai-exercise/Week2/ai-prog-1/train.csv")

test_data = data = pd.read_csv("/Users/kyuho/Codes/skt-fly-ai-exercise/Week2/ai-prog-1/test.csv")

print(data.columns)

# 결측치 처리
data["배터리용량"].fillna(data["배터리용량"].mean(), inplace=True)

# 범주형 데이터 처리
data = pd.get_dummies(data, columns=["제조사", "모델", "차량상태", "구동방식"], drop_first=True)

# 이진 데이터 처리
data["사고이력"] = data["사고이력"].apply(lambda x: 1 if x == "Yes" else 0)

# 독립 변수(X)와 종속 변수(y) 분리
X = data.drop(["ID", "가격(백만원)"], axis=1)
y = data["가격(백만원)"]

# 모델 학습
model = LinearRegression()
model.fit(X, y)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 모델 평가
test_data["배터리용량"].fillna(test_data["배터리용량"].mean(), inplace=True)
test_data = pd.get_dummies(test_data, columns=["제조사", "모델", "차량상태", "구동방식"], drop_first=True)
test_data["사고이력"] = test_data["사고이력"].apply(lambda x: 1 if x == "Yes" else 0)

X_test = test_data.drop(["ID"], axis=1)

pred = model.predict(X_test)

# print (pred)

# submit = pd.read_csv('/Users/kyuho/Codes/skt-fly-ai-exercise/Week2/ai-prog-1/sample_submission.csv')

# # submit['가격(백만원)'] = pred
# submit.head()

# submit.to_csv('./baseline_submission.csv',index=False)
