import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("/Users/kyuho/Codes/skt-fly-ai-exercise/Week2/ai-prog-1/toluca_company_dataset.csv", sep=",")

X = df["Lot_size"].values.reshape(-1,1)
y = df["Work_hours"].values

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.35, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
print (lr.score(X_test, y_test))

plt.figure(figsize=(12,8))
sns.kdeplot(y_test, label='y_test')
sns.kdeplot(y_pred, label='y_pred')
plt.legend()
plt.show()





