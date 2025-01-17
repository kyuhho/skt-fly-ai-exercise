import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

titanic = pd.read_csv("./titanic.csv", sep=',')

# 빈도 그래프
fig = plt.figure(figsize=(15, 5))

ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)

sns.countplot(x='class', palette='Set1', data=titanic, ax=ax1)
sns.countplot(x='class', palette='Set3', data=titanic, ax=ax2, hue='who')
sns.countplot(x='class', palette='Set3', data=titanic, ax=ax3, hue='who',dodge=False)


plt.show()