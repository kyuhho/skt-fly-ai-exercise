import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt    

df = pd.read_csv("./tips.csv", sep=',')

# sns.boxplot( y='tip', data=df)

# sns.countplot(x='day', data=df)

# sns.countplot(data = df, x='day', hue="sex")

# 요일별 tip 시각화, 각 요일 별로 size를 합산한 값을 해당 요일 tip에 divide
print(df.groupby('day')['tip'].sum())
print(df.groupby('day')['size'].sum())


fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # 1행 3열의 서브플롯 

# 각 요일 별 tip/size
mean_tip_df = df.groupby('day')['tip'].sum() / df.groupby('day')['size'].sum()
# sns.barplot(x=mean_tip_df.index, y=mean_tip_df.values)

# 인원 수 별 total bill displot
sns.histplot(data=df, x='total_bill', hue='size', ax=axes[0], kde=True)
axes[0].set_title("Total Bill by Size")

# 요일 별 total bill boxplot
sns.boxplot(y=df['total_bill'], x=df['day'], ax=axes[1])
axes[1].set_title("Total Bill by Day")

# 흡연자별 total bill violinplot
sns.violinplot(x='smoker', y='total_bill', hue='sex', data=df, split=True, ax=axes[2])
axes[2].set_title("Total Bill by Smoker and Sex")

plt.show()


titanic = pd.read_csv("./titanic.csv", sep=',')

df = titanic[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')


plt.show()

