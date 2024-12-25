import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("./tips.csv", sep=",")
print(df)


g = sns.FacetGrid(df, col="time", row='sex');
# g.map(sns.scatterplot, "total_bill", "tip")
g.map(sns.histplot, "total_bill")

plt.show()
