import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


plt.rc('font', family='AppleGothic')  # macOS의 경우


df = pd.read_csv("/Users/kyuho/Codes/skt-fly-ai-exercise/Week1/HW/news.csv", sep=",")

news_frequency = df.groupby(['city_do_nm', 'city_gn_nm']).size().reset_index(name='news_count')

filtered_df = news_frequency[(news_frequency['city_do_nm'] != '0') & (news_frequency['city_gn_nm'] != '0')]

# Determine the city with the highest total news count
top_city = filtered_df.groupby('city_do_nm')['news_count'].sum().idxmax()

# Filter the data for the top city
top_city_data = filtered_df[filtered_df['city_do_nm'] == top_city]

# Visualization for the top city
plt.figure(figsize=(12, 8))

# First Graph: News frequency by `city_do_nm`
plt.subplot(2, 1, 1)
sns.barplot(data=filtered_df, x='city_do_nm', y='news_count', ci=None, estimator=sum, palette="viridis")
plt.title('News Frequency by City (city_do_nm)', fontsize=16)
plt.xlabel('City (city_do_nm)', fontsize=12)
plt.ylabel('Total News Count', fontsize=12)
plt.xticks(rotation=45)

# Second Graph: News frequency for city_gn_nm within the top city
plt.subplot(2, 1, 2)
sns.barplot(data=top_city_data, x='city_gn_nm', y='news_count', ci=None, palette="plasma")
plt.title(f'News Frequency in {top_city} by District (city_gn_nm)', fontsize=16)
plt.xlabel('District (city_gn_nm)', fontsize=12)
plt.ylabel('Total News Count', fontsize=12)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()