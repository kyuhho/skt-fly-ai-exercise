import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("/Users/kyuho/Codes/skt-fly-ai-exercise/Week1/HW/data.csv", sep=";")

# df["PTS_per_min"] = df["PTS"] / df["MP"]
# MP: 경기당 평균 출전 시간 (minute)
# PTS: 총 득점
# PTS_per_Minute: 출전시간 대비 득점 효율

# PTS/MP: 경기 당 총 득점
df["PTS_per_min"] = df["PTS"] / df["MP"]

# 상위 20%의 PTS_per_min 값
percentile_80 = df["PTS_per_min"].quantile(0.8)

average_mp = df["MP"].mean()

high_efficiency_players = df[(df["MP"] < average_mp) & (df["PTS_per_min"] > percentile_80)]

print(high_efficiency_players[["Player", "MP", "PTS", "PTS_per_min"]])


# 상위 20%의 PTS per minute을 가지는 선수가 극단적으로 효율이 좋은 특정 선수를 표현할 수 있는지 확인
# plt.figure(figsize=(10, 6))
# sns.histplot(df["PTS_per_min"], kde=True, bins=30, color="skyblue")
# plt.axvline(x=percentile_80, color="red", linestyle="--", label="Top 20% Threshold")
# plt.title("Distribution of PTS_per_min with Top 20% Threshold")
# plt.xlabel("Points per Minute (PTS_per_min)")
# plt.ylabel("Frequency")
# plt.legend()
# plt.grid(True)
# plt.show()

'''
출전 시간과, PTS per Minute의 상관관계 확인

* 출전 시간이 많은 선수들은 보통 팀의 주전 선수이거나 핵심적인 득점원일 가능성이 높다

* 주전 선수들은 출전 시간이 많기 때문에 공을 더 자주 소유하거나 득점 기회를 더 많이 가질 수 있기 때문에 PTS_per_min이 높게 나타난다고 분석.
'''

# 출전 시간을 0.5 간격으로 구간화
bins = np.arange(0, df["MP"].max() + 0.5, 0.5)  # 0.5 단위로 구간화
df["MP_bin"] = pd.cut(df["MP"], bins=bins)

# 구간별 평균 PTS_per_min 계산
mp_bin_means = df.groupby("MP_bin")["PTS_per_min"].mean().reset_index()

# 구간 중심(midpoint) 추가
mp_bin_means["MP_midpoint"] = mp_bin_means["MP_bin"].apply(lambda x: x.mid)

# 그래프 4개
plt.figure(figsize=(20, 10))

# 첫 번째 그래프: PTS_per_min 분포도
plt.subplot(2, 2, 1)
sns.histplot(df["PTS_per_min"], kde=True, bins=30, color="skyblue")
plt.axvline(x=percentile_80, color="red", linestyle="--", label="Top 20% Threshold")
plt.title("Distribution of PTS_per_min with Top 20% Threshold")
plt.xlabel("Points per Minute (PTS_per_min)")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)

# 두 번째 그래프: 출전 시간과 PTS_per_min의 관계
plt.subplot(2, 2, 2)
sns.lineplot(x="MP_midpoint", y="PTS_per_min", data=mp_bin_means, marker="o")
plt.title("PTS_per_min Across MP Intervals (0.5 Stride)")
plt.xlabel("Minutes Played (MP Midpoint)")
plt.ylabel("Points per Minute (PTS_per_min)")
plt.grid(True)


# 세 번째 그래프: 효율성이 높은 선수들의 분포 

plt.subplot(2, 2, 3)
sns.scatterplot(x="MP", y="PTS_per_min", data=high_efficiency_players)
plt.title("High Efficiency Players with Low Minutes Played")
plt.xlabel("Minutes Played (MP)")
plt.ylabel("Points per Minute (PTS_per_min)")
plt.grid(True)

# 네 번째 그래프: 포지션별 높은 효율 선수들의 분포 히스토그램

plt.subplot(2, 2, 4)
sns.histplot(data=high_efficiency_players, x="Pos", hue="Pos", multiple="stack", shrink=0.8, palette="Set2")
plt.title("High Efficiency Players by Position")
plt.xlabel("Position")
plt.ylabel("Frequency")
plt.grid(True)


# 그래프 표시
plt.show()