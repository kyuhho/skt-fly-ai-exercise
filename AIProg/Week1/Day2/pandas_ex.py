import pandas as pd
import numpy as np

a = pd.Series([1, 2, 3, 4, 5])
print(a)

b = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e']) 
print(b)

#배열
arr1=np.array([['한빛','남자','20','180'],
              ['한결','남자','21','177'],
              ['한라','여자','20','160']])

columns = ['이름','성별','나이','키']

df1 = pd.DataFrame(arr1, columns=columns)

print(df1)


list1 = list([['허준호','남자','30','183'],
 	['이가원','여자','24','162'],
 	['배규민','남자','23','179'],
 	['고고림','남자','21','182'],
 	['이새봄','여자','28','160'],
 	['이보람','여자','26','163'],
 	['이루리','여자','24','157'],
 	['오다현','여자','24','172']])
col_names = ['이름','성별','나이','키']

df2 = pd.DataFrame(list1, columns=col_names)

df2.to_csv('./file.csv', header=True, index=False, encoding='utf-8')

#CSV 파일 읽기
print(pd.read_csv('./file.csv', sep=','), end='\n\n')

print (df2.describe(), end='\n\n')
print (df2.head(3), end='\n\n') # default 5
print (df2.tail(3), end='\n\n')

# 데이터 불러오기
DataUrl = 'https://raw.githubusercontent.com/Datamanim/pandas/main/chipo.csv'
chipoDf = pd.read_csv(DataUrl)

# Null 여부에 따라서 데이터 분석 결과에 큰 영향 끼침
chipoDf.info()

# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 4622 entries, 0 to 4621
# Data columns (total 5 columns):
#  #   Column              Non-Null Count  Dtype 
# ---  ------              --------------  ----- 
#  0   order_id            4622 non-null   int64 
#  1   quantity            4622 non-null   int64 
#  2   item_name           4622 non-null   object
#  3   choice_description  3376 non-null   object
#  4   item_price          4622 non-null   object
# dtypes: int64(2), object(3)
# memory usage: 180.7+ KB
# None


# df의 item_name 컬럼 값이 Steak Salad 또는 Bowl 인 데이터를 데이터 프레임화 한 후, item_name를 기준으로 중복행이 있으면 제거하되 첫번째 케이스만 남겨라
chipoDf[chipoDf['item_name'].isin(['Steak Salad','Bowl'])]['item_name'].drop_duplicates(keep='first')

print (chipoDf['item_name'].isin(['Steak Salad','Bowl']))
# 0       False
# 1       False
# 2       False
# 3       False
# 4       False
#         ...  
# 4617    False
# 4618    False
# 4619    False
# 4620    False
# 4621    False
# Name: item_name, Length: 4622, dtype: bool

print (chipoDf[chipoDf['item_name'].isin(['Steak Salad','Bowl'])])