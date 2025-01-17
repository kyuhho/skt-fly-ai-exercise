'''
24-12-23
'''
import numpy as np

arr3d = np.array([[[4,5,6], [1, 2, 3]] , [[7,8,9], [10 , 11, 12]]])

# 원본 안 바뀌는 case
arr1 = arr3d[0,1,:2] * 10

# 원본 바뀌는 case
# arr1 = arr3d[0,1,:2]
# arr1 *= 10

print(arr3d)


arr2d = np.array([[1,2,3],[4,5,6], [7,8,9]])

print(arr2d[:2, 1:])
print(arr2d[2::])
print(arr2d[:, :2])
print(arr2d[1:2, :2])

names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
np.random.seed(0)
data = np.random.randn(7, 4)

print(data)
print(names == 'Bob')
# # index로 사용 가능하다.
print(data[names == 'Bob'])

names == 'Will'
data[names == 'Will']

# name mapping, 논리 indexing
# 마치 기존 matrix의 행마다 labeling 하듯이 구현
# 관계연산자로 묶기

mask = (names == 'Will') | (names == 'Joe')
print(mask)
print(data[mask])

# 논리 index를 이용해서 반환된 배열은 배열 값 바뀌어도 원본 값 안 바뀜

arr_m = data[mask]
arr_m[0, 0] = 100
# print(arr_m)
# print(data)

# 해당 방식은 원본을 변경, 권장되지 않음
data[data < 0] = 0
data

arr_84 = np.empty((8, 4))

for i in range(8):
    arr_84[i] = i

print("arr_84")
print(arr_84)


print(arr_84[[4, 3,0, 6]])
# 음수 index로도 가능
print(arr_84[[-3, -5, -7]])

# 여러 개의 정수 배열 인덱싱은 약간 다름. 다중 정수 배열 인덱싱 결과는 항상 1차원 배열
# 그래서 reshape을 사용해야한다.
arr = np.arange(32).reshape((8, 4))
arr = np.arange(32).reshape((2, 4, 4))

# TODO: ???
print(arr_84[[1, 5, 7, 2], [0, 3, 1, 2]])
# [1. 5. 7. 2.]
print(arr_84[[1, 5, 7, 2]][:, [0, 3, 1, 2]])

# [[1. 1. 1. 1.]
#  [5. 5. 5. 5.]
#  [7. 7. 7. 7.]
#  [2. 2. 2. 2.]]


a = [1.5, 2.5, 3.5, 4.5, 5.5]
b = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
c = [[1], [2,3], [4,5,6]]

print(np.array(a))
print(np.array(b))
# print(np.array(c))