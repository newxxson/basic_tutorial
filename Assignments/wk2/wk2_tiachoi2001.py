#!/usr/bin/env python
# coding: utf-8

# # 과제

# ### Problem 1
# 
# numpy를 활용해 어떤 행렬이 singular matrix인지 확인하는 함수를 작성하세요.
# 
# (* singular matrix : 역행렬이 존재하지 않은 행렬)
# 
# - 매개변수 : 2차원 벡터(np.array)
# - 반환값 : 인자로 주어진 벡터가 singular하면 True, non-singular하면 False를 반환

# In[1]:


import numpy as np

# 내부 코드는 오로지 한 줄이어야 합니다.
# Hint: 역행렬의 존재 조건을 고민해보세요.

def p1(x):
    return np.linalg.det(x) == 0


# In[2]:


ex_1 = np.array([[1,4],[2,8]])
ex_2 = np.array([[2,3],[3,4]])

answer = (True, False)
your_work = (p1(ex_1), p1(ex_2))

print("your answer:", your_work)
print("correct" if answer is not None and answer == your_work else "wrong")


# ### Problem 2

# numpy를 활용해 어떤 벡터가 주어졌을 때 L2 norm을 구하는 함수를 작성하세요.
# 
# - 매개변수 : 1차원 벡터 (np.array)
# - 반환값 : 인자로 주어진 벡터의 L2 Norm값 (number)

# In[3]:


#import

# 내부 코드는 두 줄이어야 합니다.

def p2(x):
    s = np.sum(np.square(x))
    return np.sqrt(s)


# In[4]:


ex_1 = np.array([3,3,1,3])
print(p2(ex_1))
#ex_1 = np.array([[1,4,5],[2,8,3]])
#ex_2 = np.array([[2,3],[3,4]])

#answer = (10.560835223390939, 6.162277660168379)
#your_work = (p2(ex_1), p2(ex_2))

#print("your answer:", your_work)
#print("correct" if answer is not None and answer == your_work else "wrong")


# ### Problem 3
# 
# (wk1 과제 2번 문항에서 풀었던 문항과 동일합니다.)
# 
# one-hot encoding이란 인덱스의 집합이 주어졌을 때, 해당 인덱스에만 1을 부여하고 나머지 인덱스에는 0을 부여하여 데이터를 표현하는 방식입니다.
# 
# 예를 들어, 주어진 인덱스의 집합이 [3, 0, 2, 1] 이라면 해당 집합을 다음과 같은 2차원 행렬로 표현할 수 있습니다.
# 
# \begin{matrix}
# 0 & 0 & 0 & 1 \\
# 1 & 0 & 0 & 0 \\
# 0 & 0 & 1 & 0 \\
# 0 & 1 & 0 & 0
# \end{matrix}
# 
# 크기 N의 정수 리스트가 주어집니다. 인덱스는 음수가 될 수 없기 때문에 음수 데이터를 삭제 한 이후, one-hot encoding이 되어있는 2차원 리스트를 구하세요.
# 
# 주어지는 리스트는 0 이상의 정수가 적어도 하나 존재함이 보장됩니다.
# 
# 단, 함수 내부 코드는 최대 **9 줄**이어야 하며 **numpy를 활용**해야 합니다.

# In[5]:


#import

# 내부 코드는 오로지 최대 9 줄이어야 합니다.
# Hint: 0을 만드는 행렬을 중심으로 조건을 바꿔보세요.

def p3(arr):
    narr = [n for n in arr if n >= 0]
    marr = np.zeros((len(narr),len(narr)))
    for i in range(len(narr)):
        marr[i,narr[i]] = 1
    li = marr.astype(int).tolist()
    return li


# In[6]:


example = [5,1,0,-3,1,0,-2,4]
answer = [
    [0, 0, 0, 0, 0, 1],
    [0, 1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0]]
your_work = p3(example)

print("your answer:", your_work)
print("correct" if answer is not None and answer == your_work else "wrong")


# ### Problem 4
# 
# 1.
# 2011년부터 2020년까지의 평균 영화 러닝타임은 각각 103, 101, 99, 100, 100, 95, 95, 96, 93, 90입니다.
# 
# 연도와 러닝타임을 각각 List로 만들고, 이 리스트들을 활용해 movie_df DataFrame을 만드세요.

# In[7]:


import pandas as pd

# (1) 연도와 러닝타임 list 만들기
# 연도 리스트는 for문을 활용해 만드세요.(내부 코드는 오로지 한 줄이어야 합니다.)
p3_years = [i for i in range(2011,2021)]
p3_durations = [103, 101, 99, 100, 100, 95, 95, 96, 93, 90]

# (2) 각 리스트의 같은 인덱스끼리 같은 열에 취급하는 DataFrame 만들기
# 코드는 오로지 한 줄이어야 합니다.
# Hint1: 지난 시간에 배운 generator를 활용해보세요.
# Hint2: 열 이름을 지정하기 위해 columns=["years", "durations"] 활용해보세요.
movie_df = pd.DataFrame(list(zip(p3_years, p3_durations)), columns = ["years","durations"])
movie_df


# 2.
# movie_df를 matploblib을 활용해 years와 durations의 관계를 나타내는 Line plot을 그리세요.
# 
# 제목은 "Average Movie Durations 2011-2020"으로 설정하세요.

# In[8]:


import matplotlib.pyplot as plt

# matplotlib을 활용해 years와 durations의 관계를 나타내는 line plot 그리기
# 코드는 오로지 한 줄이어야 합니다.
# Hint: fig에 객체에 그래프를 저장해보세요.
fig = movie_df.plot(kind='line',x='years',y='durations',title="Average Movie Durations 2011-2020")


# ### Problem 5
# 
# 1.
# pandas 라이브러리를 활용해 <code>"datasets/netflix_data.csv"</code>을 불러 netflix_df에 DataFrame 형식으로 지정하고, 10번째 열까지 출력합니다.

# In[9]:


#import

# (1) pandas를 활용해 csv를 불러오기
# 코드는 오로지 한 줄이어야 합니다.
netflix_df = pd.read_csv("netflix_data.csv") 

# (2) netflix_df의 10번째 열까지 출력하기
# 코드는 오로지 한 줄이어야 합니다.
netflix_df.head(10)


# 2.
# netflix_df에서 'type' 중에 'Movie'가 아닌 데이터가 있음을 확인할 수 있습니다. 위 DataFrame에서 'type'이 'Movie'면서 'genre'가 'Comedies', 'Dramas', 'Action', 'Horror Movies'인 데이터만 필터링해 분석하려고 합니다. 이를 새로운 DataFrame을 netflix_movies_genre_subset에 지정해보세요.

# In[10]:


# netflix_df에서 'type'이 'Movie'면서 'genre'가 'Comedies', 'Dramas', 'Action', 'Horror Movies'인 데이터만 필터링하기
# 코드는 오로지 한 줄이어야 합니다.
netflix_movies_genre_subset = netflix_df[(netflix_df.type == 'Movie') & netflix_df.genre.isin(['Comedies', 'Dramas', 'Action', 'Horror Movies'])]
netflix_movies_genre_subset[:10]


# 3.
# netflix_movies_subset를 matploblib을 활용해 years와 durations의 관계를 나타내는 Scatter plot을 그리세요. 
# 
# - matplotlib이 아니라 seaborn을 활용해 scatter plot을 그리세요.
# - dot size는 20으로, color는 **카테고리형 데이터** 인 'genre'의 종류 별로 지정하세요.
# - 제목은 "Movie Durations by Release Year"로, xlabel은 "Release year"로, ylabel은 "Duration (min)"로 설정하세요.

# In[11]:


import seaborn as sns

# seaborn을 활용해 scatterplot을 그리기
# 코드는 오로지 한 줄이어야 합니다.
# Hint1: 색상은 color가 아니라, 카테고리형 데이터의 종류 별로 나눠주는 변수를 활용해 지정해보세요.
# Hint2: 제목이나 라벨은 .set()을 활용해보세요.
sns.scatterplot(x="release_year", y = "duration",data = netflix_movies_genre_subset, hue="genre", palette = {'Dramas':'darkviolet','Horror Movies':'black','Action':'powderblue','Comedies':'palevioletred'}, sizes = 20).set(title = "Movie Durations by Release Year",xlabel="Release year", ylabel="Duration (min)")

