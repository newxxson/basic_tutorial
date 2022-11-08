# 주의: 기존 코드를 수정하지 마세요.
# 주석을 통해 코드를 설명하는 것을 권장합니다.
# 세미 콜론(;)을 통한 코드 이어 붙이기는 금지입니다.

# Problem 1
# 내부 코드는 오로지 한 줄이어야 합니다.
# Hint: max 함수에 대해 알아보세요.
def p1(X):
    #한줄씩 꺼낸다
    #각 자리의 합을 구한다
    #그다음 max를 이용해서 list로 가져온다.
    return list(max(sum([int(i) for i in str(X)]))) #음! 오류난다.
#ValueError: invalid literal for int() with base 10: '['

# return list(map(lambda x: max(x, key=lambda y: sum([int(i) for i in str(y)])) ,example))    
#오류
#return max(sum(int(i) for i in X[0:]))
#list(max(sum(*X)))
# list(max(*example))

#Problem 2
# 내부 코드는 최대 세 줄이어야 합니다.
def p2(idx):
    #리스트 내에서 0이상인 숫자만 남긴다.
    # 함수를 만드는데 a에서 한개씩 뽑아온다. 
    #그리고 봅아온 숫자가 인덱스의 숫자와 같으면 1 아니면 0이 나오도록 출력한다.
    #아니 이 코드 그냥 함수로 안만들고 돌리면 돌아가는데 함수안에서 돌리면 에러납니다..                
                    
    a=list(filter(lambda x:x>=0,idx))
    b=[[(1 if a[i]== k else 0) for k in range(max(a)+1)] for i in range(len(a))]
    return b                
                    
    #idx = [5,1,0,-3,1,0,-2,4]
    #a=list(filter(lambda x:x>=0,idx1))
    #b=[[(1 if a[i]== k else 0) for k in range(max(a)+1)] for i in range(len(a))]
    #b             

# Problem 3
# 내부 코드는 최대 네 줄이어야 합니다.
# Hint: sample과 data의 크기가 클 수 있기 때문에 효율적인 자료구조를 사용하세요.
def p3(sample, data):
    #data에 있는 성과 이름 나누기(딕셔너리를 이용하여 key와 value 값으로 나누기)
    # 나올수 있는 모든 이름 조합
    # 샘플의 데이터가 만약 그 조합에 있는 new_data에 있다면 Ture를 반환하고,
                                                    #없다면 False를 반환하라.
    dic = {string[0]:string[1:3] for string in data}
    new_data = list(i + j for i in dic.keys() for j in dic.values())
    answer = list(True if sample[i] in new_data else False for i in range(len(sample)))
    return answer
    
    #dataset = set([name[0] for name in data] + [name[1:] for name in data])
    #answer = [True if (name[0] in dataset and  name[1:] in dataset) else False for name in sample]
    #return answer

# Problem 4
# 내부 코드는 최대 여섯 줄이어야 합니다.
def p4(sample, data):
    #data를 공백을 기준으로 나누고, 소문자로 변환
    # 그리고 공백을 제거하고 중복 문자 없게 한다음 알파벳 순으로 정렬한다.
    # 각 단어들의 인덱스를 순서쌍으로 반환하고 각각 index와 name로 넣어준다음 dict으로 바꾼다.
    # 익명함수를 이용하여 sample에 있는 단어가 사전에 있는 단어라면 그거의 키값을 가져오고,
    # 아니라면 새로운 인덱스()를 반환한다.
    # map을 이용하여 sample을 func를 통해 변환해준다.
    data = ' '.join(data).lower()
    dataset = sorted(set(data.split(' ')))
    #print(dataset)
    #['favorite', 'hello', 'is', 'my', 'name', 'park', 'yosemite']
    dictionary = dict([(name, index) for index, name in enumerate(dataset)])
    func = lambda x: [dictionary[word] if word in dictionary else len(dataset) for word in x.lower().split()]
    return list(map(func, sample))

class Matrix2d:
    def __init__(self, data):
        assert len(data) > 0
        self.data = [list(row) for row in data]
        self.shape = (len(data), len(data[0]))
        
    def add(self, x):
        assert self.shape == x.shape
        r, c = self.shape
        for i in range(r):
            for j in range(c):
                self.data[i][j]+=x.data[i][j]
    
    def where(self, func):
        r, c = self.shape
        for i in range(r):
            for j in range(c):
                if func(self.data[i][j]):
                    yield i, j
            
    def __eq__(self, img):
        return img.data == self.data and img.shape == self.shape
    
    def __repr__(self):
        return str([list(map(lambda x: round(x, 4), row)) for row in self.data])
    
# Problem 5
#이 문제는 시험 기간 끝나면 다시 찬찬히 보아야할듯...
# p5 함수가 정상 동작 하게끔 아래 클래스를 구현하세요.
#assert문: 원하지 않는 input값이 들어왔을때  AssertionError가 발생하는데 이때,사용자에게 어떤 종류의 조건이 만족되지 못했는지를 알려줄수 있다.


class GrayScaleImg(Matrix2d):
    def __init__(self, data):
        assert all(all(0<=x and x<=1 for x in row) for row in data)
        super().__init__(data) #일단 부모클래스를 상속받아.
    
    def ImgAdd(self, img):
        # 입력으로 들어오는 img 또한 GrayScaleImg 객체이며 해당 데이터를 현재 데이터에 더하는 메소드
        assert isinstance(img, GrayScaleImg)
        #self.add(img)
        #for row in self.data:
        #    for index, pixel in enumerate(row):
        #        if pixel >1:
        #            row[index]=1
    
    def Transpose(self):
        # 현재 데이터를 transpose 시키는 메소드
        #self.data = [[self.data[k][i] for k in range(self.shape[0])] for i in range(self.shape[1])]
        #self.shape=(len(self.data),len(self.data[0]))
        #return self
    
    def FixBrightness(self, threshold, ratio):
        # threshold 보다 높은 부분을 ratio 만큼 곱해주는 메소드. ratio는 0과 1사이 값이라는 것이 보장됩니다.
        assert 0 <= ratio and ratio <= 1
        #for row in self.data:
        #    for index, pixel in enumerate(row):
        #        if pixel > threshold:
        #            row[index]*=ratio
        #return self
    
    def Pad(self):
        # Zero-padding을 추가하는 메소드
         #height , width = self.shape
        #height+=2
        #width +=2
        #self.data = [ [0 if (i==0 or i==height-1) or (k==0 or k==width-1) else self.data[i-1][k-1] 
         #              for k in range(width)] for i in range(height)]
        #self.shape=(height,width)
        #return self

        
def p5(data, low, img_to_add, threshold, ratio):
    try:
        result = dict()
        img = GrayScaleImg(data)
        result["lower"] = list(img.where(lambda x: x < low))

        to_add = GrayScaleImg(img_to_add)
        img.ImgAdd(to_add)
        result["add"] = Matrix2d(img.data)

        img.Transpose()
        result["transpose"] = Matrix2d(img.data)

        img.FixBrightness(threshold, ratio)
        result["fix"] = Matrix2d(img.data)

        img.Pad()
        result["pad"] = Matrix2d(img.data)
        return result
    except Exception as e:
        print("[Error]", e)
        return None
    
# Problem 6
# n은 매우 큰 수가 될 수 있음을 유의하세요.
def p6(n):
    pass