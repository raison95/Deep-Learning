import sys, os
sys.path.append(os.pardir)                              # 부모 디렉토리에서 import할 수 있도록 설정
import numpy as np
from dataset.mnist import load_mnist                    # mnist data load할 수 있는 함수 import
from Logistic_Regression_class import LR

(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=True)           # mnist data 불러오기
# training data, test data
# flatten: 이미지를 1차원 배열로 읽음
# normalize: 0~1 실수로. 그렇지 않으면 0~255

x_train = np.append(np.ones((x_train.shape[0],1)),x_train,axis=1)
x_test = np.append(np.ones((x_test.shape[0],1)),x_test,axis=1)

num = np.unique(t_train,axis=0)                         # one-hot encoding
num = num.shape[0]
y=np.eye(num)[t_train]

y0=np.zeros(int(len(t_train)))
y1=np.zeros(int(len(t_train)))
y2=np.zeros(int(len(t_train)))
y3=np.zeros(int(len(t_train)))
y4=np.zeros(int(len(t_train)))
y5=np.zeros(int(len(t_train)))
y6=np.zeros(int(len(t_train)))
y7=np.zeros(int(len(t_train)))
y8=np.zeros(int(len(t_train)))
y9=np.zeros(int(len(t_train)))

w_binary = np.random.rand(int(len(x_train[0])))                 # binary classification을 위한 weight
w_multi = np.random.rand(int(len(x_train[0])),num)              # multi classification을 위한 weight

for i in range(int(len(t_train))):
    if(t_train[i]==0):
        y0[i]=1
    elif(t_train[i]==1):
        y1[i]=1
    elif(t_train[i]==2):
        y2[i]=1
    elif(t_train[i]==3):
        y3[i]=1
    elif(t_train[i]==4):
        y4[i]=1
    elif(t_train[i]==5):
        y5[i]=1
    elif(t_train[i]==6):
        y6[i]=1
    elif(t_train[i]==7):
        y7[i]=1
    elif(t_train[i]==8):
        y8[i]=1
    else:
        y9[i]=1

print('Multiple Class')
lr = LR(1,-1, x_train, y, w_multi)
lr.learn_multi(1,0.001,10)
lr.predict_multi(x_test,t_test)
print('====================================')



# print('Single Class - target class 0')
# lr0 = LR(0,0, x_train, y0, w_binary)
# lr0.learn_binary(1,0.001, 10)
# lr0.predict_binary(x_test, t_test)
# print('====================================')

#
# print('Single Class - target class 5')
# lr5 = LR(0,5, x_train, y5, w_binary)
# lr5.learn_binary(1,0.01, 100)
# lr5.predict_binary(x_test, t_test)
# print('====================================')
#
# print('Single Class - target class 9')
# lr9 = LR(0,9, x_train, y9, w_binary)
# lr9.learn_binary(1,0.01, 100)
# lr9.predict_binary(x_test, t_test)
# print('====================================')
