import numpy as np
from sklearn.datasets import load_iris
from Logistic_Regression_class import LR

iris = load_iris()

# x_train,t_train,x_test,t_test 초기화 과정

x_train = np.append(np.ones((6,1)),iris.data[:6, :],axis=1)
t_train = iris.target[:6]
x_test = np.append(np.ones((4,1)),iris.data[6:10,:],axis=1)
t_test = iris.target[6:10]


for i in range(10, len(iris.data), 10):
    tmp1=np.append(np.ones((6,1)),iris.data[i:i+6,:].reshape(6,int(len(iris.data[0]))),axis=1)
    x_train= np.append(x_train, tmp1.reshape(6,int(len(iris.data[0]))+1), axis=0)
    t_train = np.append(t_train, iris.target[i:i+6])

    tmp2=np.append(np.ones((4,1)),iris.data[i+6:i+10,:].reshape(4,int(len(iris.data[0]))),axis=1)
    x_test = np.append(x_test, tmp2.reshape(4,int(len(iris.data[0]))+1), axis=0)
    t_test = np.append(t_test, iris.target[i+6:i+10])

num = np.unique(iris.target,axis=0)                                 # one-hot encoding
num = num.shape[0]
y=np.eye(num)[t_train]

y0=np.zeros(int(len(t_train)))
y1=np.zeros(int(len(t_train)))
y2=np.zeros(int(len(t_train)))

w_binary = np.random.rand(int(len(iris.data[0])) + 1)               # binary classification을 위한 weight
w_multi = np.random.rand(int(len(iris.data[0])) + 1,num)            # multi classification을 위한 weight


for i in range(int(len(t_train))):
    if(t_train[i]==0):
        y0[i]=1
    elif(t_train[i]==1):
        y1[i]=1
    else:
        y2[i]=1

print('Multiple Class')
lr = LR(1,-1, x_train, y, w_multi)
lr.learn_multi(0,0.00001,10000)
lr.predict_multi(x_test,t_test)
print('====================================')

# print('Single Class - target class 0')
# lr0 = LR(0,0, x_train, y0, w_binary)
# lr0.learn_binary(0,0.001,100)
# lr0.predict_binary(x_test,t_test)
# print('====================================')
#
# print('Single Class - target class 1')
# lr1 = LR(0,1, x_train, y1, w_binary)
# lr1.learn_binary(0,0.001,100)
# lr1.predict_binary(x_test,t_test)
# print('====================================')

#
# print('Single Class - target class 2')
# lr2 = LR(0,2, x_train, y2, w_binary)
# lr2.learn_binary(0,0.001,200)
# lr2.predict_binary(x_test,t_test)
# print('====================================')





