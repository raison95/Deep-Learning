import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from TwoLayerNeuralNetwork import TwoLayerNeuralNetwork

iris = load_iris()

x_train = iris.data[:8, :]                                          # 훈련 데이터 : 시험 데이터 = 8 : 2로 맞추기 위함. 클래스를 고루 나누기 위해서 10개 단위로 끊었음.
t_train = iris.target[:8]
x_test = iris.data[8:10,:]
t_test = iris.target[8:10]

for i in range(10, len(iris.data), 10):                             # 훈련 데이터 : 시험 데이터 = 8 : 2로 맞추기 위함.
    x_train= np.append(x_train, iris.data[i:i+8,:],axis=0)
    t_train = np.append(t_train, iris.target[i:i+8])

    x_test = np.append(x_test, iris.data[i+8:i+10,:],axis=0)
    t_test = np.append(t_test, iris.target[i+8:i+10])

num = np.unique(iris.target,axis=0)                                 # one-hot encoding
num = num.shape[0]
t_train=np.eye(num)[t_train]

num = np.unique(iris.target,axis=0)                                 # one-hot encoding
num = num.shape[0]
t_test=np.eye(num)[t_test]

batch_size = 10                                                     # batch_size
epoch =2000                                                         # epoch
lr=0.1                                                              # lr(learning rate)

x=np.arange(0,epoch,1)

NeuralNet = TwoLayerNeuralNetwork(x_train,t_train,4,4,3)            # 신경망 인스턴스 생성; 입력층은 feature 수가 4개이므로 4로, 은닉층은 임의로 설정, 출력층은 정답 레이블의 종류가 3개이므로 3으로.
y1,y2=NeuralNet.learn(lr,epoch,batch_size)                          # 신경망 학습; loss와 accuracy를 y1,y2에 차례로 대입.
y3 = NeuralNet.accuracy(x_test,t_test)                              # 오버피팅 되어 있는지 확인하기 위한 시험데이터로 정확도 계산

print('Training Accuracy = ', y2[epoch-1])                          # 훈련데이터의 정확도 中 마지막
print('Test Accuracy = ', y3)                                       # 시험데이터의 정확도

plt.plot(x, y1, label="loss")                                       # plot
plt.plot(x, y2, label="training accuracy")
plt.legend()
plt.show()








