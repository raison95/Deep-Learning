import numpy as np
import matplotlib.pyplot as plt

class LR:
    def __init__(self,s_or_m,class_num,x_train,y,w):
        self.s_or_m = s_or_m                                            # s_or_m : 0이면 single 1이면 MNIST multi
        self.class_num=class_num                                        # single에서 어떤 클래스에 해당되는지
        self.x_train=x_train
        self.y=y
        self.w=w

    def sigmoid(self,x):                                                # sigmoid 함수
        return (1.0 / (1 + np.exp(-x)))

    def cost(self,h_x):                                                 # log오류에 대한 처리로 epsilon이라는 작은 수를 더하여 준다.
        epsilon = 1e-7
        return ((-1/h_x.shape[0]) * (np.sum(self.y * np.log10(h_x+epsilon) + (1 - self.y) * np.log10(1 - h_x+epsilon),axis=0)))

    def learn_binary(self, i_or_M, alpha, epoch):                           # i_or_M : 0이면 iris data 1이면 MNIST data
        cost = np.empty((epoch, int(np.dot(self.x_train, self.w)[0].size)))
        for i in range(epoch):
            h_x = self.sigmoid(np.dot(self.x_train, self.w)/100)            # hypothesis 함수 (결과값이 잘 안나올때 100을 나눈다)
            cost[i] = self.cost(h_x)                                        # cost[i] 함수에서는 i번째 epoch에서 구한 cost값이다.
            print('epoch: {epoch}\tcost: {cost}'.format(epoch=i, cost=self.cost(h_x)))
            for j in range(self.w.shape[0]):
                self.w[j] = self.w[j] - alpha * (np.sum((h_x - self.y) * self.x_train[:, j]))   # 모든 weight를 갱신해준다.

        self.plotting(i_or_M, epoch, cost)                                  # plotting 함수 호출

    def learn_multi(self, i_or_M, alpha, epoch):                            # i_or_M : 0이면 iris data 1이면 MNIST data
        cost=np.empty((epoch,np.dot(self.x_train,self .w).shape[1]))
        for i in range(epoch):
            h_x = self.sigmoid(np.dot(self.x_train, self.w)/100)
            cost[i]=self.cost(h_x)                                          # hypothesis 함수 (결과값이 잘 안나올때 100을 나눈다)
            print('epoch: {epoch}\tcost: {cost}'.format(epoch=i, cost=self.cost(h_x)))
            for j in range(self.w.shape[0]):
                self.w[j] = self.w[j] - alpha * (np.sum((h_x - self.y) * self.x_train[:, j].reshape(h_x.shape[0], 1), axis=0)) # 모든 weight를 갱신해준다.

        self.plotting(i_or_M, epoch, cost)                                  # plotting 함수 호출

    def predict_binary(self, x_test, t_test):                       # binary classificaion의 predict함수
        cnt=0                                                       # accuracy 계산을 위한 카운트 변수 도입
        for i in range(int(len(t_test))):
            h_x=self.sigmoid(np.dot(x_test[i],self.w)/100)          # 위에서와 마찬가지로 결과가 잘 안나올시에 sigmoid함수에 도입전에 숫자를 작게 나누어 보낸다
            if(h_x>0.5):                                            # h_x가 0.5이상이라면 positive(실제값과 비교시 맞으면 카운트)
                if(self.class_num==t_test[i]):
                    cnt=cnt+1
            else:                                                   # h_x가 0.5미만이라면 negative(실제값과 비교시 틀리면 카운트)
                if(self.class_num!=t_test[i]):
                    cnt=cnt+1

        accuracy=cnt/len(t_test)                                    # accuracy 계산

        print('Accuracy = {Accuracy}'.format(Accuracy=accuracy))

    def predict_multi(self, x_test, t_test):                         # multiclass classificaion의 predict함수
        cnt=0                                                        # accuracy 계산을 위한 카운트 변수 도입
        for i in range(int(len(t_test))):
            h_x=self.sigmoid(np.dot(x_test[i],self.w)/100)           # 위에서와 마찬가지로 결과가 잘 안나올시에 sigmoid함수에 도입전에 숫자를 작게 나누어 보낸다
            h_max=np.max(h_x)                                        # 최대값을 가지는 h_x찾기
            h_max_idx=np.where(h_max==h_x)                           # 최대값 인덱스 저장(계산된 값)
            if(h_max_idx==t_test[i]):                                # 계산결과와 실제값이 같다면 카운트
                cnt=cnt+1

        accuracy=cnt/len(t_test)                                     # accuracy 계산

        print('Accuracy = {Accuracy}'.format(Accuracy=accuracy))

    def plotting(self, i_or_M, epoch, cost):
        if((i_or_M==0) and (self.s_or_m==0)):                   # iris data, single classification
            x = np.arange(0, epoch, 1)
            plt.plot(x,cost,label="target {t}".format(t=self.class_num))
            plt.xlabel("number of iterations")
            plt.ylabel("cost")
            plt.legend()
            plt.show()

        elif((i_or_M==0) and (self.s_or_m==1)):                 # iris data, multi classification
            x = np.arange(0, epoch, 1)
            y0 = np.empty(epoch)
            y1 = np.empty(epoch)
            y2 = np.empty(epoch)


            for i in range(epoch):
                y0[i] = cost[i][0]
                y1[i] = cost[i][1]
                y2[i] = cost[i][2]

            plt.plot(x, y0, label="target 0")
            plt.plot(x, y1, label="target 1")
            plt.plot(x, y2, label="target 2")

            plt.xlabel("number of iterations")
            plt.ylabel("cost")
            plt.legend()
            plt.show()              # i

        elif ((i_or_M == 1) and (self.s_or_m == 0)):            # MNIST data, single classification
            x = np.arange(0, epoch, 1)
            plt.plot(x,cost,label="target {t}".format(t=self.class_num))
            plt.xlabel("number of iterations")
            plt.ylabel("cost")
            plt.legend()
            plt.show()

        else:                                                   # MNIST data, multi classification
            x = np.arange(0, epoch, 1)
            y0 = np.empty(epoch)
            y1 = np.empty(epoch)
            y2 = np.empty(epoch)
            y3 = np.empty(epoch)
            y4 = np.empty(epoch)
            y5 = np.empty(epoch)
            y6 = np.empty(epoch)
            y7 = np.empty(epoch)
            y8 = np.empty(epoch)
            y9 = np.empty(epoch)

            for i in range(epoch):
                y0[i] = cost[i][0]
                y1[i] = cost[i][1]
                y2[i] = cost[i][2]
                y3[i] = cost[i][3]
                y4[i] = cost[i][4]
                y5[i] = cost[i][5]
                y6[i] = cost[i][6]
                y7[i] = cost[i][7]
                y8[i] = cost[i][8]
                y9[i] = cost[i][9]

            plt.plot(x, y0, label="target 0")
            plt.plot(x, y1, label="target 1")
            plt.plot(x, y2, label="target 2")
            plt.plot(x, y3, label="target 3")
            plt.plot(x, y4, label="target 4")
            plt.plot(x, y5, label="target 5")
            plt.plot(x, y6, label="target 6")
            plt.plot(x, y7, label="target 7")
            plt.plot(x, y8, label="target 8")
            plt.plot(x, y9, label="target 9")
            plt.xlabel("number of iterations")
            plt.ylabel("cost")
            plt.legend()
            plt.show()
