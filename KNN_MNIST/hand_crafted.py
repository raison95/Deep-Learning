import sys, os
sys.path.append(os.pardir)                              # 부모 디렉토리에서 import할 수 있도록 설정
import numpy as np
from dataset.mnist import load_mnist                    # mnist data load할 수 있는 함수 import
from knn import KNN

(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=False)           # mnist data 불러오기
# training data, test data
# flatten: 이미지를 1차원 배열로 읽음
# normalize: 0~1 실수로. 그렇지 않으면 0~255

size = 100                                                # sample size
cnt = 0                                                  # sample중 computed와 true가 같은 것의 개수

x1_train = np.empty((x_train.shape[0],49))               # handcrafted train
x1_test = np.empty((x_test.shape[0],49))                 # handcrafted test

def hand_crafted():                                        # handcrafted-crafted feature를 만드는 함수
    for i in range(x_train.shape[0]):
        idx=0
        for j in range(0,28,4):
            for k in range(0,28,4):
                sum = 0
                for l in range(j,j+4):
                    for m in range(k,k+4):
                        sum=sum+x_train[i][28*l+m]
                x1_train[i][idx]=int(sum/16)                # 4*4 행렬원소의 평균값(합/개수)
                idx=idx+1

    for i in range(x_test.shape[0]):
        idx=0
        for j in range(0,28,4):
            for k in range(0,28,4):
                sum = 0
                for l in range(j,j+4):
                    for m in range(k,k+4):
                        sum=sum+x_test[i][28*l+m]
                x1_test[i][idx]=int(sum/16)                 # 4*4 행렬원소의 평균값(합/개수)
                idx=idx+1

hand_crafted()

x2_test=np.empty((size,49))                              # KNN 클래스에 인자로 들어갈 sampled x_test
t2_test=np.empty(size)                                   # sampled t_test

sample = np.random.randint(0, t_test.shape[0], size)     # sample에 사용할 0~9999중 랜덤 idx뽑기

def sampling():
    j = 0
    for i in sample:
        x2_test[j] = x1_test[i]
        t2_test[j] = t_test[i]
        j=j+1

sampling()



for K in [100]:
    print('-------------------------- K={k} --------------------------'.format(k=K))
    print('<weighted_majority_vote>')
    print('sample size = {s}'.format(s=size))
    print('-------------------------- ----- --------------------------')
    w_computed_target = np.empty(size)
    idx=0
    for i in range(size):
        w_computed_target[i] = KNN(K, x1_train, t_train, x2_test[i]).weighted_majority_vote()
        if(w_computed_target[i]==t2_test[i]):               # 계산한 결과와 원래의 target이 같다면
            cnt=cnt+1
        print('{s_num} th data result {computed_class}  label {true_class}'.format(s_num=sample[i],computed_class=int(w_computed_target[i]), true_class=int(t2_test[i])))
        idx=idx+1
    accuracy = cnt/size                                     # 정확도 = 맞은 개수 / 전체 개수
    print('accuraccy = {a}'.format(a=accuracy))

