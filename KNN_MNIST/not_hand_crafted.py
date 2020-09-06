import sys, os
sys.path.append(os.pardir)                              # 부모 디렉토리에서 import할 수 있도록 설정
import numpy as np
from dataset.mnist import load_mnist                    # mnist data load할 수 있는 함수 import
from knn import KNN

(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=False)
# training data, test data
# flatten: 이미지를 1차원 배열로 읽음
# normalize: 0~1 실수로. 그렇지 않으면 0~255

size = 100                                                # sample size
cnt = 0                                                  # sample중 computed와 true가 같은 것의 개수

x1_test=np.empty((size,784))
t1_test=np.empty(size)

sample = np.random.randint(0, t_test.shape[0], size)     # sample에 사용할 0~9999중 랜덤 idx뽑기

def sampling():
    j = 0
    for i in sample:
        x1_test[j] = x_test[i]
        t1_test[j] = t_test[i]
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
        w_computed_target[i] = KNN(K, x_train, t_train, x1_test[i]).weighted_majority_vote()
        if(w_computed_target[i]==t1_test[i]):                   # 계산한 결과와 원래의 target이 같다면
            cnt=cnt+1
        print('{s_num} th data result {computed_class}  label {true_class}'.format(s_num=sample[i],computed_class=int(w_computed_target[i]), true_class=int(t1_test[i])))
        idx=idx+1
    accuracy = cnt/size                                         # 정확도 = 맞은 개수 / 전체 개수
    print('accuraccy = {a}'.format(a=accuracy))

