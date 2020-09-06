# coding: utf-8
# 2020/인공지능/final/B******/***
import sys
import os
from collections import OrderedDict
import pickle
import numpy as np
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정

# python train.py --sf="params.pkl" --epochs=2000 --mini_batch_size=100 --learning_rate=0.02
# python test.py --sf="params.pkl"

def sigmoid(x):         # sigmoid함수
    eMin = -np.log(np.finfo(type(0.1)).max)
    xSafe = np.array(np.maximum(x, eMin))
    return (1.0/(1+np.exp(-xSafe)))

def softmax(x):         # softmax 함수
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)

        return y.T

    x = x - np.max(x)  # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))

def cross_entropy_error(y, t):          # 교차 엔트로피 오차
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

class Relu:                             # ReLU 계층
    def __init__(self):
        self.mask = None

    def forward(self, x):               # 순전파. x가 0보다 작거나 같은 경우는 0, 0보다 큰 경우는 그대로 출력.
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):           # 역전파. x가 0보다 작거나 같은경우에는 0, 0보다 큰 경우는 1.
        dout[self.mask] = 0
        dx = dout

        return dx

class Sigmoid:          # Sigmoid 계층
    def __init__(self):
        self.out = None

    def forward(self, x):               # 순전파. 위에서 정의한 sigmoid함수를 사용함
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):           # 역전파. sigmoid의 경우 순전파의 출력값*(1-순전파의 출력값)으로 나타낼수 있다.
        dx = dout * (1.0 - self.out) * self.out

        return dx

class Affine:       # Affine 계층
    def __init__(self, W, b):
        self.W=W
        self.b=b
        self.x=None
        self.dw=None
        self.db=None

    def forward(self, x):           # 순전파. y=xw+b
        self.x=x
        out=np.dot(x,self.W)+self.b

        return out

    def backward(self, dout):       # 역전파. 순전파에서 편향의 경우 각각의 데이터에 더해지므로, 역전파시에는 한군데로 다시 모아야한다.
        dx=np.dot(dout,self.W.T)
        self.dW=np.dot(self.x.T, dout)
        self.db=np.sum(dout,axis=0)

        return dx

class SoftmaxWithLoss:        # Softmax와 loss를 합친 계층
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):            # 순전파. 위에서 정의한 softmax함수와 cross_entropy_error함수를 이용
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y,self.t)

        return self.loss

    def backward(self, dout=1):         # 역전파. batch_size로 나누어서 데이터 1개당 오차를 앞 계층으로 전파하게 된다.
        batch_size = self.t.shape[0]
        dx = (self.y-self.t) / batch_size
        return dx

class Dropout:              # 드롭아웃
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:                                   # 순전파. 훈련중일땐 dropout_criteria보다 큰 값만 선택.
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:                                           # 시험중일땐 비율을 빼서 곱한다.
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask

class BatchNormalization:       # 배치 정규화 : 평균 0,분산 1로 만들어주고, 확대(gamma)와 이동(beta)을 이용해 적절한 값으로 조정
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None

        self.running_mean = running_mean
        self.running_var = running_var

        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)

        return out.reshape(*self.input_shape)

    def __forward(self, x, train_flg):              # 실질적 순전파
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if train_flg:
            mu = x.mean(axis=0)                     # 평균
            xc = x - mu                             # 편차
            var = np.mean(xc ** 2, axis=0)          # 편차 제곱의 평균 = 분산
            std = np.sqrt(var + 10e-7)              # 분산의 제곱근 = 표준편차
            xn = xc / std                           # x-평균/표준편차 -> 표준화된 변수(평균0, 분산1)

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))

        out = self.gamma * xn + self.beta           # 확대(gamma)와 이동(beta)을 시켜줌.
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):             # 실질적 역전파. 위에서 구한 통계량과 계산 그래프를 이용하여 역전파를 구할 수 있다.
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx

class SGD:          # SGD 최적화
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr*grads[key]

class CustomOptimizer:          # AdaGrad
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:                  # 첫번째 루프일때
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():           # params 갱신
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)

class Model:
    """
    네트워크 모델 입니다.

    """

    def __init__(self, lr=0.01):
        """
        클래스 초기화
        """
        self.params = {}                                        # 매개변수를 저장할 params 딕셔너리변수 선언
        self.layers = OrderedDict()                             # backpropagation을 위해 순서가 있는 딕셔너리 변수 사용
        self.last_layer = None                                  # 마지막의 계층. softmax_with_loss로 사용할 예정
        self.input_size = 6                                     # input feature가 6이므로 input layer의 노드 수는 6
        self.hidden_node_size = [100,100,100]                   # 은닉층 구조를 나타내는 리스트
        self.hidden_layer_num = len(self.hidden_node_size)      # 은닉층은 몇개인가
        self.output_size = 6                                    # output class가 6이므로 output layer의 노드 수는 6
        self.activation = 'relu'                                # 'relu' or 'sigmoid'
        self.weight_init_method = 'he'                          # 'he' or 'xavier'
        self.lambda_value = 0.1                                 # L2 놈의 람다
        self.dropout_flg = True                                 # dropout을 사용할 것인가
        self.dropout_criteria = 0.1                             # dropout을 할때 비교대상의 변수
        self.batchnorm_flg = True                               # 배치 정규화를 사용할 것인가

        self.__init_weight()                                    # 가중치의 초기화
        self.__init_layer()                                     # 초기화된 가중치로부터 계층 생성

        self.optimizer = CustomOptimizer(lr)                    # 최적화.이 때는 구현해놓은 Adagrad사용

    def __init_layer(self):                                     # 출력층을 제외하면 affine->batch normalization->activation->dropout
                                                                # 출력층은 affine->softmaxwithloss
        """
        레이어를 생성하시면 됩니다.
        """

        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu}      # 활성화함수는 두가지 사용. sigmoid와 relu.

        for idx in range(1, self.hidden_layer_num + 1):            # 층의 idx에 따라 params및 layer초기화
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                      self.params['b' + str(idx)])
            if self.batchnorm_flg:                              #  배치 정규화를 할 경우
                self.params['gamma' + str(idx)] = np.ones(self.hidden_node_size[idx - 1])
                self.params['beta' + str(idx)] = np.zeros(self.hidden_node_size[idx - 1])
                self.layers['BatchNorm' + str(idx)] = BatchNormalization(self.params['gamma' + str(idx)],
                                                                         self.params['beta' + str(idx)])

            self.layers['Activation_function' + str(idx)] = activation_layer[self.activation]()

            if self.dropout_flg:                                # 드롭아웃을 할 경우
                self.layers['Dropout' + str(idx)] = Dropout(self.dropout_criteria)

        idx = self.hidden_layer_num + 1                         # 마지막 출력층의 경우 affine->softamaxwithloss
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])

        self.last_layer = SoftmaxWithLoss()

    def __init_weight(self):
        """
        레이어에 탑재 될 파라미터들을 초기화 하시면 됩니다.
        """

        network_node_size = [self.input_size] + self.hidden_node_size + [self.output_size]      # 신경망의 모든 층의 크기를 담은 리스트
        for idx in range(1, len(network_node_size)):
            if self.weight_init_method is ('he'):                                               # ReLU를 사용할 때 he 초기화
                scale = np.sqrt(2.0 / network_node_size[idx - 1])
            elif self.weight_init_method is ('xavier'):                                         # sigmoid를 사용할 때 xavier 초기화
                scale = np.sqrt(1.0 / network_node_size[idx - 1])
            self.params['W' + str(idx)] = scale * np.random.randn(network_node_size[idx-1], network_node_size[idx])
            self.params['b' + str(idx)] = scale * np.random.randn(network_node_size[idx])

    def update(self, x, t):
        """
        train 데이터와 레이블을 사용해서 그라디언트를 구한 뒤
         옵티마이저 클래스를 사용해서 네트워크 파라미터를 업데이트 해주는 함수입니다.

        :param x: train_data
        :param t: test_data
        """
        grads = self.gradient(x, t)                     # 기울기를 구한다.
        self.optimizer.update(self.params, grads)       # 가중치와 기울기를 넘겨주고 optimizer별로 구현해놓은 update함수를 실행

    def predict(self, x, train_flg=False):
        """
        데이터를 입력받아 정답을 예측하는 함수입니다.

        :param x: data
        :return: predicted answer
        """
        for key, layer in self.layers.items():
            if "Dropout" in key or "BatchNorm" in key:          # dropout이나 배치정규화 계층의 경우
                x = layer.forward(x, train_flg)
            else:                                               # dropout이나 배치정규화가 아닌 계층의 경우
                x = layer.forward(x)

        return x

    def loss(self, x, t, train_flg=False):
        """
        데이터와 레이블을 입력받아 로스를 구하는 함수입니다.
        :param x: data
        :param t: data_label
        :return: loss
        """
        y = self.predict(x, train_flg)

        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(idx)]
            weight_decay += 0.5 * self.lambda_value * np.sum(W ** 2)        # L2 노름사용

        return self.last_layer.forward(y, t) + weight_decay                 # L2 Regularization

    def gradient(self, x, t):
        """
        train 데이터와 레이블을 사용해서 그라디언트를 구하는 함수입니다.
        첫번째로 받은데이터를 forward propagation 시키고,
        두번째로 back propagation 시켜 grads에 미분값을 리턴합니다.
        :param x: data
        :param t: data_label
        :return: grads
        """
        # 순전파 과정
        self.loss(x,t,train_flg=True)

        # 역전파 과정
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()                # 순전파의 역순으로 역전파
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        for idx in range(1, self.hidden_layer_num+2):               # 각 층 별로 가중치와 편향의 기울기를 구한다.
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW + self.lambda_value * self.params['W' + str(idx)]
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

            if self.batchnorm_flg and idx != self.hidden_layer_num+1:           # 배치 정규화를 사용하는 경우(마지막 출력층 제외)
                grads['gamma' + str(idx)] = self.layers['BatchNorm' + str(idx)].dgamma
                grads['beta' + str(idx)] = self.layers['BatchNorm' + str(idx)].dbeta

        return grads

    def save_params(self, file_name="params.pkl"):
        """
        네트워크 파라미터를 피클 파일로 저장하는 함수입니다.

        :param file_name: 파라미터를 저장할 파일 이름입니다. 기본값은 "params.pkl" 입니다.
        """

        # 저장할 params와 layers를 pickle파일에 저장.

        params = {}
        layers = OrderedDict()
        for key, val in self.params.items():
            params[key] = val
        for key, val in self.layers.items():
            layers[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)
            pickle.dump(layers, f)

    def load_params(self, file_name="params.pkl"):
        """
        저장된 파라미터를 읽어와 네트워크에 탑재하는 함수입니다.

        :param file_name: 파라미터를 로드할 파일 이름입니다. 기본값은 "params.pkl" 입니다.
        """

        # save_params에서 저장한 params와 layers를 현재 모델에 불러오기.

        with open(file_name, 'rb') as f:
            params = pickle.load(f)
            layers = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val
        for key, val in layers.items():
            self.layers[key] = val

