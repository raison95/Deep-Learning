import numpy as np

class TwoLayerNeuralNetwork():
    def __init__(self, x, t, input_size, hidden_size, output_size):                 # 생성자
        self.x=x                                                                    # 초기의 x는 120개의 훈련데이터 모두를 불러온다.
        self.t=t                                                                    # 120개의 x에 해당되는 정답 레이블을 불러온다.

        self.params = {}                                                            # class의 attribute로 params라는  딕셔너리 변수를 선언
        self.params['W1'] = np.random.randn(input_size, hidden_size)*0.01           # 입력층과 은닉층 사이의 가중치
        self.params['b1'] = np.zeros(hidden_size)                                   # 입력층과 은닉층 사이의 편향
        self.params['W2'] = np.random.randn(hidden_size, output_size)*0.01          # 은닉층과 출력층 사이의 가중치
        self.params['b2'] = np.zeros(output_size)                                   # 은닉층과 출력층 사이의 편향

    def sigmoid(self,x):                        # sigmoid 함수
        return (1 / (1 + np.exp(-x)))

    def softmax(self,x):                        # softmax 함수
        exp_a = np.exp(x)
        sum_exp_a = np.sum(exp_a)
        return (exp_a / sum_exp_a)

    def cross_entropy_error(self, y, t):        # 교차 엔트로피 오차함수
        if (y.ndim == 1):                       # batch 처리한 안한 경우 2차원으로 만들어준다.
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)
        batch_size = y.shape[0]
        return -np.sum(t * np.log(y + 1e-7)) / batch_size       # 교차 엔트로피 오차의 평균을 구한다.

    def predict(self, x):                                                   # 계산된 값을 리턴하는 predict함수; x의 차원(batch 유무)에 따라 구분한다.
        if(x.ndim==1):                                                      # batch를 하지 않은 경우
            a1 = np.dot(x, self.params['W1']) + self.params['b1']
            z1 = self.sigmoid(a1)                                           # sigmoid함수에 집어 넣어 0과 1사이의 값을 갖도록 한다.

            a2 = np.dot(z1, self.params['W2']) + self.params['b2']
            y = self.softmax(a2)                                            # softmax함수에 집어 넣어 레이블에 대한 확률을 나타낸다.
            return y

        else:                                                               # batch를 한 경우
            y=np.zeros((x.shape[0],self.params['b2'].shape[0]))
            for i in range(x.shape[0]):
                a1 = np.dot(x[i], self.params['W1']) + self.params['b1']
                z1 = self.sigmoid(a1)                                       # sigmoid함수에 집어 넣어 0과 1사이의 값을 갖도록 한다.

                a2 = np.dot(z1, self.params['W2']) + self.params['b2']
                y[i]=self.softmax(a2)                                       # softmax함수에 집어 넣어 레이블에 대한 확률을 나타낸다.
            return y

    def loss(self, x, t):                                                   # 지표로 사용할 손실함수의 결과값을 받아오는 함수
        y = self.predict(x)                                                 # 정답 레이블과 비교하기 위해 y값을 계산한다.
        return self.cross_entropy_error(y, t)                               # 교차 엔트로피 오차를 구한다.

    def accuracy(self, x, t):                                           # accuracy 계산 함수
        cnt = 0
        for i in range(x.shape[0]):                                     # batch수 만큼 루프를 돌린다.
            if(np.argmax(self.predict(x[i]))==np.argmax(t[i])):         # 계산된 y의 값중 가장 큰 확률을 가지는 레이블과 정답 레이블이 일치하는 경우
                cnt += 1

        return float(cnt)/float(x.shape[0])                                    # float형으로 리턴을 받기위해 형변환을 해준다.

    def get_gradient(self, x, t, params, w_or_b, one_or_two):       # params : 가중치 또는 편향, w_or_b : 가중치냐 편향이냐를 알려주는 변수, one_or_two : 입력-은닉 사이의 값이냐 은닉-출력사이의 값이냐를 알려주는 변수
        h = 1e-4                                                    # 수치 미분에서 h->0일때를 나타내는 작은 값
        ret = np.zeros_like(params)                                 # 해당 함수를 가중치 또는 편향으로 편미분할때, 형태는 가중치 또는 편향과 같다. 따라서 ret을 할당해준다.
        if((w_or_b=='W') and (one_or_two ==1)):                     # params가 입력-은닉 사이의 가중치일때
            for i in range(params.shape[0]):                        # 매 원소별로 (f(x+h)-f(x-h))/2*h 를 실행하여 ret에 저장한다
                for j in range(params.shape[1]):
                    tmp = self.params['W1'][i][j]
                    self.params['W1'][i][j] = tmp + h
                    f_plus = self.loss(x, t)
                    self.params['W1'][i][j] = tmp - h
                    f_minus = self.loss(x, t)
                    self.params['W1'][i][j] = tmp
                    ret[i][j] = (f_plus-f_minus)/(2*h)

        elif((w_or_b=='W') and (one_or_two ==2)):                   # params가 은닉-출력 사이의 가중치일때
            for i in range(params.shape[0]):                        # 매 원소별로 (f(x+h)-f(x-h))/2*h 를 실행하여 ret에 저장한다
                for j in range(params.shape[1]):
                    tmp = self.params['W2'][i][j]
                    self.params['W2'][i][j] = tmp + h
                    f_plus = self.loss(x, t)
                    self.params['W2'][i][j] = tmp - h
                    f_minus = self.loss(x, t)
                    self.params['W2'][i][j] = tmp
                    ret[i][j] = (f_plus-f_minus)/(2*h)

        elif((w_or_b=='b') and (one_or_two ==1)):                   # params가 입력-은닉 사이의 편향일때
            for i in range(params.shape[0]):                        # 매 원소별로 (f(x+h)-f(x-h))/2*h 를 실행하여 ret에 저장한다
                tmp = self.params['b1'][i]
                self.params['b1'][i] = tmp + h
                f_plus = self.loss(x, t)
                self.params['b1'][i] = tmp - h
                f_minus = self.loss(x, t)
                self.params['b1'][i] = tmp
                ret[i] = (f_plus - f_minus) / (2 * h)

        else:                                                       # params가 은닉-출력 사이의 편향일때
            for i in range(params.shape[0]):                        # 매 원소별로 (f(x+h)-f(x-h))/2*h 를 실행하여 ret에 저장한다
                tmp = self.params['b2'][i]
                self.params['b2'][i] = tmp + h
                f_plus = self.loss(x, t)
                self.params['b2'][i] = tmp - h
                f_minus = self.loss(x, t)
                self.params['b1'][i] = tmp
                ret[i] = (f_plus - f_minus) / (2 * h)

        return ret                                                  # 가중치 또는 편향의 기울기를 리턴한다

    def numerical_gradient(self, x, t):                                         # 기울기(가중치와 편향 모두의)를 리턴하는 함수
        gradients={}
        gradients['W1'] = self.get_gradient(x,t,self.params['W1'],'W',1)        # 입력-은닉 사이의 가중치 초기화
        gradients['b1'] = self.get_gradient(x,t,self.params['b1'],'b',1)        # 입력-은닉 사이의 편향 초기화
        gradients['W2'] = self.get_gradient(x,t,self.params['W2'],'W',2)        # 은닉-출력 사이의 가중치 초기화
        gradients['b2'] = self.get_gradient(x,t,self.params['b2'],'b',2)        # 은닉-출력 사이의 편향 초기화
        return gradients                                                        # 기울기 리턴

    def learn(self, lr, epoch, batch_size):
        loss = []                                                           # plot하기 위한 손실함수 리턴값 리스트 할당
        training_accuracy = []                                              # plot하기 위한 정확도함수 리턴값 리스트 할당

        for i in range(epoch):                                              # 매개변수로 받은 epoch횟수 만큼
            batch_idx = np.random.choice(self.x.shape[0],batch_size)        # 인스턴스 생성시 초기화 된 훈련 데이터 수,즉,[0,훈련 데이터 수) 중의 숫자중에서 batch_size 개수만큼 랜덤으로 뽑는다.
            batch_x = self.x[batch_idx]                                     # x를 batch_size만큼 랜덤하게 뽑아낸다.
            batch_t = self.t[batch_idx]                                     # t를 batch_size만큼 랜덤하게 뽑아낸다

            gradients = self.numerical_gradient(batch_x,batch_t)            # 미니 배치의 손실 함수값을 줄이기 위한 매개변수의 기울기를 받아온다.

            self.params['W1'] -= lr * gradients['W1']                       # 매개변수의 갱신 과정; lr는 학습률
            self.params['b1'] -= lr * gradients['b1']
            self.params['W2'] -= lr * gradients['W2']
            self.params['b2'] -= lr * gradients['b2']

            loss.append(self.loss(batch_x,batch_t))                         # 나중에 plot하기 위해 리스트에 추가

            training_accuracy.append(self.accuracy(batch_x,batch_t))        # 나중에 plot하기 위해 리스트에 추가

        return loss,training_accuracy                                       # plot하기 위해 리턴
