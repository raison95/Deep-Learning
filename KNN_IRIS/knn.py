import numpy as np

class KNN:
    def __init__(self, k, train_data, train_target, test_data):
        self.K = k
        self.train_data = train_data
        self.train_target = train_target
        self.test_data = test_data

    def distance(self, idx):
        return np.sqrt(np.sum(np.power((self.train_data[idx] - self.test_data), 2)))

    def obtain_k(self):
        train_size = int(len(self.train_data))
        arr = np.empty(train_size)

        for i in range(0, train_size):
            arr[i] = self.distance(i)

        tmp = np.sort(arr)
        ret = np.empty(self.K)

        for i in range(self.K):
            for j in range(train_size):
                if(tmp[i]==arr[j]):
                    ret[i]=j
                    break

        return ret

    def obtain_weighted_k(self):
        train_size = int(len(self.train_data))
        arr = np.empty(train_size)

        for i in range(0, train_size):
            arr[i] = 1/self.distance(i)

        tmp = np.sort(arr)[::-1]
        ret = np.empty(self.K)

        for i in range(self.K):
            for j in range(train_size):
                if(tmp[i]==arr[j]):
                    ret[i]=j
                    break

        return ret, tmp[:self.K]

    def majority_vote(self):
        arr = self.obtain_k()
        arr2 = {}

        for i in range(self.K):
            arr2[self.train_target[int(arr[i])]] = 0

        for i in range(self.K):
            arr2[self.train_target[int(arr[i])]] = arr2[self.train_target[int(arr[i])]]+1

        print('majority:{arr2}'.format(arr2=arr2))
        freq_max = arr2[self.train_target[int(arr[0])]]
        ret = self.train_target[int(arr[0])]

        for i in range(self.K):
            j = self.train_target[int(arr[i])]
            if(freq_max < arr2[j]):
                freq_max = arr2[j]
                ret = j

        return ret

    def weighted_majority_vote(self):
        idx, arr = self.obtain_weighted_k()
        arr2 = {}

        for i in range(self.K):
            arr2[self.train_target[int(idx[i])]] = 0

        for i in range(self.K):
            arr2[self.train_target[int(idx[i])]] += arr[i]

        print('weighted_majority:{arr2}'.format(arr2=arr2))
        freq_max = arr2[self.train_target[int(idx[0])]]
        ret = self.train_target[int(idx[0])]

        for i in range(self.K):
            j = self.train_target[int(idx[i])]
            if(freq_max < arr2[j]):
                freq_max = arr2[j]
                ret = j

        return ret
