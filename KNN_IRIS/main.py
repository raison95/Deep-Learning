import numpy as np
from sklearn.datasets import load_iris
from knn import KNN

iris = load_iris()

for K in [3, 5, 10]:
    print('-------------------------- K={k} --------------------------'.format(k=K))

    train_data = iris.data[:14,:]
    train_target = iris.target[:14]
    test_data = iris.data[14,:].reshape(1,int(len(iris.data[0])))
    test_target = iris.target[14]

    for i in range(15, len(iris.data), 15):
        train_data = np.append(train_data, iris.data[i:i+14,:].reshape(14,int(len(iris.data[0]))), axis=0)
        train_target = np.append(train_target, iris.target[i:i+14])
        test_data = np.append(test_data, iris.data[i+14,:].reshape(1,int(len(iris.data[0]))), axis=0)
        test_target = np.append(test_target, iris.target[i+14])

    computed_target = np.zeros(int(len(iris.data) / 15))
    w_computed_target = np.zeros(int(len(iris.data) / 15))

    for i in range(len(computed_target)):
        tmp = KNN(K, train_data, train_target, test_data[i,:])
        computed_target[i] = tmp.majority_vote()
        w_computed_target[i] = tmp.weighted_majority_vote()

    print('<majority_vote>')
    for i in range(len(computed_target)):
        print('Test Data Index: {idx} Computed class: {computed_class}, True class: {true_class}'.format(idx=i,
        computed_class = iris.target_names[int(computed_target[i])], true_class=iris.target_names[int(test_target[i])]))

    print('<weighted_majority_vote>')
    for i in range(len(w_computed_target)):
        print('Test Data Index: {idx} Computed class: {computed_class}, True class: {true_class}'.format(idx=i,
        computed_class = iris.target_names[int(w_computed_target[i])], true_class=iris.target_names[int(test_target[i])]))
