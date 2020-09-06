import numpy as np
from scipy.stats import skew
from scipy.stats import kurtosis
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd
import stemgraphic

iris = load_iris()

raw_data1 = iris.data[:100,:]
raw_data2 = iris.target[:100]
raw_data = np.append(raw_data1,raw_data2.reshape(100,1),axis=1)
data = pd.DataFrame(raw_data,columns=['sepal length', 'sepal width', 'petal length', 'petal width', 'iris class'])

def scatter_plot():
    x0_min = raw_data[:,0].min()-.5
    x1_min = raw_data[:,1].min()-.5
    x2_min = raw_data[:,2].min()-.5
    x3_min = raw_data[:,3].min()-.5

    x0_max = raw_data[:,0].max()+.5
    x1_max = raw_data[:,1].max()+.5
    x2_max = raw_data[:,2].max()+.5
    x3_max = raw_data[:,3].max()+.5

    # plt.figure(2, figsize=(8, 6))
    plt.subplot(211)
    plt.scatter(raw_data[:, 0], raw_data[:, 1], c=raw_data[:,4], cmap=plt.cm.Set1,edgecolor='k')
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(x0_min, x0_max)
    plt.ylim(x1_min, x1_max)
    plt.xticks(())
    plt.yticks(())

    plt.subplot(212)
    plt.scatter(raw_data[:, 2], raw_data[:, 3], c=raw_data[:,4], cmap=plt.cm.Set1,edgecolor='k')
    plt.xlabel('petal length')
    plt.ylabel('petal width')
    plt.xlim(x2_min, x2_max)
    plt.ylim(x3_min, x3_max)
    plt.xticks(())
    plt.yticks(())

    plt.show()

scatter_plot()


print('(sepal length - sepal width) corrcoef : {corr}'.format(corr=np.corrcoef(raw_data[:,0],raw_data[:,1])[0,1]))
print('(petal length - petal width) corrcoef : {corr}'.format(corr=np.corrcoef(raw_data[:,2],raw_data[:,3])[0,1]))




# 평균, 중앙값, 사분위수 범위, 범위, 표준편차, 첨도, 왜도
mean0=np.mean(raw_data[:, 0])
mean1=np.mean(raw_data[:, 1])
mean2=np.mean(raw_data[:, 2])
mean3=np.mean(raw_data[:, 3])

print('<평균>')
print('sepal length : {mean0}'.format(mean0=mean0))
print('sepal width : {mean1}'.format(mean1=mean1))
print('petal length : {mean2}'.format(mean2=mean2))
print('petal width : {mean3}'.format(mean3=mean3))

median0=np.median(raw_data[:, 0])
median1=np.median(raw_data[:, 1])
median2=np.median(raw_data[:, 2])
median3=np.median(raw_data[:, 3])

print('<중앙값>')
print('sepal length : {median0}'.format(median0=median0))
print('sepal width : {median1}'.format(median1=median1))
print('petal length : {median2}'.format(median2=median2))
print('petal width : {median3}'.format(median3=median3))

q0_1=np.percentile(raw_data[:, 0],25)
q0_3=np.percentile(raw_data[:, 0],75)
q0=q0_3-q0_1

q1_1=np.percentile(raw_data[:, 1],25)
q1_3=np.percentile(raw_data[:, 1],75)
q1=q1_3-q1_1

q2_1=np.percentile(raw_data[:, 2],25)
q2_3=np.percentile(raw_data[:, 2],75)
q2=q2_3-q2_1

q3_1=np.percentile(raw_data[:, 3],25)
q3_3=np.percentile(raw_data[:, 3],75)
q3=q3_3-q3_1

print('<사분위수 범위>')
print('sepal length : {q0}'.format(q0=q0))
print('sepal width : {q1}'.format(q1=q1))
print('petal length : {q2}'.format(q2=q2))
print('petal width : {q3}'.format(q3=q3))

max0=np.max(raw_data[:, 0])
max1=np.max(raw_data[:, 1])
max2=np.max(raw_data[:, 2])
max3=np.max(raw_data[:, 3])

min0=np.min(raw_data[:, 0])
min1=np.min(raw_data[:, 1])
min2=np.min(raw_data[:, 2])
min3=np.min(raw_data[:, 3])

range0=max0-min0
range1=max1-min1
range2=max2-min2
range3=max3-min3

print('<범위>')
print('sepal length : {range0}'.format(range0=range0))
print('sepal width : {range1}'.format(range1=range1))
print('petal length : {range2}'.format(range2=range2))
print('petal width : {range3}'.format(range3=range3))

std0=np.std(raw_data[:, 0])
std1=np.std(raw_data[:, 1])
std2=np.std(raw_data[:, 2])
std3=np.std(raw_data[:, 3])

print('<표준편차>')
print('sepal length : {std0}'.format(std0=std0))
print('sepal width : {std1}'.format(std1=std1))
print('petal length : {std2}'.format(std2=std2))
print('petal width : {std3}'.format(std3=std3))

skew0=skew(raw_data[:, 0],axis=0,bias=True)
skew1=skew(raw_data[:, 1],axis=0,bias=True)
skew2=skew(raw_data[:, 2],axis=0,bias=True)
skew3=skew(raw_data[:, 3],axis=0,bias=True)

print('<왜도>')
print('sepal length : {skew0}'.format(skew0=skew0))
print('sepal width : {skew1}'.format(skew1=skew1))
print('petal length : {skew2}'.format(skew2=skew2))
print('petal width : {skew3}'.format(skew3=skew3))

kurt0=kurtosis(raw_data[:, 0], axis=0)
kurt1=kurtosis(raw_data[:, 1], axis=0)
kurt2=kurtosis(raw_data[:, 2], axis=0)
kurt3=kurtosis(raw_data[:, 3], axis=0)

print('<첨도>')
print('sepal length : {kurt0}'.format(kurt0=kurt0))
print('sepal width : {kurt1}'.format(kurt1=kurt1))
print('petal length : {kurt2}'.format(kurt2=kurt2))
print('petal width : {kurt3}'.format(kurt3=kurt3))


def histo_():
    plt.style.use('ggplot')
    plt.subplot(221)
    plt.title('<sepal length histogram>')
    plt.xlabel('sepal length(cm)')
    plt.ylabel('frequency')
    plt.hist(data['sepal length'],bins=100,color='red')

    plt.style.use('ggplot')
    plt.subplot(222)
    plt.title('<sepal width histogram>')
    plt.xlabel('sepal width(cm)')
    plt.ylabel('frequency')
    plt.hist(data['sepal width'],bins=100,color='blue')

    plt.style.use('ggplot')
    plt.subplot(223)
    plt.title('<petal length histogram>')
    plt.xlabel('petal length(cm)')
    plt.ylabel('frequency')
    plt.hist(data['petal length'],bins=100,color='green')

    plt.style.use('ggplot')
    plt.subplot(224)
    plt.title('<petal width histogram>')
    plt.xlabel('petal width(cm)')
    plt.ylabel('frequency')
    plt.hist(data['petal width'],bins=100,color='purple')

    plt.tight_layout()
    plt.show()

def stem_leaf():
    stemgraphic.stem_graphic(raw_data[:, 0])
    stemgraphic.stem_graphic(raw_data[:, 1])
    stemgraphic.stem_graphic(raw_data[:, 2])
    stemgraphic.stem_graphic(raw_data[:, 3])

