import numpy as np
import pandas as pd

#使用一个分类的数据
import pandas as pd
df = pd.read_csv('./datas/iris.data')
df['Iris-setosa'] = pd.Categorical(df['Iris-setosa']).codes
df = df[df['Iris-setosa'] != 2]
x = df.drop(['Iris-setosa'],1)
y = df['Iris-setosa']

#使用梯度下降法求解

#概率转换成01的类数据
def prob2class(y_prob):
    y_class = [1 if i>=0.5  else 0 for i in y_prob]
    return y_class

#sigmoid函数
def sigmoid(theta,x):
    prob = 1/(1+np.exp(-x.dot(theta)))
    return prob.values

#损失函数
def lr_loss(y_true,prob):
    laplace = 1e-10
    lr_loss = - sum(y_true * np.log(prob).ravel() + (1-y_true) * np.log(1-prob+laplace).ravel()) + 1/len(y)*np.power(theta,2).sum()
    return lr_loss
#初始化aplha 和 theta
alpha =0.02
theta = np.zeros((4,1))
theta

#梯度下降法开始迭代
y_true = y.reshape(-1,1)
lr_prob = sigmoid(theta,x)
#print(lr_prob)
lr_class = prob2class(lr_prob)
print(lr_class)
theta = theta + alpha * x.T.dot(y_true - sigmoid(theta,x))
lr_loss(y,lr_prob)
