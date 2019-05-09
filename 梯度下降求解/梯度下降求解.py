import numpy as np
import pandas as pd

#定义一个简单的数据，使得数据符合y=x1+x2
df = pd.DataFrame({'x1':[1,2,3,4,5,6],'x2':[1,2,1,2,1,2],'y':[2,4,4,6,6,8]})
x = df[['x1','x2']]
y = df['y'].values.reshape([-1,1])
x

#批量梯度下降------初始化theta，给定一个alpha
theta = np.array([[0],[0]])
alpha = 0.01

#批量梯度下降------迭代
theta = theta - alpha * x.T.dot(x.dot(theta)-y)
loss = np.power(x.dot(theta)-y,2).sum()
print('loss值为%f \ntheta值为\n'%loss,theta.values)