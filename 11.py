import pandas as pd
import numpy as np
import torch
from torch import nn
from torch import tensor
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from sklearn import preprocessing


df = pd.read_excel("D:\python data\Dataset - Copy.xlsx")
np.random.seed(3)
df = np.random.permutation(np.array(df))
X_data = df[:,0:-1]
y_data = df[:,-1]

#process data
test_radio = 0.2 #set test radio to be 0.2
test_set_size = int(len(df)*test_radio)


#normalize data
X_data = preprocessing.StandardScaler().fit_transform(X_data)
X_train = X_data[test_set_size:,:]
X_test = X_data[:test_set_size,:]
y_train = y_data[test_set_size:]
y_test = y_data[:test_set_size]
assert (len(X_train) == len(y_train))

#build 1 hidden layer NN
x = torch.tensor(X_train,dtype = float)
y = torch.tensor(y_train,dtype = float)

input_size = X_data.shape[1]
hidden_size = 4
output_size = 1
batch_size = 16

my_NN = torch.nn.Sequential(
    torch.nn.Linear(input_size,hidden_size),
    torch.nn.Sigmoid(),
    torch.nn.Linear(hidden_size,output_size),
)
cost = torch.nn.MSELoss(reduction = 'mean')
optimizer = torch.optim.Adam(my_NN.parameters(),lr = 0.001)

#train model using mini-batch
losses = []
for i in range (1000):
    batch_loss = []
    for start in range(0,len(X_train),batch_size):
        end = start + batch_size if start + batch_size < len(X_train) else len(X_train)
        xx = torch.tensor(X_train[start:end], dtype = torch.float, requires_grad = True)
        yy = torch.tensor(y_train[start:end], dtype = torch.float, requires_grad = True)
        prediction = my_NN(xx)
        prediction = prediction.squeeze(-1)
        loss = cost(prediction,yy)
        optimizer.zero_grad()
        loss.backward(retain_graph = True)
        optimizer.step()
        batch_loss.append(loss.data.numpy())

    #print loss
    if i % 100 == 0:
        losses.append(np.mean(batch_loss))
        print(i, np.mean(batch_loss))


x = torch.tensor(X_test, dtype = torch.float)
predict = my_NN(x).data.numpy()
print(np.array(predict))
print('=======================')
print(y_test)


