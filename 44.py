import pandas as pd
import numpy as np
import torch
from torch import nn
from torch import tensor
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn import preprocessing
from sklearn.model_selection import RepeatedKFold

lst = []
#input marks
a1 = float(input("What marks your a1 marks : "))
lst.append(a1)
a2 = float(input("What marks your a2 marks : "))
lst.append(a2)
a3 = float(input("What marks your a3 marks : "))
lst.append(a3)
a4 = float(input("What marks your a4 marks : "))
lst.append(a4)
a5 = float(input("What marks your a5 marks : "))
lst.append(a5)
b1 = float(input("What marks your b1 marks : "))
lst.append(b1)
marks = torch.FloatTensor(lst)

np.random.seed(0)
df = pd.read_excel("D:\python data\Dataset - Copy.xlsx")
df = np.array(df)
X_data = df[:,0:-1]
y_data = df[:,-1]
kf = RepeatedKFold(n_splits=5, n_repeats=1, random_state=1)
for train_index, test_index in kf.split(X_data):
    X_train, X_test = X_data[train_index], X_data[test_index]
    y_train, y_test = y_data[train_index], y_data[test_index]


    x = torch.tensor(X_train, dtype=float)
    y = torch.tensor(y_train, dtype=float)

    input_size = X_data.shape[1]
    hidden1_size = 64
    hidden2_size = 16
    output_size = 1
    batch_size = 16

    my_NN = torch.nn.Sequential(
        torch.nn.Linear(input_size, hidden1_size),
        torch.nn.Sigmoid(),
        torch.nn.Linear(hidden1_size, output_size),
    )
    cost = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(my_NN.parameters(), lr=0.001,weight_decay= 1e-8)

    # train model using mini-batch
    losses = []
    y_test = tensor(y_test, dtype=float)
    for i in range(2001):
        batch_loss = []
        loss_func = F.mse_loss
        x = torch.tensor(X_test, dtype=torch.float)
        predict = my_NN(x).data.numpy()
        predict = predict.squeeze(-1)
        predict = tensor(predict, dtype=float)

        for start in range(0, len(X_train), batch_size):
            end = start + batch_size if start + batch_size < len(X_train) else len(X_train)
            xx = torch.tensor(X_train[start:end], dtype=torch.float, requires_grad=True)
            yy = torch.tensor(y_train[start:end], dtype=torch.float, requires_grad=True)
            prediction = my_NN(xx)
            prediction = prediction.squeeze(-1)
            loss = cost(prediction, yy)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            batch_loss.append(loss.data.numpy())



print(my_NN(marks).data.numpy().squeeze(-1))
torch.save()







