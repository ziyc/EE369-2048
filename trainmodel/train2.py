import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import time
import pandas as pd
import numpy as np
import csv

batch_size = 128
NUM_EPOCHS = 20

#loading data
channels=11
def grid_ohe(input):
    output=[]
    onelayer=[]
    for counter in range(len(input)):
        oneofinput = input[counter, :]
        print('starting counter {}'.format(counter))
        for layer in range(channels):
            ret=np.zeros(shape=(4,4),dtype=int)
            for r in range(4):
                for c in range(4):
                    if layer==oneofinput[4*r+c]:
                        ret[r,c]=1
            onelayer.append(ret)
        output.append(onelayer)
        onelayer = []
    return output

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=(2, 2), padding=(1, 1)) #5*5
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(1, 4), padding=(0, 1))      #5*4
        self.conv2 = nn.Conv2d(128, 128, kernel_size=(4, 1), padding=(1, 0))     #4*4
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1))     #4*4
        self.conv5 = nn.Conv2d(128, 128, kernel_size=(4, 4), padding=(2, 2))     #5*5
        self.conv6 = nn.Conv2d(128,128,kernel_size=(2,2))                        #4*4
        self.batch_norm1 = nn.BatchNorm1d(128 * 4 * 4)
        self.fc1 = nn.Linear(128 * 4 * 4, 2048)
        self.batch_norm2 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 512)
        self.batch_norm3 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 4)
        self.initialize()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = x.view(-1, 128 * 4 * 4)
        x = self.batch_norm1(x)
        x = F.relu(self.fc1(x))
        x = self.batch_norm2(x)
        x = F.relu(self.fc2(x))
        x = self.batch_norm3(x)
        x = self.fc3(x)

        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in')

data1 = pd.read_csv('0to2048.csv')
data1 = data1.values
data1_X = data1[:,0:16]
data1_Y = data1[:,16]
data1_X = np.int64(data1_X)
data1_Y = np.int64(data1_Y)
print(len(data1_X))
print(len(data1_Y))

data2 = pd.read_csv('0to2048c.csv')
data2 = data2.values
data2_X = data2[:,0:16]
data2_Y = data2[:,16]
data2_X = np.int64(data2_X)
data2_Y = np.int64(data2_Y)
print(len(data2_X))
print(len(data2_Y))

data3 = pd.read_csv('0to2048d.csv')
data3 = data3.values
data3_X = data3[:,0:16]
data3_Y = data3[:,16]
data3_X = np.int64(data3_X)
data3_Y = np.int64(data3_Y)
print(len(data3_X))
print(len(data3_Y))

data4 = pd.read_csv('0to2048b.csv')
data4 = data4.values
data4_X = data4[:,0:16]
data4_Y = data4[:,16]
data4_X = np.int64(data4_X)
data4_Y = np.int64(data4_Y)
print(len(data4_X))
print(len(data4_Y))

X=np.concatenate((data1_X,data2_X,data3_X,data4_X),axis=0)
Y=np.concatenate((data1_Y,data2_Y,data3_Y,data4_Y),axis=0)
print(len(X))
X = grid_ohe(X)
print(len(Y))
del data1_X
del data2_X
del data3_X
del data4_X
del data1_Y
del data2_Y
del data3_Y
del data4_Y
del data1
del data2
del data3
del data4


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.01,shuffle=False)
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
Y_train = torch.LongTensor(Y_train)
Y_test = torch.LongTensor(Y_test)

del X
del Y

train_dataset = torch.utils.data.TensorDataset(X_train,Y_train)
test_dataset = torch.utils.data.TensorDataset(X_test,Y_test)

del X_train
del Y_train
del X_test
del Y_test

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True
)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False
)

del train_dataset
del test_dataset


model = Net()
#model.load_state_dict(torch.load('para673.pkl'))
model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

def train(epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        #try lower menmory
        #data = grid_ohe(data)
        #data = torch.FloatTensor(data)
        data, target = Variable(data).cuda(), Variable(target).cuda()
        #data = data.unsqueeze(dim=1)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), 'c11para0608{}.pkl'.format(epoch))
    print(epoch)

def test(epoch):
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        with torch.no_grad():
            #try lower menmory
            #data = grid_ohe(data)
            #data = torch.FloatTensor(data)
            data = Variable(data).cuda()
            target =Variable(target).cuda()
        #data = data.unsqueeze(dim=1)
        output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    print(epoch)
    test_loss /= len(test_loader.dataset)


for epoch in range(NUM_EPOCHS):
    model.train()
    train(epoch)
    model.eval()
    test(epoch)
