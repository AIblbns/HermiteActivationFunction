import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
from numpy.polynomial.hermite import Hermite, hermval
from torch import abs, exp
import numpy.polynomial.hermite as hm
from math import pi
import numpy as np
from sklearn.decomposition import PCA
from ConvModels import *
#from SaveDataCsv import SaveDataCsv
import os,sys
DataPath='/disks/disk2/hjx/Hermite/data/'
sys.path.append(DataPath)
from CifarDataLoader import data_loading


def train_model(net1,coef1,layers,learning_rate,num_epochs,batch_size,train_loader):
    # Loss and Optimizer  
    criterion = nn.CrossEntropyLoss()
    optimizer1 = torch.optim.Adam(net1.parameters(), lr=learning_rate)
     # Train the Model  
    k=0
    px, p1 = [], [] 
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader): 
            images = Variable(images)
            # Forward + Backward + Optimize
            optimizer1.zero_grad()  # zero the gradient buffer  
            outputs1= net1(images)
            loss1 = criterion(outputs1, labels)  
           # loss1.register_hook(print)
            loss1.backward() 
            optimizer1.step() 
           # print(epoch)
            k=k+1
            px.append(k)
            p1.append(loss1.item())
            if (i + 1) % batch_size == 0:
                print(' Epoch [%d/%d],  Loss: %.4f'
                      % (epoch + 1, num_epochs,  loss1.item()))
    return p1


def test_model(layers):
    correct1=0
    total1=0
    for images, labels in test_loader:
        images = Variable(images)
        outputs1 = net(images,coef1)
        _, predicted1 = torch.max(outputs1.data, 1)  
        total1 += labels.size(0) 
        correct1 += (predicted1 == labels).sum()  
    print('Accuracy of the net on the 10000 test images: %d %%' % (100 * correct1 / total1))
    
    
if __name__== "__main__":
    # Hyper Parameters   
    dataset='MNIST'
    PathSave='./Results/'
    
    if dataset=='MNIST':
        input_size = 28*28  
        hidden_size = 256 
        batch_size = 250  
        num_classes = 10  
        num_epochs = 1 
        learning_rate = 0.01  
        channels=1
        MonteSize=5
        layers=3

    if dataset=='CIFAR10':
        input_size = 32*32  
        hidden_size = 400 
        batch_size = 250  
        num_classes = 10  
        num_epochs = 1 
        channels=3
        learning_rate = 0.01  
        MonteSize=10
        layers=3
        
    train_loader, test_loader = data_loading(DataPath,dataset,batch_size)
        
    Error= []
    values=['Sigmoid','Tanh','ReLU','LReLU','ELU','SELU','Swish']
    coef1= np.arange(0.7,0.3,-0.1)

    for p in range(len(values)):
        print( values[p])
        if values[p]=="Sigmoid":
            net=sigmoidNet(channels,hidden_size)
        elif values[p]=="Tanh":
            net=tanhNet(channels,hidden_size)                      
        elif values[p]=="ReLU":
            net=ReluNet(channels,hidden_size)
        elif values[p]=="ELU":
            net=ELUNet(channels,hidden_size)
        elif values[p]=="LReLU":
            net=LeakyReluNet(channels,hidden_size)
        elif values[p]=="SELU":
            net=SeluNet(channels,hidden_size)
        elif values[p]=="Swish":
            net=SwishNN(channels,hidden_size)
        elif values[p]=="Mish":
            net=MishNet(channels,hidden_size)
        else:
            raise Exception ("Input wrong activation function")
            
        ErrorTime=[]
        c1 = 0
        t1 = 0
        for k in range(MonteSize):
            ErrorTime.append(train_model(net,coef1,layers,learning_rate,num_epochs,batch_size,train_loader))
            correct1=0
            total1=0
            for images, labels in test_loader:
                images = Variable(images)
                outputs1 = net(images)
                _, predicted1 = torch.max(outputs1.data, 1)  
                total1 += labels.size(0) 
                correct1 += (predicted1 == labels).sum()
            #print(type(total1),type(correct1))
            print('Accuracy of the net on the 10000 test images:',(correct1.numpy() / total1))
            c1 += correct1.numpy()
            t1 += total1
        print('Average accuracy:',c1/t1)
        ErrorAvg=[sum(x)/len(ErrorTime) for x in zip(*ErrorTime)]
        Error.append([ErrorAvg])
        
    Parameters=[num_epochs,batch_size,num_classes]
    np.save('./Results/ActivationFunctionCompare_{}.npy'.format(dataset),np.array([Parameters,values,Error]))
    



