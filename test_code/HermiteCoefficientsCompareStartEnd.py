# coding=utf-8
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch import abs, exp
import numpy.polynomial.hermite as hm
from math import pi
import numpy as np
from pdb import set_trace
import os,sys
from ConvModels import *
DataPath='/disks/disk2/hjx/Hermite/data/'
sys.path.append(DataPath)
from CifarDataLoader import data_loading

#choose net mode
def train_model(net1, coef1,learning_rate,num_epochs,batch_size,train_loader,channels,hidden_size):
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
            outputs1= net1(images,coef1)
            loss1 = criterion(outputs1, labels)  #
           # loss1.register_hook(print)
            loss1.backward() 
            optimizer1.step() 
    #         print(epoch)
            k=k+1
            px.append(k)
            p1.append(loss1.item())
           
            if (i + 1) % batch_size == 0:
                print(' Epoch [%d/%d],  Loss: %.4f'
                      % (epoch + 1, num_epochs,  loss1.item()))
    return p1


def test_model():
    correct1=0
    total1=0
    for images, labels in test_loader: 
        images = Variable(images)
        outputs1 = net1(images,coef1)
        _, predicted1 = torch.max(outputs1.data, 1)  
        total1 += labels.size(0) 
        correct1 += (predicted1 == labels).sum()  
    #     print('Accuracy of the net on the 10000 test images: %d %%' % (100 * correct1 / total1))

    
if __name__== "__main__":
    datasets='MNIST'
    PathSave='./Results/'
  # Hyper Parameters   
    torch.manual_seed(1) 
    if datasets=='MNIST':
        input_size = 28*28  
        hidden_size = 256
        batch_size = 250 
        num_classes = 10  
        num_epochs = 1
        learning_rate = 0.01  
        channels=1


    if datasets=='CIFAR10':
        input_size = 28*28 
        hidden_size = 400 
        batch_size = 250  
        num_classes = 10  
        num_epochs = 1
        channels=3
        learning_rate = 0.01  


   
    [train_loader,test_loader]=data_loading(DataPath,datasets,batch_size)
    net1 = ConvNet1(channels,hidden_size)
#     values=[0.3 ,  0.4 , 0.5 ,  0.6 ,  0.7]

#     values =np.arange(0.9,0.5,-0.1)
#     values=[0.06,0.08,0.1,0.12,0.14]
#     start=0.65
    #values =[ 0.5, 0.4, 0.3,0.2,0.1]
    #start=0.8
#     values =[0.3, 0.35, 0.4, 0.45,0.5,0.55]
        
    Error= []
    #for p in range(len(values)):

    start = [0.95,0.9,0.85,0.8,0.75,0.7,0.65]
    values = [0.69,0.59,0.49, 0.39, 0.29, 0.19, 0.09]
    coef_log = []
    for x in start:
        for p in range(len(values)):
            coef1= np.arange(x,values[p],-0.1)
            if(len(coef1) < 3 or len(coef1) > 6):
                continue
            print(coef1)
            coef_log.append(coef1)
            loss = 0
            c1 = 0
            t1 = 0
            times = 5
            for y in range(times):
                #Error[p].append(train_model(net1,coef1,learning_rate,num_epochs,batch_size,
                #                            train_loader,channels,hidden_size))
                loss += np.array(train_model(net1,coef1,learning_rate,num_epochs,batch_size,
                                            train_loader,channels,hidden_size))          
                correct1=0
                total1=0
                for images, labels in test_loader:  
                    images = Variable(images)
                    outputs1 = net1(images,coef1)
                    _, predicted1 = torch.max(outputs1.data, 1)  
                    total1 += labels.size(0) 
                    correct1 += (predicted1 == labels).sum()  
                print('Accuracy of the net on the 10000 test images:',(correct1.numpy() / total1))
                c1 += correct1.numpy()
                t1 += total1
            print('Average accuracy:',c1/t1)
            Error.append(loss/times)
    Parameters=[num_epochs,batch_size,num_classes]
    np.save(PathSave+datasets+'ErrorConvergenceHermiteCoefficientCompareEnd.npy',np.array([Parameters,coef_log,Error]))
    