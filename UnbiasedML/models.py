import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from time import time
from utils import Metrics, find_threshold, LegendreIntegral, LegendreFitter
from losses import WeightedMSE


class Classifier(nn.Module):
    def __init__(self,input_size=10,name=None):
        """
         DNN Model inherits from torch.nn.Module. Can be initialized with input_size: Number of features per sample.

        This is a class wrapper for a simple DNN model. Creates an instance of torch.nn.Module that has 4 linear layers. Use torchsummary for details.abs 

        Parameters
        ----------
        input_size : int=10
            The number of features to train on.
        name : string=None 
            Specifiy a name for the DNN.break
        """
        super().__init__()
        self.linear = nn.Linear(input_size,64)
        self.linear1 = nn.Linear(64,64)
        #self.linear2 = nn.Linear(32,64)
        self.batchnorm = nn.BatchNorm1d(64)
        #self.linear3 = nn.Linear(64,128)
        self.out = nn.Linear(64,1)
        # Defaults
        self.optimizer = torch.optim.SGD(self.parameters(),lr=1e-3)
        self.yhat_val = None
        self.yhat = None
        self.name = name
    def forward(self, x):
        x = nn.functional.relu(self.linear(x))
        x = self.batchnorm(x)
        x = nn.functional.relu(self.linear1(x))
        x = nn.functional.relu(self.linear1(x))
        #x = nn.functional.relu(self.linear2(x))
        #x = nn.functional.relu(self.linear3(x))
        x = torch.sigmoid(self.out(x))
        return x

    def fit(self,traindataset,epochs=200,batch_size=None, shuffle=False, num_workers=None, optimizer=None,scheduler=None,loss=None,interval=100,valdataset=None,drop_last=False,metrics=None,delay_loss=False,pass_x_biased=False,device='cpu',log=None):
        """
        Fit model to traindataset. 

        Parameters
        ----------
        traindataset : DataSet
            The DataSet [torch.utils.data.Dataset] instance containing the training data. Used to create a DataLoader. Must return x,y,m where x is the vector of features, y is the label and m is the biased feature.
        epochs : int
            Number of epochs to train.
        batch_size : int
            Size of the batch.
        shuffle : bool
            If True shuffles the training data.
        num_workers : int
            Passed to DataLoader. 
        optimizer : torch.optim
            Optimizer to use in the training. Defaults to torch.optim.SGD(lr-1e-3).
        scheduler : torch.optim.lr_scheduler
            Scheduler used to change the learning rate.
        loss : Callable
            Criterion to minimize. Defaults to torch.nn.MSELoss
        interval : int
            Log and print progress every epochs mod interval == 0.
        valdataset : DataSet
            Same as traindataset but for the validation data.
        drop_last : bool
            If drop_last the DataLoader will only keep floor(len(traindataset)/batch_size). Used if the loss requires batches of the same size.
        metrics : [Metrics,Metrics]
            Metrics object where to store the training/validation metrics.
        delay_loss : int
            Delay using the provided loss for delay_loss epochs. Optmizer uses WeightedMSE before that. 
        pass_x_biased : bool
            If true, passes the biased feature as a third argument to the loss function.
        device : str or torch.device
            Which device to use. Defaults to cpu.
        log : Logger
            Logging object.     
        """
        if optimizer:
            self.optimizer = optimizer
        if loss:
            self.loss = loss
        if metrics is None:
            metrics = [Metrics(),Metrics(validation=True)]
        if log:
            params = locals()
            [params.pop(item) for item in ['self','scheduler','log',"traindataset","valdataset","optimizer",'loss',"metrics"]]
            log.initialize(model=self,params=params,loss=self.loss,optimizer=self.optimizer,scheduler=scheduler)

        if valdataset:
            validation_generator = DataLoader(valdataset,batch_size=len(valdataset),shuffle=False)
        training_generator = DataLoader(traindataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers,drop_last=drop_last)
        t0 = time()       
        loss = 0
        acc = 0
        print("Entering Training...")
        for epoch in range(1,epochs+1):
           # Feed forward 
            #self.loss.m = torch.Tensor().to(device)
            #self.loss.pred_long = torch.Tensor().to(device)
            for item in training_generator:
                x,y,m,weights = item
                if device!='cpu':
                    x,y,m,weights = x.to(device),y.to(device),m.to(device), weights.to(device)
                self.train()
                self.yhat = self(x).view(-1)
                if epoch<delay_loss:
                    l = torch.nn.MSELoss()(self.yhat,y)
                elif pass_x_biased==False:
                    l = self.loss(pred=self.yhat,target=y)
                else:
                    l = self.loss(pred=self.yhat,target=y,x_biased=m,weights=weights) 
                l.backward()
                loss = l.item()
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            #Validation and Printing
            if valdataset:
                if epoch % interval ==0 or epoch == epochs or epoch==1:
                    self.train(False)
                    for x,yval,m,_ in  validation_generator:
                        break
                    if device!='cpu':
                        x,yval,m = x.to(device),yval.to(device),m.to(device)
                    self.yhat_val = self(x).view(-1)
                    valloss = WeightedMSE(yval)
                    l_val = valloss(self.yhat_val,yval)
                    metrics[0].calculate(pred=self.yhat.data,target=y,l=l.item())
                    metrics[1].calculate(pred=self.yhat_val.data,target=yval,l=l_val.item(),m=m)
                    acc = metrics[0].accs[-1]
                    R50 = metrics[1].R50[-1]
                    JSD = metrics[1].JSD[-1]
                    acc_val = metrics[1].accs[-1]
                    entry = 'Epoch:{:04d}/{:04d}  ({t:<5.1f}s)\n Train: loss:{:.4f}, acc:{:.1f}% || Val: loss: {:.4f}, acc:{:.1f}%, R50: {:.4f}, 1/JSD: {:.4f}'.format(
                epoch,epochs,loss, 100.* acc,
                l_val.item(), 100.* acc_val,R50,1/JSD,t=time()-t0)
                    print(entry)
                    if log is not None:
                        log.entry(entry)
                    if scheduler:
                        scheduler.step(l_val.item())
                    del x, yval, m, valloss, l_val, 
            else:
                if epoch % interval ==0:
                    metrics[0].calculate(pred=self.yhat.data,target=y,l=l.item())
                    acc = metrics[0].accs[-1]
                    entry = 'Epoch:{:04d}/{:04d} loss: {:.4f}, accuracy:({:.0f}%)'.format(
                        epoch,epochs,l.item(), 100.* acc)
                    print(entry)
                    if log is not None:
                        log.entry(entry)
        if log is not None:
            log.finished()

