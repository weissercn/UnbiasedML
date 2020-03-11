import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from time import time
from utils import Metrics, find_threshold


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
        self.linear = nn.Linear(input_size,32)
        self.linear2 = nn.Linear(32,64)
        self.linear3 = nn.Linear(64,128)
        self.out = nn.Linear(128,1)
        # Defaults
        self.optimizer = torch.optim.SGD(self.parameters(),lr=1e-3)
        self.yhat_val = None
        self.yhat = None
        self.name = name
    def forward(self, x):
        x = nn.functional.relu(self.linear(x))
        x = nn.functional.relu(self.linear2(x))
        x = nn.functional.relu(self.linear3(x))
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
            [params.pop(item) for item in ['self','scheduler','device','log',"traindataset","valdataset","optimizer",'loss',"metrics"]]
            log.initialize(params=params,loss=self.loss,optimizer=self.optimizer)

        if valdataset:
            validation_generator = DataLoader(valdataset,batch_size=len(valdataset),shuffle=False)
        training_generator = DataLoader(traindataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers,drop_last=drop_last)
        t0 = time()        
        print("Entering Training...")
        for epoch in range(1,epochs+1):
            for x,y,m in training_generator:
                if device!='cpu':
                    x,y,m = x.to(device),y.to(device),m.to(device)
                self.train()
                self.yhat = self(x).view(-1)
                if epoch<delay_loss:
                    l = torch.nn.MSELoss()(self.yhat,y)
                elif pass_x_biased==False:
                    l = self.loss(self.yhat,y)
                else:
                    l = self.loss(self.yhat,y,m)
                l.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                if scheduler:
                    scheduler.step()


        #Validation and Printing
            if valdataset:
                if epoch % interval ==0 or epoch == epochs:
                    self.train(False)
                    for x,yval,m in validation_generator:
                        self.yhat_val = self(x).view(-1)
                    valloss = WeightedMSE(yval)
                    l_val = valloss(self.yhat_val,yval)
                    metrics[0].calculate(pred=self.yhat,target=y,l=l.item())
                    metrics[1].calculate(pred=self.yhat_val,target=yval,l=l_val.item(),m=m)
                    R50 = metrics[1].R50[-1]
                    JSD = metrics[1].JSD[-1]
                    acc_val = metrics[1].accs[-1]
                    acc = metrics[0].accs[-1]
                    entry = 'Epoch:{:04d}/{:04d}  ({t:<5.1f}s)\n Train: loss:{:.4f}, acc:{:.0f}% || Val: loss: {:.4f}, acc:{:.0f}%, R50: {:.4f}, 1/JSD: {:.4f}'.format(
                epoch,epochs,l.item(), 100.* acc,
                l_val.item(), 100.* acc_val,R50,1/JSD,t=time()-t0)
                    print(entry)
                    if log is not None:
                        log.entry(entry)
            else:
                if epoch % interval ==0:
                    acc = metrics[0].accs[-1]
                    entry = 'Epoch:{:04d}/{:04d} loss: {:.4f}, accuracy:({:.0f}%)'.format(
                        epoch,epochs,l.item(), 100.* acc)
                    print(entry)
                    if log is not None:
                        log.entry(entry)
        if log is not None:
            log.finished()


class WeightedMSE():
    def __init__(self,labels):
        ones = sum(labels)
        self.ones_frac = ones/(labels.shape[0]-ones)
    def __call__(self,pred,target):
        weights = target/self.ones_frac + (1-target)
        return torch.mean(weights*(pred-target)**2)
    def __repr__(self):
        return "Weighted MSE:  c0={:.1f}   c1={:.3f}".format(1.,1/self.ones_frac)

class FlatLoss():
    def __init__(self,labels,frac,bins=32,recalculate=True,background_only=True,norm='L2'):
        self.frac = frac
        self.mse = WeightedMSE(labels)
        self.backonly = background_only
        self.bins = bins
        self.norm = norm
        self.recalculate = recalculate
    def __call__(self,pred,target,x_biased):
        mse = (1-self.frac)*self.mse(pred,target)
        if not self.recalculate:
            if self.backonly:
                truthmask = target==1
                mod = truthmask.sum()%self.bins
                if mod !=0:
                    pred = pred[truthmask][:-mod]
                    target = target[truthmask][:-mod]
                    x_biased = x_biased[truthmask][:-mod]
                else:
                    pred = pred[truthmask]
                    target = target[truthmask]
                    x_biased = x_biased[truthmask]
            LLoss = LegendreLoss(x_biased,self.bins,norm=self.norm)
            return self.frac*LLoss(pred,target) + mse 
        else:
            if self.backonly:
                mask = target==1
                x_biased = x_biased[mask]
                pred = pred[mask]
                target = target[mask]
                mod = x_biased.shape[0]%self.bins
                if mod !=0:
                    x_biased = x_biased[:-mod]
                    pred = pred[:-mod]
                    target = target[:-mod] 
            LLoss = LegendreLoss(x_biased,self.bins,norm=self.norm)
            return self.frac*LLoss(pred,target) + mse
    def __repr__(self):
        str1 = "Flat Loss: frac/strength={:.2f}/{:.2f}, norm={}, background_only={}, bins={}".format(self.frac,self.frac*(1-self.frac),self.norm, self.backonly,self.bins)
        str2 = repr(self.mse)
        return "\n".join([str1,str2])

class LegendreLoss():
    def __init__(self,x_biased,bins=32,norm="L2"):
        self.mass, self.ordered_mass = torch.sort(x_biased)
        self.dm = (self.mass.view(-1,bins)[:,-1] - self.mass.view(-1,bins)[:,0]).view(-1,1)
        self.m = self.mass.view(-1,bins).mean(axis=1).view(-1,1)
        self.p0 = 1
        self.p1 = self.m
        self.p2 = (self.m**2-1)/2
        self.scores = 0
        self.legendre = 0
        self.bins = bins
        self.norm = norm
    def __call__(self,pred,target):   
        pred_bins = pred[self.ordered_mass].view(-1,self.bins)
        ordered_s = pred_bins.argsort(axis=1)
        self.scores = pred_bins 
        self.scores = torch.cumsum(self.scores,axis=1)/self.scores.sum(axis=1).view(-1,1)
        a0 = 1/2 * (self.scores*self.dm).sum(axis=0)
        a1 = 3/2 * (self.scores*self.p1*self.dm).sum(axis=0)
        self.legendre = a0 + a1*self.p1
        if self.norm == "L2":
            legendre_loss = ((self.scores - self.legendre)**2).mean()
        elif self.norm == "L1":
            legendre_loss = torch.abs(self.scores - self.legendre).mean()
        else:
            raise ValueError("{} is not a valid norm. Choose L1 or L2.hex".format(self.norm))
        return legendre_loss

    
class JiangLoss():  #need to remove numpy later
    def __init__(self,truth,x_biased,eta=1e-3,range=(0.25,0.75)):
        self.gx = (x_biased<rang[1])&(x_biased>range[0])
        self.ytrue = (truth==1)
        self.Z_g = self.gx.sum()/x_biased.size
        self.P_g = (self.ytrue&self.gx).sum()/x_biased.size
        self.P_x = (self.ytrue).sum()/x_biased.size
        self.cx = (self.gx/self.Z_g -1)
        self.lambda1 = 0
        self.weights = np.ones_like(y_train)
        self.eta = eta 
        self.scores = torch.from_numpy(np.random.randint(0,2,size=x_biased.size))
    def __call__(self,pred,target):
        self.weights = np.array(self.weights.tolist()).flatten()
        self.delta =  (np.array(self.scores.tolist()).flatten()*self.cx).mean()      
        self.lambda1 -= self.eta*self.delta
        weights_ = np.exp(self.lambda1*self.cx)
        self.weights[y_train==1] = (weights_/(1+weights_))[y_train==1]
        self.weights[y_train==0] = (1/(1+weights_))[y_train==0]
        self.weights = torch.from_numpy(self.weights).view(-1,1)
        self.scores = pred
        return torch.mean(self.weights*(pred-target)**2)

