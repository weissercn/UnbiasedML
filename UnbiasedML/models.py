import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from time import time
from utils import Metrics, find_threshold, LegendreIntegral, LegendreFitter
from torchviz import make_dot

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
        #self.linear3 = nn.Linear(64,128)
        self.out = nn.Linear(64,1)
        # Defaults
        self.optimizer = torch.optim.SGD(self.parameters(),lr=1e-3)
        self.yhat_val = None
        self.yhat = None
        self.name = name
    def forward(self, x):
        x = nn.functional.relu(self.linear(x))
        x = nn.functional.relu(self.linear2(x))
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
            [params.pop(item) for item in ['self','scheduler','device','log',"traindataset","valdataset","optimizer",'loss',"metrics"]]
            log.initialize(params=params,loss=self.loss,optimizer=self.optimizer)

        if valdataset:
            validation_generator = DataLoader(valdataset,batch_size=len(valdataset),shuffle=False)
        training_generator = DataLoader(traindataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers,drop_last=drop_last)
        t0 = time()       
        loss = 0
        acc = 0
        print("Entering Training...")
        for epoch in range(1,epochs+1):
        #Validation and Printing
            if valdataset:
                if epoch % interval ==0 or epoch == epochs or epoch==1:
                    self.train(False)
                    for x,yval,m in validation_generator:
                        if device!='cpu':
                            x,yval,m = x.to(device),yval.to(device),m.to(device)
                        self.yhat_val = self(x).view(-1)
                    valloss = WeightedMSE(yval)
                    l_val = valloss(self.yhat_val,yval)
                    if epoch != 1:
                        metrics[0].calculate(pred=self.yhat,target=y,l=l.item())
                        acc = metrics[0].accs[-1]
                    metrics[1].calculate(pred=self.yhat_val,target=yval,l=l_val.item(),m=m)
                    R50 = metrics[1].R50[-1]
                    JSD = metrics[1].JSD[-1]
                    acc_val = metrics[1].accs[-1]
                    entry = 'Epoch:{:04d}/{:04d}  ({t:<5.1f}s)\n Train: loss:{:.4f}, acc:{:.0f}% || Val: loss: {:.4f}, acc:{:.0f}%, R50: {:.4f}, 1/JSD: {:.4f}'.format(
                epoch,epochs,loss, 100.* acc,
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
           # Feed forward 
            for x,y,m in training_generator:
                if device!='cpu':
                    x,y,m = x.to(device),y.to(device),m.to(device)
                self.train()
                self.yhat = self(x).view(-1)
                if epoch<delay_loss:
                    l = torch.nn.MSELoss()(self.yhat,y)
                elif pass_x_biased==False:
                    l = self.loss(pred=self.yhat,target=y)
                else:
                    l = self.loss(pred=self.yhat,target=y,x_biased=m)
                l.backward()
                loss = l.item()
                self.optimizer.step()
                self.optimizer.zero_grad()
                if scheduler:
                    scheduler.step()
        if log is not None:
            log.finished()


class WeightedMSE():
    def __init__(self,labels):
        """
        Mean square error loss function. Weighted such as the classifier is agnostic to label composition.

        The weight of the class with label 1 is the number of 0 labels divided by the number of 1 labels. The weight of class 0 is 1.

        Parameters
        ----------
        labels : list
            List of labels used to calculate the composition of the different classes in the dataset.
        """
        ones = sum(labels)
        self.ones_frac = ones/(labels.shape[0]-ones)
    def __call__(self,pred,target):
        weights = target/self.ones_frac + (1-target)
        return torch.mean(weights*(pred-target)**2)
    def __repr__(self):
        return "Weighted MSE:  c0={:.3}   c1={:.3f}".format(1.,1/self.ones_frac)

class FlatLoss():
    def __init__(self,labels,frac,bins=32,sbins=32,recalculate=True,background_only=True,norm='L2',order=1):
        """
        Wrapper for Legendre Loss and WeightedMSE.

        The total flat loss = frac*LegendreLoss + (1-frac)*WeightedMSE 

        Parameters
        ----------
        labels : list
            List of labels passed to WeightedMSE to calculate the proportion of class labels.
        frac : float=[0,1]
            Between 0 and 1. Used to determine the relative strength of the flat part of the loss the MSE. 
            Loss = frac*(LegendreLoss) + (1-frac)WeightedMSE 
            frac = strength/(1+strength)
        bins : int
            Number of bins in the biased feature to integrate over.
        sbins : int
            Number of bins of scores values.
        recalculate : bool, default True
            If True, integrate over biased feature locally i.e. on a per batch basis. Otherwise only calculate the biased feature Legendre polynomials once and integrate over it globally.
        background_only : bool, default True
            If True, only try to flatten the response of background events (label 1.) Otherwise, flatten the response for both classes at the same time.
        norm : string={'L1','L2"}, default 'L2'
            Normalization used to calculate the flat part of the loss. E.g. L2: LegendreLoss=mean((F(s)-F_flat(s))**2)
        order : int={0,1,2}, default 1
            Order up tp which the Legendre expansion is computed.
        """
        self.frac = frac
        self.mse = WeightedMSE(labels)
        self.bins = bins
        self.sbins = sbins
        self.backonly = background_only
        self.norm = norm
        self.recalculate = recalculate
        self.order = order
    def __call__(self,pred,target,x_biased):
        """
        Calculate the total loss (flat and MSE.)


        Parameters
        ----------
        pred : Tensor
            Tensor of predictions.
        target : Tensor
            Tensor of target labels.
        x_biased : Tensor
            Tensor of biased feature.
        """
        mse = (1-self.frac)*self.mse(pred,target)
        if not self.recalculate: #broken for now
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
           # LLoss = LegendreLoss(x_biased,bins=self.bins,sbins=self.sbins,order=self.order,norm=self.norm)
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
            mbins = self.bins
            m,msorted = x_biased.sort()
            pred = pred[msorted].view(mbins,-1)
            fitter = LegendreFitter(m=m.view(mbins,-1), power=2) 
            LLoss = LegendreIntegral.apply(pred, fitter, self.sbins)
            return self.frac*LLoss + mse
    def __repr__(self):
        str1 = "Flat Loss: frac/strength={:.2f}/{:.2f}, norm={}, background_only={}, order={}, bins={}, sbins={}".format(self.frac,self.frac/(1-self.frac),self.norm, self.backonly,self.order,self.bins,self.sbins)
        str2 = repr(self.mse)
        return "\n".join([str1,str2])

class LegendreLoss():
    def __init__(self,x_biased,bins=32,norm="L2",sbins=100,order=1):
        """
        Calculate the n-th order Legendre expansions of the CDF of the predictions as a function of the biased feature and tries to minimize the difference between the Legendre expansion and the CDF of the predicted scores across bins of the biased feature.

        Bins x_biased into number_bins = bins. Calculate cumsum of the scores in every bin and integrates the ith cumsum across bins of mass with legendre polynomials to find the coefficients of the expansion.
        Then calculates the norm of the difference between the cumsum and its legendre expansion.

        Parameters
        ----------
        x_biased : Tensor or Array or List
            Vector of the biased feature.
        bins : int, default 32
            Number of bins in biased feature to integrate over.
        norm : string={'L1','L2'}, default 'L2'
            Norm used to calculate the difference between cumsum and its legendre exapnsion.
        sbins : int, default 100
            Number of score (ypred) bins to use.
        order : int, default 1
            The maximum order of legendre polynomial to compute. 
        """
        self.mass, self.ordered_mass = torch.sort(x_biased)
        self.dm = (self.mass.view(bins,-1)[:,-1] - self.mass.view(bins,-1)[:,0]).view(-1)
        self.m = self.mass.view(bins,-1).mean(axis=1).view(-1)
        self.p0 = 1
        self.p1 = self.m
        self.p2 = (self.m**2-1)/2
        self.bins = bins
        self.sbins = sbins
        self.norm = norm
        self.order = order
    def __call__(self,pred,target):   
        pred_bins = pred[self.ordered_mass].view(self.bins,-1)
        self.s = torch.linspace(pred.min().item(),pred.max().item(),self.sbins).view(-1,1,1)
        self.s.requires_grad_(True)
        self.F = (self.s>pred_bins.sort(axis=1)[0]).sum(axis=2).float()/self.m.shape[0] 
        a0 = 1/2 * (self.F*self.dm).sum(axis=1).view(-1,1)
        self.legendre = a0
        if self.order>0:
            a1 = 3/2 * (self.F*self.p1*self.dm).sum(axis=1).view(-1,1)
            self.legendre = a0 +  a1*self.p1
        if self.order>1:
            a2 = 5/2 * (self.F*self.p2*self.dm).sum(axis=1).view(-1,1)
            self.legendre += a2*self.p2
        if self.norm == "L2":
            legendre_loss = ((self.F - self.legendre)**2).mean()
        elif self.norm == "L1":
            legendre_loss = torch.abs(self.F - self.legendre).mean()
        else:
            raise ValueError("{} is not a valid norm. Choose L1 or L2.hex".format(self.norm))
        breakpoint()
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

class Disco():
    def __init__(self,labels,frac,background_only=True,order=1):
        self.frac = frac
        self.mse = WeightedMSE(labels)
        self.backonly = background_only
        self.order = order
    def __call__(self,pred,target,x_biased):
        """
        Calculate the total loss (flat and MSE.)


        Parameters
        ----------
        pred : Tensor
            Tensor of predictions.
        target : Tensor
            Tensor of target labels.
        x_biased : Tensor
            Tensor of biased feature.
        """
        mse = (1-self.frac)*self.mse(pred,target)
        if self.backonly:
            mask = target==1
            x_biased = x_biased[mask]
            pred = pred[mask]
            target = target[mask]
        #ones_frac = self.mse.ones_frac
        #ones = (target==1).sum()
        #weights = (target/ones_frac + (1-target))*target.shape[0]/2/(target.shape[0]-ones)
        weights = torch.ones_like(target)
        disco = distance_corr(x_biased,pred,weights,power=self.order)
        return self.frac*disco + mse
    def __repr__(self):
        str1 = "DisCo Loss: frac/strength={:.2f}/{:.2f}, order={}".format(self.frac,self.frac/(1-self.frac),self.order)
        str2 = repr(self.mse)
        return "\n".join([str1,str2])


def distance_corr(var_1,var_2,normedweight,power=1):
    """var_1: First variable to decorrelate (eg mass)
    var_2: Second variable to decorrelate (eg classifier output)
    normedweight: Per-example weight. Sum of weights should add up to N (where N is the number of examples)
    power: Exponent used in calculating the distance correlation

    va1_1, var_2 and normedweight should all be 1D torch tensors with the same number of entries

    Usage: Add to your loss function. total_loss = BCE_loss + lambda * distance_corr
    """


    xx = var_1.view(-1, 1).repeat(1, len(var_1)).view(len(var_1),len(var_1))
    yy = var_1.repeat(len(var_1),1).view(len(var_1),len(var_1))
    amat = (xx-yy).abs()

    xx = var_2.view(-1, 1).repeat(1, len(var_2)).view(len(var_2),len(var_2))
    yy = var_2.repeat(len(var_2),1).view(len(var_2),len(var_2))
    bmat = (xx-yy).abs()

    amatavg = torch.mean(amat*normedweight,dim=1)
    Amat=amat-amatavg.repeat(len(var_1),1).view(len(var_1),len(var_1))\
        -amatavg.view(-1, 1).repeat(1, len(var_1)).view(len(var_1),len(var_1))\
        +torch.mean(amatavg*normedweight)

    bmatavg = torch.mean(bmat*normedweight,dim=1)
    Bmat=bmat-bmatavg.repeat(len(var_2),1).view(len(var_2),len(var_2))\
        -bmatavg.view(-1, 1).repeat(1, len(var_2)).view(len(var_2),len(var_2))\
        +torch.mean(bmatavg*normedweight)

    ABavg = torch.mean(Amat*Bmat*normedweight,dim=1)
    AAavg = torch.mean(Amat*Amat*normedweight,dim=1)
    BBavg = torch.mean(Bmat*Bmat*normedweight,dim=1)

    if(power==1):
        dCorr=(torch.mean(ABavg*normedweight))/torch.sqrt((torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight)))
    elif(power==2):
        dCorr=(torch.mean(ABavg*normedweight))**2/(torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight))
    else:
        dCorr=((torch.mean(ABavg*normedweight))/torch.sqrt((torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight))))**power

    return dCorr
