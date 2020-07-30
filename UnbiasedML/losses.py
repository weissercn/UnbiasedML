import numpy as np
import torch
from time import time
from utils import Metrics, find_threshold, LegendreIntegral, LegendreFitter


class WeightedMSE():
    def __init__(self,labels=None):
        """
        Mean square error loss function. Weighted such as the classifier is agnostic to label composition.

        The weight of the class with label 1 is the number of 0 labels divided by the number of 1 labels. The weight of class 0 is 1.

        Parameters
        ----------
        labels : list
            List of labels used to calculate the composition of the different classes in the dataset.
        """
        self.labeled = False
        if labels is not None:
            ones = sum(labels)
            self.ones_frac = ones/(labels.shape[0]-ones)
            self.labeled = True
    def __call__(self,pred,target,weights=None):
        if  weights is None: weights =  1
        if self.labeled:    
            weights_ = weights * (target/self.ones_frac + (1-target))
        return torch.mean(weights_*(pred-target)**2)
    def __repr__(self):
        return "Weighted MSE:  c0={:.3}   c1={:.3f}".format(1.,1/self.ones_frac)

class FlatLoss():
    def __init__(self,labels,frac,bins=32,sbins=32,memory=False,background_only=True,power=2,order=0,msefrac=1,lambd=None,max_slope=None):
        """
        Wrapper for Legendre Loss and WeighedMSE.

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
        power : int, default 2
            Power used to calculate the flat part of the loss. E.g. L2: LegendreLoss=mean((F(s)-F_flat(s))**2)
        order : int={0,1,2}, default 1
            Order up tp which the Legendre expansion is computed.
        """
        self.frac = frac
        self.msefrac = msefrac
        self.mse = WeightedMSE(labels)
        self.bins = bins
        self.sbins = sbins
        self.backonly = background_only
        self.power = power
        self.order = order
        self.memory = memory
        self.lambd = lambd
        self.max_slope = max_slope
        self.m = torch.Tensor()
        self.pred_long = torch.Tensor()
        self.fitter = LegendreFitter(order=self.order, power=self.power,lambd=self.lambd,max_slope=self.max_slope) 
    def __call__(self,pred,target,x_biased,weights=None):
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
        mse = self.mse(pred,target,weights=weights)
        if self.backonly:
            mask = target==1
            x_biased = x_biased[mask]
            pred = pred[mask]
            target = target[mask]
            if weights is not None: weights = weights[mask]
            mod = x_biased.shape[0]%self.bins
            if mod !=0:
                x_biased = x_biased[:-mod]
                pred = pred[:-mod]
                target = target[:-mod] 
                if weights is not None: weights = weights[:-mod] #Not used currently 
        if self.memory:
            self.m = torch.cat([self.m,x_biased])
            self.pred_long = torch.cat([self.pred_long,pred])
            self.pred_long = self.pred_long.detach()
            m,msorted = self.m.sort()
            pred_long = self.pred_long[msorted].view(self.bins,-1)
            self.fitter.initialize(m=m.view(self.bins,-1),overwrite=True)
            m,msorted = x_biased.sort()
            pred = pred[msorted].view(self.bins,-1)
            if weights is not None:weights = weights[msorted].view(self.bins,-1) 
            LLoss = LegendreIntegral.apply(pred, weights, self.fitter, self.sbins,pred_long)
        else:
            m,msorted = x_biased.sort()
            pred = pred[msorted].view(self.bins,-1)
            if weights is not None:weights = weights[msorted].view(self.bins,-1) 
            self.fitter.initialize(m=m.view(self.bins,-1),overwrite=True)
            LLoss = LegendreIntegral.apply(pred,weights, self.fitter, self.sbins)
        return self.msefrac*mse + self.frac* LLoss 

    def __repr__(self):
        str1 = "Flat Loss: frac={:.2f}, power={}, background_only={}, order={}, bins={}, sbins={}".format(self.frac,self.power, self.backonly,self.order,self.bins,self.sbins)
        str2 = repr(self.mse)
        return "\n".join([str1,str2])
    def reset(self):
        self.pred_long = torch.Tesnor()
        self.m = torch.Tesnor()
        return 
    
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
    def __init__(self,labels,frac,background_only=True,power=1,msefrac=1):
        self.frac = frac
        self.msefrac = msefrac
        self.mse = WeightedMSE(labels)
        self.backonly = background_only
        self.power = power
    def __call__(self,pred,target,x_biased,weights):
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
        mse = self.mse(pred,target,weights=weights)
        if self.backonly:
            mask = target==1
            x_biased = x_biased[mask]
            pred = pred[mask]
            target = target[mask]
            if weights is not None:
                weights =  weights[mask]
            else:
                weights = torch.ones_like(target)
            del mask
        disco = distance_corr(x_biased,pred,normedweight=weights,power=self.power)
        return self.frac*disco + self.msefrac *mse
    def __repr__(self):
        str1 = "DisCo Loss: frac={:.2f}, order={}".format(self.frac,self.power)
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
