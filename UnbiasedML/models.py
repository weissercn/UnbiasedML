import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from time import time

class Classifier(nn.Module):
    """ DNN Model Class. Can be initialized with input_size: Number of features per sample"""
    def __init__(self,input_size=10):
        super().__init__()
        self.linear = nn.Linear(input_size,32)
        self.linear2 = nn.Linear(32,64)
        self.linear3 = nn.Linear(64,128)
        self.out = nn.Linear(128,1)
        # Defaults
        self.optimizer = torch.optim.SGD(self.parameters(),lr=1e-3)
        self.yhat_val = None
        self.yhat = None
    def forward(self, x):
        x = nn.functional.relu(self.linear(x))
        x = nn.functional.relu(self.linear2(x))
        x = nn.functional.relu(self.linear3(x))
        x = torch.sigmoid(self.out(x))
        return x

    def fit(self,traindataset,epochs=200,batch_size=None, shuffle=False, num_workers=None, optimizer=None,scheduler=None,loss=None,interval=100,valdataset=None,drop_last=False,metrics=None,delay_loss=False,pass_x_biased=False,device='cpu'):
        """ Method used to train the model on traindataset (torch.utils.data.Dataset) """
        if optimizer:
            self.optimizer = optimizer
        if loss:
            self.loss = loss
        if metrics is None:
            metrics = [Metrics(),Metrics(validation=True)]
        if valdataset:
            validation_generator = DataLoader(valdataset,batch_size=len(valdataset),shuffle=False)
        training_generator = DataLoader(traindataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers,drop_last=drop_last)
        
        t0 = time()        
        print("Entering Training...")
        for epoch in range(1,epochs+1):
            for x,y,m in training_generator:
                if device!='cpu':
                    x,y,m = x.to(device),y.to(device),m.to(device)
                #x,y,m = x.float(),y.int(),m.float() #moved to user
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
                       # yval = yval.float().to(device)
                       # x = x.float().to(device)
                       # m = m.float().to(device)
                        self.yhat_val = self(x).view(-1)
                    #l_val = torch.nn.MSELoss()(self.yhat_val,val_data[1])
                    valloss = WeightedMSE(yval)
                    l_val = valloss(self.yhat_val,yval)
                    metrics[0].calculate(pred=self.yhat,target=y,l=l.item())
                    metrics[1].calculate(pred=self.yhat_val,target=yval,l=l_val.item(),m=m)
                    R50 = metrics[1].R50[-1]
                    JSD = metrics[1].JSD[-1]
                    acc_val = metrics[1].accs[-1]
                    acc = metrics[0].accs[-1]
                    print('Epoch:{:04d}/{:04d} || Train: loss:{:.4f}, acc:{:.0f}% || Test: loss: {:.4f}, acc:{:.0f}%, R50: {:.4f}, 1/JSD: {:.4f}  || {:04.1f}s'.format(
                epoch,epochs,l.item(), 100.* acc,
                l_val.item(), 100.* acc_val,R50,1/JSD,time()-t0))
            else:
                if epoch % interval ==0:
                    acc = metrics[0].accs[-1]
                    print('Epoch:{:04d}/{:04d} loss: {:.4f}, accuracy:({:.0f}%)'.format(
                        epoch,epochs,l.item(), 100.* acc))

class DataSet(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, samples,labels,m=None):
        'Initialization'
        self.labels = labels
        self.samples = samples
        self.m = m
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = self.samples[index]
        y = self.labels[index]
        if self.m is not None:
            return X, y, self.m[index]
        else:
            return X, y, 0


class Metrics():
    def __init__(self,validation=False):
        self.validation = validation
        self.losses = []
        self.accs = []
        self.signalE = []
        self.backgroundE= []
        if self.validation:
            self.R50 = []
            self.JSD = []
    def calculate(self,pred,target,l=None,m=None):
        preds = np.array(pred.tolist()).flatten()
        targets = np.array(target.tolist()).flatten()
        acc = (preds.round()==targets).sum()/targets.shape[0]
        signal_efficiency = ((preds.round()==targets)&(targets==1)).sum()/(targets==1).sum()
        background_efficiency = ((preds.round()==targets)&(targets==0)).sum()/(targets==0).sum()
        
        
        if self.validation:
            c = find_threshold(preds,(targets==0),0.5)
            R50 = 1/((preds[targets==1]<c).sum()/(targets==1).sum())
            self.R50.append(R50)
            if m is not None:
                m = np.array(m.tolist()).flatten()
                p, bins = np.histogram(m[targets==1],bins=50,density = True)
                q, _ = np.histogram(m[(targets==1)&(preds<c)],bins=bins,density = True)
                goodidx = (p!=0)&(q!=0)
                p = p[goodidx]
                q = q[goodidx]
                JSD = np.sum(.5*(p*np.log2(p)+q*np.log2(q)-(p+q)*np.log2((p+q)*0.5)))*(bins[1]-bins[0])
                self.JSD.append(JSD)
        self.accs.append(acc)
        self.signalE.append(signal_efficiency)
        self.backgroundE.append(background_efficiency)
        if l:
            self.losses.append(l)

class WeightedMSE():
    def __init__(self,labels):
        ones = sum(labels)
        self.ones_frac = ones/(labels.shape[0]-ones)
    def __call__(self,pred,target):
        weights = target/self.ones_frac + (1-target)
        return torch.mean(weights*(pred-target)**2)

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

class LegendreLoss():
    def __init__(self,x_biased,bins=32,norm="L2"):
        self.mass, self.ordered_mass = torch.sort(x_biased)
        #self.mass = 2*(self.mass-self.mass.min())/(self.mass.max()-self.mass.min())-1 #move to preprocessing later
        self.dm = (self.mass.view(-1,bins)[:,-1] - self.mass.view(-1,bins)[:,0]).view(-1,1)
        self.m = self.mass.view(-1,bins).mean(axis=1).view(-1,1)
        self.p0 = 1
        self.p1 = self.m
        self.p2 = (self.m**2-1)/2
        self.scores = 0
        self.legendre = 0
        self.bins = bins
        self.norm = norm
        #if norm =="L2":
           # self.norm = lambda x,y: (
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
        else:
            legendre_loss = torch.abs(self.scores - self.legendre).mean()
        return legendre_loss

    
class JiangLoss():
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


#def fun(cut,signalE,validation_predictions):
#        passing_cut = (validation_predictions>cut).astype(int)
#        return abs(((passing_cut==y_val)&(y_val==1)).sum()/(y_val==1).sum()-signalE)
#def get_cuts(signalEs,validaiton_predictions):
#    cuts =[]
#    for signalE in signalEs:
#        cuts.append(minimize(fun,[0.5],args=(signalE,validation_predicitons),method="Nelder-mead",bounds=(0,1)).x[0])
#    return cuts

def find_threshold(L, mask, x_frac):
    max_x = mask.sum()
    x = int(np.round(x_frac * max_x))
    L_sorted = np.sort(L[mask.astype(bool)])
    return L_sorted[-x]

