import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from scipy.optimize import minimize

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10,32)
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

    def fit(self,x,y,n_epochs=200,optimizer=None,scheduler=None,loss=None,interval=100,val_data=[],metrics=None,delay_loss=False):
        if optimizer:
            self.optimizer = optimizer
        if loss:
            self.loss = loss
        for epoch in range(n_epochs):
            self.train()
            self.yhat = self(x)
            if epoch<delay_loss:
                l = torch.nn.MSELoss()(self.yhat,y)
            else:
                l = self.loss(self.yhat,y)
            l.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            if scheduler:
                scheduler.step()

            if metrics is None:
                metrics = [Metrics(),Metrics()]

    #Validation and Printing
            if val_data:
                if epoch % interval ==0 or epoch == n_epochs-1:
                    self.train(False)
                    self.yhat_val = self(val_data[0])
                    l_val = torch.nn.MSELoss()(self.yhat_val,val_data[1] )
                    
                    metrics[0].calculate(pred=self.yhat,target=y)
                    metrics[1].calculate(pred=self.yhat_val,target=val_data[1],l=l_val.item())
                    acc_val = metrics[1].accs[-1]
                    acc = metrics[0].accs[-1]
                    print('Epoch:{:04d}/{:04d} || Train: loss:{:.4f}, acc:{:.0f}% || Test: loss: {:.4f}, acc:{:.0f}%'.format(
                epoch,n_epochs,l.item(), 100.* acc,
                l_val.item(), 100.* acc_val))
            else:
                if epoch % interval ==0:
                    acc = metrics[0].accs[-1]
                    print('Epoch:{:04d}/{:04d} loss: {:.4f}, accuracy:({:.0f}%)'.format(
                        epoch,n_epochs,l.item(), 100.* acc))




class Metrics():
    def __init__(self):
        self.losses = []
        self.accs = []
        self.signalE = []
        self.backgroundE= []
    def calculate(self,pred,target,l=None,validation=False):
        acc = (pred.round()==target).sum().item()/target.shape[0]
        signal_efficiency = ((pred.round()==target)&(target==1)).sum().item()/(target==1).sum().item()
        background_efficiency = ((pred.round()==target)&(target==0)).sum().item()/(target==0).sum().item()
        self.accs.append(acc)
        self.signalE.append(signal_efficiency)
        self.backgroundE.append(background_efficiency)
        if l:
            self.losses.append(l)

class LegendreLoss():
    def __init__(self,x_biased,frac=0.9):
        self.frac = frac
        self.mass = np.sort(x_biased)
        self.ordered_mass = np.argsort(x_biased)
        self.dm = torch.from_numpy(self.mass.reshape(-1,100)[:,-1] - self.mass.reshape(-1,100)[:,0]).float().view(-1,1)
        self.m = torch.from_numpy(self.mass.reshape(-1,100).mean(axis=1)).float().view(-1,1)
        self.p0 = 1
        self.p1 = self.m
        self.p2 = (self.m**2-1)/2
        self.scores = 0
        self.legendre = 0
    def __call__(self,pred,target):   
        pred_bins = pred[self.ordered_mass].reshape(-1,100)
        ordered_s = pred_bins.argsort(axis=1)
        self.scores = pred_bins 
        self.scores = torch.cumsum(self.scores,axis=1)/self.scores.sum(axis=1).view(-1,1)
        a0 = 1/2 * (self.scores*self.dm).sum(axis=0)
        a1 = 3/2 * (self.scores*self.p1*self.dm).sum(axis=0)
        self.legendre = a0 + a1*self.p1 
        legendre_loss = ((self.scores - self.legendre)**2).mean()
        return legendre_loss*self.frac + torch.nn.MSELoss()(pred,target)*(1-self.frac)

    
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


def fun(cut,signalE,validation_predictions):
        passing_cut = (validation_predictions>cut).astype(int)
        return abs(((passing_cut==y_val)&(y_val==1)).sum()/(y_val==1).sum()-signalE)
def get_cuts(signalEs,validaiton_predictions):
    cuts =[]
    for signalE in signalEs:
        cuts.append(minimize(fun,[0.5],args=(signalE,validation_predicitons),method="Nelder-mead",bounds=(0,1)).x[0])
    return cuts

def find_threshold(L, mask, x_frac):
    max_x = mask.sum()
    x = int(np.round(x_frac * max_x))
    L_sorted = np.sort(L[mask.astype(bool)])
    return L_sorted[-x]

