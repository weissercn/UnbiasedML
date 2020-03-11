from datetime import datetime
import numpy as np
import os
from torch.utils.data import Dataset

import string
class PartialFormatter(string.Formatter):
    def __init__(self, missing='~~', bad_fmt='None'):
        self.missing, self.bad_fmt=missing, bad_fmt

    def get_field(self, field_name, args, kwargs):
        # Handle a key not found
        try:
            val=super(PartialFormatter, self).get_field(field_name, args, kwargs)
            # Python 3, 'super().get_field(field_name, args, kwargs)' works
        except (KeyError, AttributeError):
            val=None,field_name
        return val

    def format_field(self, value, spec):
        # handle an invalid format
        if value==None: return self.missing
        try:
            return super(PartialFormatter, self).format_field(value, spec)
        except ValueError:
            if self.bad_fmt is not None: return self.bad_fmt
            else: raise

class Logger():
    def __init__(self,file="./logs/log.txt",overwrite=True):
        """
        Creates a log object to be passed to Classifer.fit()

        Parameters
        ----------
        file : string 
            Location of the log file. If the the path to the file does not exist. The logger will create one.
        overwrite  :  bool 
            If False and the file already exists the name will be incremented. Otherwise, the file is overwritten.
        """
        path, fname= '/'.join(file.split("/")[:-1]), file.split("/")[-1]
        if not os.path.exists(path):
            os.mkdir(path)
        i=0
        if not overwrite:
            try:
                name,ext = fname.split(".")
                while os.path.exists("{}/{}({}).{}".format(path,name,i,ext)):
                    i += 1
                file = "{}/{}({}).{}".format(path,name,i,ext)
            except ValueError:
                while os.path.exists("{}{}({})".format(path,name,i)):
                    i += 1
                file = "{}{}({}).{}".format(path,name,i,ext)
        self.file= file
        f = open(self.file,"w")
        f.close()
    def initialize(self,params,loss,optimizer):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        f = open(self.file,"a")
        f.write("**"*40+"\n")
        f.write("Model initialized on: {}\n".format(dt_string))
        maxstr = max([len(key) for key in params.keys()])
        string = '{:^%d}' %(maxstr-1)*len(params)
        keys = string.format(*params.keys())
        values = PartialFormatter().format(string,*params.values())
        f.write("\n".join([keys,values]))
        f.write("\n\n")
        f.write("**Loss**\n")
        f.write(repr(loss))
        f.write("\n\n")
        f.write("**Optimizer**\n")
        f.write("".join(repr(optimizer).strip().split("\n")))
        f.write("\n\n")
        f.close()
    def entry(self,string):
        f = open(self.file,"a")
        f.write(string)
        f.write("\n")
        f.close()
    def finished(self):
        f = open(self.file,"a")
        f.write("*"*15+"Finished Training Successfully"+"*"*15)
        f.write("\n\n\n")
        f.close()

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


def find_threshold(L, mask, x_frac):
    max_x = mask.sum()
    x = int(np.round(x_frac * max_x))
    L_sorted = np.sort(L[mask.astype(bool)])
    return L_sorted[-x]
