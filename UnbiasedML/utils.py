from datetime import datetime
import numpy as np
import os
from torch.utils.data import Dataset
import string
from torch.autograd import Function
import torch

def expand_dims(tensor, loc, ntimes=1):
    if ntimes != 1:
        if loc == 0:
            return tensor[(None,)*ntimes]
        elif loc == -1:
            return tensor[(...,)+(None,)*ntimes]
        else:
            raise ValueError('Cannot insert arbitray number of dimensions in the middle of the tensor.')
    else:
        return tensor.unsqueeze(loc)

def expand_dims_as(t1,t2):
    result = t1[(...,)+(None,)*t2.dim()] 
    return result 

def Heaviside(tensor):
    tensor[tensor>0] = 1
    tensor[tensor==0] = 0.5
    tensor[tensor<0] = 0
    return tensor



class LegendreFitter():
    def __init__(self,mbins=None,m=None,order=0,power=1):
        """
        Object used to fit an array of using Legendre polynomials.

        Parameters
        ----------
        mbins :int or Array[float] (optional)
            Array of bin edges or number of bins in m used in the fit. The fit is integrated along m.
        m : Array[float] (optional)
            Array of all masses. Has shape (mbins,bincontent)
        order : int, default 0
            The highest order of legendre polynomial used in the fit.
        power : int, default 1
            Power used in the norm of the difference between the input and the fit. |fit(input) - input|**power
        """
        if m is None:
            if mbins is None:
                raise ValueError("Provide either m or mbins.")
            mbins = torch.linspace(-1,1,mbins+1) # mass bin edges
            m  = (mbins[1:] + mbins[:-1])*0.5    # mass bin centers
            dm = mbins[1:] - mbins[:-1]          # mass bin widths
        else:
            if type(m) != torch.Tensor:
                m = torch.DoubleTensor(m)
            dm = m.max(axis=1)[0] - m.min(axis=1)[0]  # bin widths for each of the mbins.
            m  = m.mean(axis=1) # bin centers
        self.m = m.view(-1)
        self.dm = dm.view(-1)
        self.mbins = self.m.shape[0]
        self.power = power
        self.order = order

    def __call__(self,F):
        """
        Fit F with Legendre polynomials and return the fit.

        Parameters
        ----------
        F : torch.Tensor
            Tensor of CDFs F_m(s) has shape (N,mbins) where N is the number of scores 
        """
        a0 = 1/2 * (F*self.dm).sum(axis=-1).view(-1,1) #integrate over mbins
        fit = a0.expand_as(F) # make boradcastable
        if self.order>0:
            a1 = 3/2 * (F*self.m*self.dm).sum(axis=-1).view(-1,1)
            fit = fit + a1*self.m
        if self.order>1:
            p2 = (self.m**2-1)*0.5
            a2 = 5/2 * (F*p2*self.dm).sum(axis=-1).view(-1,1)
            fit = fit+ a2*p2
        return fit

class LegendreIntegral(Function): 
    @staticmethod
    def forward(ctx, input, fitter,sbins=None):
        """
        Calculate the Flat loss of input integral{Norm(F(s)-F_flat(s))} integrating over sbins.

        Parameters
        ----------
        input : torch.Tensor
            Scores with shape (mbins,bincontent) where mbins * bincontent = N (or the batch size.)
        fitter : LegendreFitter
            Fitter object used to calculate F_flat(s)
        sbins : int
            Number of s bins to use in the integral.
        """
        s_edges = torch.linspace(0,1,sbins+1,dtype=torch.double).to(input.device) #create s edges to integrate over
        s = (s_edges[1:] + s_edges[:-1])*0.5
        s = expand_dims_as(s,input)
        ds = s_edges[1:] - s_edges[:-1]

        F = Heaviside(s-input).sum(axis=-1).double()/input.shape[-1] # get CDF at s from input values
        integral = (ds.matmul((F-fitter(F))**fitter.power)).sum(axis=0)/input.shape[0]
        
        F_s_i =  expand_dims_as(input.view(-1),input) #make a flat copy of input and add dimensions for boradcasting
        F_s_i =  Heaviside(F_s_i-input).sum(axis=-1).double()/input.shape[-1] #sum over bin content to get CDF
        residual = F_s_i - fitter(F_s_i)
        ctx.fitter = fitter
        ctx.residual = residual
        ctx.shape = input.shape
        return integral
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        shape = ctx.shape
        power = ctx.fitter.power
        order = ctx.fitter.order
        dm = ctx.fitter.dm
        m = ctx.fitter.m
        if ctx.needs_input_grad[0]:
            first_term = ctx.residual[torch.repeat_interleave(torch.eye(shape[0],dtype=bool),shape[1],axis=0)]
            # repeat_interleave is like numpy.repeat it repeats entries along some axis.
            first_term  = first_term.view(shape)
            second_term = ctx.residual.sum(axis=-1)
            second_term = second_term.view(shape) * dm.view(-1,1) * -0.5
            summation = first_term + second_term

            if order >0:
                third_term  = (ctx.residual*m).sum(axis=-1).view(shape) * (dm*m).view(-1,1) * -1.5
                summation += third_term

            grad_input  = grad_output  \
             * (-power)*(summation)/np.prod(shape)

        return grad_input, None, None

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
        f.write("\n"+"*"*25+"Finished Training Successfully"+"*"*25)
        f.write("\n\n")
        f.close()

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


class DataSet(Dataset):
    def __init__(self, samples,labels,m=None):
        'Initialization'
        self.labels = labels
        self.samples = samples
        self.m = m
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
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
                m = np.array(m.tolist()).flatten()*250
                p, bins = np.histogram(m[targets==1],bins=20,density = True)
                q, _ = np.histogram(m[(targets==1)&(preds<c)],bins=bins,density = True)
                goodidx = (p!=0)&(q!=0)
                p = p[goodidx]
                q = q[goodidx]
                JSD = np.sum(.5*(p*np.log2(p)+q*np.log2(q)-(p+q)*np.log2((p+q)*0.5)))#*(bins[1]-bins[0])
                self.JSD.append(JSD)
        self.accs.append(acc)
        self.signalE.append(signal_efficiency)
        self.backgroundE.append(background_efficiency)
        if l:
            self.losses.append(l)


def find_threshold(L,mask,x_frac):
    """
    Calculate c such that x_frac of the array is less than c.

    Parameters
    ----------
    L : Array
        The array where the cutoff is to be found
    mask : Array,
        Mask that returns L[mask] the part of the original array over which it is desired to calculate the threshold.
    x_frac : float
        Of the area that is lass than or equal to c.

    returns c (type=L.dtype)
    """
    max_x = mask.sum()
    x = int(np.round(x_frac * max_x))
    L_sorted = np.sort(L[mask.astype(bool)])
    return L_sorted[x]
