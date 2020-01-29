from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
#import uproot
import ROOT
from sklearn.preprocessing import MinMaxScaler


_lmax  = 20


def plot_hist(bkgnd, sign):
    #plt.hist(bkgnd, alpha=0.5)
    #plt.hist([bkgnd, sign], bins=bins, histtype="barstacked", alpha=0.5, label=['bkgnd', 'sign'])
    
    plt.hist(bkgnd, label="bkgnd", bins=bins, alpha=0.5)
    #plt.hist(sign, label="sign", bottom=bkgnd,  bins=bins, alpha=0.5)


    #plt.hist([bkgnd, sign], bins=bins, histtype='step', fill=False)
    plt.yscale("log")
    plt.xlabel('mass')
    plt.ylabel('count')
    #plt.show()
    plt.savefig("bkgnd.png")

def poly_lin(n,nbins):
    # could use hermite, legendre, Chebychev polynomials.
    res = []
    for i in range(n):
        res.append([pow(-1 +2*(0.5+ ibin)/nbins, i) for ibin  in range(nbins) ]) #moving to a range between -1 and 1, which all bins span. want bin centers
    return res




def ML_Cut(data, bins, fun, label, PLOT=False, MOMENTS= "legendre"):
    #bin_indices = np.digitize(data, bins)
    #pass_bool = [np.random.uniform()>0.5 for ebkgnd  in bkgnd]

    z = 1.96 # for 95% CL

    #data_filt = [datum  for datum in data  if  np.random.uniform()>0.2]
    #data_filt = [datum  for datum in data  if  np.random.uniform()>datum]


    data_filt = fun(data)

    cont_all, _ = np.histogram(data, bins=bins)
    cont_filt, _ = np.histogram(data_filt, bins=bins)
    bin_centers = 0.5*( bins[1:] + bins[:-1])

    frac = cont_filt*1./ cont_all
    frac_errs = z*1./cont_all * np.sqrt(cont_filt*(cont_all-cont_filt)/cont_all)


    if MOMENTS== "legendre":
        _hleg = get_legendre(bins)
        nb = len(bins)-1
        moments = []
        moment_errs = []
        for l in range(_lmax):
            val = 0
            val_err_sq = 0
            for b in range(1, nb):
                #print("hi")
                #print("l, b, _hleg[l] ", l, b, _hleg[l])
                dx = _hleg[l].GetBinWidth(b)
                val += frac[b]* _hleg[l].GetBinContent(b)*dx
                val_err_sq += (frac_errs[b]* _hleg[l].GetBinContent(b)*dx)**2
            moments.append(val*(2*l+1)/2.)  #/_hdata->GetBinWidth(1) 
            moment_errs.append(math.sqrt(val_err_sq)*(2*l+1)/2.)

    if MOMENTS=="linear": 
        llin = poly_lin(5,len(bins)-1)
        moments = []
        for l in range(5):
            moments.append(sum(i[0] * i[1] for i in zip(frac, llin[l])))

    if PLOT:
        
        fig = plt.figure() 
        ax = fig.add_subplot(1, 1, 1)
        #ax.plot(np.arange(10),12*np.arange(10)) 
        #ax.text(0.4, 0.7, 'Correct Position', transform=ax.transAxes)   

        #plt.plot(bin_centers, frac)

        plt.errorbar(bin_centers, frac , frac_errs)
        plt.title(label)
        #plt.text(60, .025, 'hi')
        ax.text(0.04, 0.92, r'moment 0 {0:.1f}'.format( moments[0] ), transform = ax.transAxes)
        ax.text(0.04, 0.84, r'moment 1 {0:.1f}'.format( moments[1] ), transform = ax.transAxes)
        ax.text(0.04, 0.76, r'moment 2 {0:.1f}'.format( moments[2] ), transform = ax.transAxes)
        ax.text(0.04, 0.68, r'moment 3 {0:.1f}'.format( moments[3] ), transform = ax.transAxes)
        plt.xlabel("mass")
        plt.ylabel('frac passing')
        #plt.show()
        plt.savefig(label+".png")
        plt.clf()

    return moments, moment_errs

def cut_unif(data):
    cutoff = 0.5
    return [datum  for datum in data  if  np.random.uniform()>cutoff]

def cut_lin(data):
    return [datum  for datum in data  if  np.random.uniform()<datum]

def cut_square(data):
    return [datum  for datum in data  if  np.random.uniform()<np.power(datum, 2.)]

def cut_legendre(l):
    def cut_alegendre(data):
        # assumes data is between -1 and 1
        from scipy.special import legendre
        Pn = legendre(l)
        return [datum  for datum in data  if  np.random.uniform()<Pn(datum)]

    return cut_alegendre


def plot_moments( moments, moment_errs):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.errorbar(range(len(moments)), moments , moment_errs)
        plt.xlabel("legendre modes")
        plt.ylabel('moment')
        #plt.show()
        plt.savefig("legendre.png")
        plt.clf()


def test_toy():

    bkgnd = np.random.exponential(scale=0.1, size = 1000000)
    sign  = np.random.normal(loc=0.3, scale=0.1, size = 50000)

    bins = np.linspace(0.,1., 31)

    ML_Cut(bkgnd, bins, cut_unif, "unif")
    ML_Cut(bkgnd, bins, cut_lin, "lin")
    ML_Cut(bkgnd, bins, cut_square, "square")

    plot_hist(bkgnd, sign)

def get_legendre(bins):
        assert 10000%(len(bins)-1)==0, "Need to be able to rebin properly"
        rebin = 10000/(len(bins)-1) # should be integer

        _hleg = []
        #f = uproot.open("data/leg_poly.root")
        f = ROOT.TFile.Open("data/leg_poly.root")
        for ileg in range(_lmax):
            h = f.Get("hL{}".format(ileg))
            h.Rebin(rebin)
            h.Scale(1/float(rebin))
            h.SetDirectory(0)
            _hleg.append(h)

        return _hleg
"""
def legYield(b, pars, _hleg): # b is bin
        dx = lh[0].GetBinWidth(b)
        val = 0
        for i in range(l_max):
            val += pars[i]*_hleg[i].GetBinContent(b)*dx
        return val

    # calculate the moment of order l on _hdata
    # if _hbkg != 0, subtract this from the moments
def legMoment(l, bins, _hdata, _hleg):
        p = [0]*21
        p[l] = 1

        val = 0;

        nb = len(nbins)-1
        for b in range(1, nb):
            o = _hdata.GetBinContent(b)
            y = legYield(b,p, _hleg);
            val += y*o;
        # we are not vetoing regions and hence, dont't have to rescale based on that.

        return val*(2*l+1)/2./_hdata.GetBinWidth(1);
"""


if __name__=="__main__":

    bkgnd_raw = np.random.exponential(scale=0.1, size = 1000000) #.reshape(-1, 1)
    sign_raw  = np.random.normal(loc=0.3, scale=0.1, size = 50000) #.reshape(-1, 1)

    """
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(bkgnd_raw)
    bkgnd = scaler.transform(bkgnd_raw)[:,0]
    sign = scaler.transform(sign_raw)[:,0]
    """
    bkgnd, sign = bkgnd_raw, sign_raw

    print("bkgnd : ", bkgnd, bkgnd.shape)

    bins = np.linspace(-1 ,1., 51)

    #test_toy()
    #_hleg = get_legendre(bins)
    #moments = []
    #for l in range(_lmax):
    #    moments.append(legMoment(l, bins, _hdata, _hleg))


    moments, moment_errs = ML_Cut(bkgnd, bins, cut_legendre(0), "leg0")

    plot_moments( moments, moment_errs)

    print("moments : ", moments)

