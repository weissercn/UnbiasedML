from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
#import uproot
import ROOT
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.proportion import proportion_confint

_lmax  = 20


def plot_hist(bkgnd, sign, bins):
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

def plot_hist_filt(bkgnd, bkgnd_filt, bins, label):

    #plt.hist(bkgnd, alpha=0.5)
    #plt.hist([bkgnd, sign], bins=bins, histtype="barstacked", alpha=0.5, label=['bkgnd', 'sign'])
    plt.hist(bkgnd, label="bkgnd", bins=bins, alpha=0.5)
    plt.hist(bkgnd_filt, label="bkgnd filter", bins=bins, alpha=0.5)


    #plt.hist([bkgnd, sign], bins=bins, histtype='step', fill=False)
    plt.legend()
    plt.yscale("log")
    plt.xlabel('mass')
    plt.ylabel('count')
    plt.title('Events')
    #plt.show()
    plt.savefig("bkgnd_filt_{}.png".format(label))

def poly_lin(n,nbins):
    # could use hermite, legendre, Chebychev polynomials.
    res = []
    for i in range(n):
        res.append([pow(-1 +2*(0.5+ ibin)/nbins, i) for ibin  in range(nbins) ]) #moving to a range between -1 and 1, which all bins span. want bin centers
    return res




def ML_Cut(data, bins, fun, label, PLOT=False, MOMENTS= "legendre"):
    """
    ML_Cut simulates a machine learning cut on the data by accepting events probabilistically depending on the function fun. 
    It then calculates the fraction of accepted events and the corresponding moments and their errors and plots the result
    """
    #bin_indices = np.digitize(data, bins)
    #pass_bool = [np.random.uniform()>0.5 for ebkgnd  in bkgnd]

    z = 1.96 # for 95% CL

    #data_filt = [datum  for datum in data  if  np.random.uniform()>0.2]
    #data_filt = [datum  for datum in data  if  np.random.uniform()>datum]


    data_filt = fun(data)

    if label== "leg3":
        plot_hist_filt(data, data_filt, bins, label)

    cont_all, _ = np.histogram(data, bins=bins)
    cont_filt, _ = np.histogram(data_filt, bins=bins)
    bin_centers = 0.5*( bins[1:] + bins[:-1])

    np.seterr(divide="ignore")
    frac = cont_filt*1./ cont_all
    #frac_errs_old = z*1./cont_all * np.sqrt(cont_filt*(cont_all-cont_filt)/cont_all) #problem: gaussian approximation doesn't hold. # is the same as proportion_confint(method="normal")
    frac_errs = proportion_confint(cont_filt, cont_all, method="agresti_coull") #{'normal', 'agresti_coull', 'beta', 'wilson', 'binom_test'}
    frac_errs = (frac_errs[1]-frac_errs[0])/2.
    #print("frac_errs : ", frac_errs)
    #print("frac_errs_new : ", frac_errs_new) 
    np.seterr(divide="warn")
    #print("\n ", label)
    #print("cont_filt : ",cont_filt)
    #print("cont_all : ", cont_all)
    #print("frac : ", frac)      




    if MOMENTS== "legendre":
        _hleg = get_legendre(bins)
        nb = len(bins)-1
        moments = []
        moment_errs = []
        for l in range(_lmax):
            val = 0
            val_err_sq = 0
            nempty = 0
            for b in range(nb):
                #print("hi")
                #print("l, b, _hleg[l] ", l, b, _hleg[l])
                dx = _hleg[l].GetBinWidth(1+b)
                if frac[b]==np.nan: nempty +=1; continue #sometimes we just don't have any event in a bin
                val += frac[b]* _hleg[l].GetBinContent(1+b)*dx
                val_err_sq += (frac_errs[b]* _hleg[l].GetBinContent(1+b)*dx)**2
            moments.append(val*(2*l+1)*(nb/(float(nb)-nempty)))  #/_hdata->GetBinWidth(1) 
            moment_errs.append(math.sqrt(val_err_sq)*(2*l+1)/2.*(nb/(float(nb)-nempty)))

    if MOMENTS=="linear": 
        llin = poly_lin(_lmax,len(bins)-1)
        moments = []
        for l in range(_lmax):
            moments.append(sum(i[0] * i[1] for i in zip(frac, llin[l])))

    if PLOT:
        print("PLOT is True")
        fig = plt.figure() 
        ax = fig.add_subplot(1, 1, 1)
        #ax.plot(np.arange(10),12*np.arange(10)) 
        #ax.text(0.4, 0.7, 'Correct Position', transform=ax.transAxes)   

        #plt.plot(bin_centers, frac)

        plt.errorbar(bin_centers, frac , frac_errs)
        plt.title("ML response: Legendre {}".format(label[-1]))
        #plt.text(60, .025, 'hi')
        ax.text(0.04, 0.92, r'moment 0 {0:.1f}'.format( moments[0] ), transform = ax.transAxes)
        ax.text(0.04, 0.84, r'moment 1 {0:.1f}'.format( moments[1] ), transform = ax.transAxes)
        ax.text(0.04, 0.76, r'moment 2 {0:.1f}'.format( moments[2] ), transform = ax.transAxes)
        ax.text(0.04, 0.68, r'moment 3 {0:.1f}'.format( moments[3] ), transform = ax.transAxes)
        plt.xlabel("mass")
        plt.ylabel('Frac passing')
        #plt.show()
        plt.savefig("frac_"+label+".png")
        plt.clf()

    return moments, moment_errs

def random_linear(m=1, n=1E6):
    # y= 1 + (x-0.5)*m 
    # produces lines that pass through (0.5,1) and are normalised between 0 and 1.
    # the cdf of this is 
    # (sqrt(m^2 + m (8 z - 4) + 4) + m - 2)/(2 m)
    assert(abs(m)<=2)
    samples = np.random.uniform(size=int(n))
    samples = (np.sqrt(m*m + m *(8*samples - 4) + 4) + m - 2)/(2 *m)
    return samples
    

def cut_unif(data):
    cutoff = 0.5
    return [datum  for datum in data  if  np.random.uniform()>cutoff]

def cut_lin(data):
    return [datum  for datum in data  if  np.random.uniform()<datum]

def cut_square(data):
    return [datum  for datum in data  if  np.random.uniform()<np.power(datum, 2.)]

def cut_legendre(l):
    # given a legendre order, create a function that rejects events probabilistically according to a legendre polynomial
    def cut_alegendre(data):
        # assumes data is between -1 and 1
        from scipy.special import legendre
        Pn = legendre(l)
        return np.array([datum  for datum in data  if  2*np.random.uniform()-1<Pn(datum)])

    return cut_alegendre


def plot_moments( moments, moment_errs, title):
    # plot the moments and their error. If the data is only a single legendre mode, then only this mode should be consistent with  1, the others with 0. the 0th moment will always be large
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.errorbar(range(len(moments)), moments , moment_errs)
    plt.xlabel("legendre modes for {}".format(title))
    plt.ylabel('moment')
    plt.title('Input: Legendre mode {}'.format(title[-1]))
    #plt.show()
    plt.savefig(title +".png")
    plt.clf()





def get_legendre(bins):
    # This gets the integral of the (normalised) legendre polynomial over the range of the bin
    # To calculate the moments multiply the bin content by the integral of the (normalised) legendre polynomial over the range of the bin.
    # In order to compute this we cannot just use the integral of the polynomial by hand, because of numerical instabilities. Instead rely on the recursive formula.
    # This takes quite some time and instead can just download the result here (https://gitlab.cern.ch/LHCb-QEE/darkphotonrun2/blob/master/prompt/bump_hunt/leg_poly.root).
    # We just need to rebin. Look at code here (https://gitlab.cern.ch/LHCb-QEE/darkphotonrun2/blob/master/prompt/Fit.h)
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

def test_toy():
    bkgnd_raw = np.random.exponential(scale=0.15, size = 1000000).reshape(-1, 1)
    sign_raw  = np.random.normal(loc=0.3, scale=0.1, size = 50000).reshape(-1, 1)

    #plot_hist(bkgnd_raw, sign_raw, bins=np.linspace(0. ,1., 51))

    if False:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(bkgnd_raw)
        bkgnd = scaler.transform(bkgnd_raw)[:,0]
        sign = scaler.transform(sign_raw)[:,0]
    else:
        # We have to avoid bins without events for now
        bkgnd = 2*bkgnd_raw-1
        sign = 2*sign_raw-1



    bins = np.linspace(-1. ,1., 51)

    
    #for l in range(5):
    for l in [3]:
        moments, moment_errs = ML_Cut(bkgnd, bins, cut_legendre(l), "leg{}".format(l), PLOT=True)
        plot_moments( moments, moment_errs, "leg{}".format(l))

    print("moments : ", moments)
    

if __name__=="__main__":

    test_toy()

