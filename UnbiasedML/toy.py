from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



bkgnd = np.random.exponential(scale=0.1, size = 1000000)
sign  = np.random.normal(loc=0.3, scale=0.1, size = 50000)

bins = np.linspace(0.,1., 31)



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




def ML_Cut(data, bins, fun, label):
    #bin_indices = np.digitize(data, bins)
    #pass_bool = [np.random.uniform()>0.5 for ebkgnd  in bkgnd]

    z = 1.96 # for 95% CL

    #data_filt = [datum  for datum in data  if  np.random.uniform()>0.2]
    #data_filt = [datum  for datum in data  if  np.random.uniform()>datum]

    data_filt = fun(data)

    cont_all, _ = np.histogram(data, bins=bins)
    cont_filt, _ = np.histogram(data_filt, bins=bins)
    bin_centers = 0.5*( bins[1:] + bins[:-1])


    frac = cont_filt/ cont_all
    frac_errs = z/cont_all * np.sqrt(cont_filt*(cont_all-cont_filt)/cont_all)
    sfrac = sum(frac)

    llin = poly_lin(5,len(bins)-1)

    
    fig = plt.figure() 
    ax = fig.add_subplot(1, 1, 1)
    #ax.plot(np.arange(10),12*np.arange(10)) 
    #ax.text(0.4, 0.7, 'Correct Position', transform=ax.transAxes)   

    #plt.plot(bin_centers, frac)

    plt.errorbar(bin_centers, frac , frac_errs)
    plt.title(label)
    #plt.text(60, .025, 'hi')
    ax.text(0.04, 0.92, r'moment 0 {0:.1f}'.format( sum(i[0] * i[1] for i in zip(frac, llin[0])) ), transform = ax.transAxes)
    ax.text(0.04, 0.84, r'moment 1 {0:.1f}'.format( sum(i[0] * i[1] for i in zip(frac, llin[1]))  ), transform = ax.transAxes)
    ax.text(0.04, 0.76, r'moment 2 {0:.1f}'.format( sum(i[0] * i[1] for i in zip(frac, llin[2])) ), transform = ax.transAxes)
    ax.text(0.04, 0.68, r'moment 3 {0:.1f}'.format( sum(i[0] * i[1] for i in zip(frac, llin[3])) ), transform = ax.transAxes)
    plt.xlabel("mass")
    plt.ylabel('frac passing')
    #plt.show()
    plt.savefig(label+".png")
    plt.clf()

def cut_unif(data):
    cutoff = 0.5
    return [datum  for datum in data  if  np.random.uniform()>cutoff]

def cut_lin(data):
    return [datum  for datum in data  if  np.random.uniform()<datum]

def cut_square(data):
    return [datum  for datum in data  if  np.random.uniform()<np.power(datum, 2.)]


ML_Cut(bkgnd, bins, cut_unif, "unif")
ML_Cut(bkgnd, bins, cut_lin, "lin")
ML_Cut(bkgnd, bins, cut_square, "square")




plot_hist(bkgnd, sign)






