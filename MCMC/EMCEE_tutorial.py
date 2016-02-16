import numpy as np
import emcee
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

ndim = 1
icov = 0.25**2 * np.eye(ndim)

def eta(m):
    return 3.0 * (m**2)

def lnprior(m):
    return -(m-0.5)**2/(2.0 * 0.25**2)

def lnlike(m,icov,y):
    eta_m = eta(m)
    return - np.dot(np.dot(y-eta_m, icov),y-eta_m)/2.0

def lnprob(m,icov,y):
    lp = lnprior(m)
    ll = lnlike(m,icov,y)
    return lp + ll

nwalkers = 10
y = np.random.multivariate_normal(np.zeros(ndim),icov)
p0 = [np.random.rand(ndim) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[icov,y])
sampler.run_mcmc(p0,1000)

print np.mean(sampler.flatchain,axis=0)
print np.var(sampler.flatchain)
