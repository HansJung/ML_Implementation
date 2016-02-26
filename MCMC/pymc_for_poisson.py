import numpy as np
import pymc3 as pm
import scipy as sp
import matplotlib.pyplot as plt


''' Data '''

y = [5,1,5,14,3,19,1,1,4,22] # Number of failure
t = [94,16,63,126,5,31,1,1,2,10] # Observation time length

# Define hyperparameters
alpha = 1.8
gam = 0.01
delta = 1.0
Nobs = len(y)


''' Model '''
HansModel = pm.Model()
with HansModel:
    beta = pm.Gamma('beta_est',alpha=delta,beta=gam)
    lamb = pm.Gamma('lamb_est',alpha=alpha,beta=beta,shape=Nobs)

    # Model param
    poi_mu = t*lamb

    # likelihood
    data = pm.Poisson('data',mu=poi_mu,observed=y)

''' Model fitting'''
with HansModel:
    start = pm.find_MAP(fmin=sp.optimize.fmin_powell)
    Method = pm.HamiltonianMC(vars=[beta,lamb])
    trace = pm.sample(10000,step=Method,start=start)

burnin = 5000
pm.traceplot(trace[burnin:])
print(pm.summary(trace[burnin:]))
plt.show()
