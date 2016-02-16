__author__ = 'jeong-yonghan'

# http://www.people.fas.harvard.edu/~plam/teaching/methods/mcmc/mcmc_print.pdf

import numpy as np
import pymc
import matplotlib.pyplot as plt
import pymc3

# class StandardNormal(pymc.Gibbs):
#     def __init__(self, stochastic, verbose=None):
#         pymc.Gibbs.__init__(self, stochastic, verbose=verbose)
#
#     def step(self):
#         self.stochastic.value = np.random.normal()


y = [5,1,5,14,3,19,1,1,4,22] # Number of failure
t = [94,16,63,126,5,31,1,1,2,10] # Observation time length


# Define hyperparameters
alpha = 1.8
gam = 0.01
delta = 1.0
Nobs = len(y)

beta = pymc.Gamma('beta',alpha=delta, beta=gam, value=1.0)
# lamb = pymc.Gamma('lamb',alpha=alpha, beta=beta, value=np.ones(Nobs))
lamb = np.asarray([pymc.Gamma('lamb_%i'%i,alpha=alpha, beta=beta, value=1.0) for i in range(Nobs)])
lamb = pymc.Container(lamb)
# print lamb
# lamb = np.empty(Nobs,dtype=object)

# for i in range(Nobs):
#     lamb[i] = pymc.Gamma('lamb_%i' %(i+1), alpha = alpha, beta = beta, value=0.5)

@pymc.deterministic
def poi_mu(lamb = lamb, t = t):
    return lamb*t

# @pymc.stochastic
# def data_gen(poi_mu,y):
#     return -np.sum(poi_mu) + np.sum(np.log(poi_mu)*y)
#
# # @pymc.stochastic
# # def data_gen(poi_mu, y):
# #     return pymc.Poisson('data',mu=poi_mu, value = y, observed=True)
# #
data = pymc.Poisson('data',mu=poi_mu, value = y, observed=True)
sampler = pymc.MCMC([lamb,beta,data,y,t])
sampler.use_step_method(pymc.Gibbs,lamb[0],beta)
sampler.sample(iter=10000,burn=3000,thin=10)

print np.mean(beta.trace())
# print np.mean(lamb.trace())
for i in range(Nobs):
    print np.mean(lamb[i].trace())



# MCMC
## Define prior distribution with the initial value