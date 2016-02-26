import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
import scipy as sp


np.random.seed(123)

'''Data generation'''
alpha_true = 1.0
sigma_true = 1.0
beta_true = [1.0,1.25]
Nobs = 100

X1 = np.random.randn(Nobs)
X2 = np.random.randn(Nobs)

Y = alpha_true + beta_true[0]*X1 + beta_true[1]*X2 + np.random.normal(0,sigma_true,Nobs)

# fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10,4))
# axes[0].scatter(X1, Y)
# axes[1].scatter(X2, Y)
# axes[0].set_ylabel('Y'); axes[0].set_xlabel('X1'); axes[1].set_xlabel('X2');

'''Model generation'''
Niter = 3000
Hans_Model = pm.Model()
with Hans_Model:
    # Define prior
    alpha = pm.Normal('alpha_est',mu=0,sd=10)
    beta = pm.Normal('beta_est',mu=0,sd=10,shape=2)
    sigma=pm.HalfNormal('sigma_est',sd=1)

    # Model parameter
    mu = alpha + beta[0]*X1 + beta[1]*X2

    # Likelihood
    Y_rv = pm.Normal('Y_rv',mu=mu,sd=sigma,observed=Y)



''' Model fitting'''
with Hans_Model:
# step = pm.Metropolis(vars=[alpha,beta,sigma])
    param_MAP = pm.find_MAP(fmin = sp.optimize.fmin_powell)
    Method = pm.Slice(vars=[alpha,beta,sigma])
    trace = pm.sample(Niter,step=Method,start=param_MAP)

pm.traceplot(trace)

print pm.summary(trace)

plt.show()
#
# plt.plot(trace['alpha_est'])
# print pm.summary(trace)
# plt.show()