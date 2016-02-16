__author__ = 'jeong-yonghan'
import numpy as np
import matplotlib.pyplot as plt

def loglike(param,y,x,yerr):
    m,b,f = param
    # model_mean = m*x + b
    sum_val = 0.0
    for i in range(len(y)):
        xi = x[i]
        yi = y[i]
        erri = yerr[i]
        model_mean = m*xi + b
        sample_var = (model_mean ** 2) * f**2 + erri**2
        sum_val += -0.5 *((yi - model_mean) ** 2) / sample_var
        sum_val += -0.5 * np.log(sample_var)
    return sum_val

def logprior(param):
    m,b,f = param
    # if -5.0 < m < 0.5:
    m_prior = -(m+1.0) ** 2 / (2.0 * 0.25**2)
    b_prior = -(b-4.0) ** 2 / (2.0 * 0.25**2)
    # if 0.0 < b < 10.0:
    #     b_prior = 0.0
    # else:
    #     b_prior = -np.inf
    if 0 < f < 1:
        f_prior = 0.0
    else:
        f_prior = -np.inf
    return m_prior + f_prior + b_prior

def logprob(param,y,x,yerr ):
    lp = logprior(param)
    ll = loglike(param,y,x,yerr)
    return lp + ll


# Choose the "true" parameters.
m_true = -0.9594
b_true = 4.294
f_true = 0.534

# Generate some synthetic data from the model.
N = 50
x = np.sort(10*np.random.rand(N))
yerr = 0.1+0.5*np.random.rand(N)
y = m_true*x+b_true
y += np.abs(f_true*y) * np.random.randn(N)
y += yerr * np.random.randn(N)

# MCMC
param_dim = 3
MCMC_iter = 5000
burnin = 2000
param_box = np.zeros((MCMC_iter,param_dim))
cov_prop = np.eye(param_dim) * 0.2
param = np.array([-1.0,5.0,0.5])

for i in range(MCMC_iter):
    print i,"th iteration"
    param_cand = np.squeeze(np.random.multivariate_normal(param,cov_prop,1))
    val1 = logprob(param_cand,y,x,yerr)
    val2 = logprob(param,y,x,yerr)
    # numer = np.exp(logprob(param_cand,z,x_true,v))
    # deno = np.exp(logprob(param,z,x_true,v))
    # if not np.isfinite(-val1) or not np.isfinite(-val2):
    #     pass

    alpha = min(1,np.exp(val1-val2))
    print alpha, np.exp(val1-val2)
    uni = np.random.rand(1)
    if alpha > uni[0]:
        param = param_cand
        param_box[i] = param
    elif alpha <= uni[0]:
        param_box[i] = param

param_box = param_box[burnin:,:]
print "-"*50
print np.mean(param_box,axis=0)


for i in range(param_dim):
    plt.figure(i)
    plt.plot(param_box[:,i])
plt.show()
