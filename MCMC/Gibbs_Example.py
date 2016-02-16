__author__ = 'jeong-yonghan'

import numpy as np
import math
import matplotlib.pyplot as plt


# Refer: http://pareto.uab.es/mcreel/IDEA2015/MCMC/mcmc.pdf

def fact_log(x):
    sumval = 0.0
    for i in range(x):
        sumval += i
    return sumval

# Data generation
y = [5,1,5,14,3,19,1,1,4,22] # Number of failure
t = [94,16,63,126,5,31,1,1,2,10] # Observation time length

''' 1. MCMC '''

N = len(y)

# Initialize the parameters
lamb = np.random.uniform(1,2,N)
beta = np.random.uniform(1,2,1)
# lamb = np.zeros(N)
# beta = np.zeros(1)
theta = np.concatenate([lamb,beta])


# print gamma_fun(10)

def logprior(theta):
    lamb = theta[:N]
    beta = theta[N]

    alpha = 1.8
    gam = 0.01
    delta = 1.0

    sumval = 0.0
    for i in range(len(lamb)):
        lami = lamb[i]
        if lami < 0.0:
            return -np.inf
        sumval += alpha*np.log(beta) - np.log(math.gamma(alpha)) + (alpha-1) * np.log(lami) - beta*lami

    sumval += gam*np.log(delta)  + (gam-1)*np.log(beta) - delta*beta
    return sumval

def loglike(theta,y,t):
    lamb = theta[:N]
    beta = theta[N]

    sumval = 0.0
    for i in range(len(lamb)):
        lami = lamb[i]
        if lami < 0.0:
            return -np.inf
        yi = y[i]
        ti = t[i]
        sumval += -lami*ti + yi*np.log(lami * ti)
    return sumval


def logpost(theta, y,t):
    lp = logprior(theta)
    ll = loglike(theta,y,t)

    return lp + ll


# Gibbs implementation
param_dim = 11
lam_dim = 10
Gibbs_iter = 1000
burn_in = 0

## Hyper parameter definition
alpha = 1.8
gam = 0.01
delta = 1.0

## parameter saving
param_box = np.zeros((Gibbs_iter,param_dim))
param_box[0] = theta

# print np.random.gamma(14,126,1)

# #
# for idx in range(1,Gibbs_iter):
#     # lambda update
#     beta = param_box[idx-1][lam_dim]
#     sum_lam = 0.0
#     for lam_idx in range(lam_dim):
#         yi = y[lam_idx]
#         ti = t[lam_idx]
#         param_box[idx][lam_idx] = np.random.gamma(shape=yi+alpha, scale=ti+beta,size=1)
#         sum_lam += param_box[idx][lam_idx]
#
#     # beta update
#     param_box[idx][lam_dim] = np.random.gamma(shape=lam_dim*alpha+gam, scale=delta+sum_lam,size=1)
#     print sum_lam, param_box[idx]
#
# param_box = param_box[burn_in:,:]
# print param_box.shape
# print "-"*50
# Est = np.mean(param_box,axis=0)
# Var = np.var(param_box,axis=0)
#
# for i in range(len(Est)):
#     if i < 10:
#         print "lam",i,Est[i],Var[i]
#     else:
#         print "beta",Est[i],Var[i]



# MCMC example
param_dim = 11
MCMC_iter = 10000
burn_in = 3000
param_box = np.zeros((MCMC_iter,param_dim))
Param_BOX = list()
cov_prop = np.eye(param_dim) * 0.01
for i in range(MCMC_iter):
    print i,"th iteration"
    theta_cand = np.squeeze(np.random.multivariate_normal(theta,cov_prop,1))
    val1 = logpost(theta_cand,y,t)
    val2 = logpost(theta,y,t)
    # numer = np.exp(logprob(param_cand,z,x_true,v))
    # deno = np.exp(logprob(param,z,x_true,v))
    # if not np.isfinite(-val1) or not np.isfinite(-val2):
    #     pass

    alpha = min(1.0,np.exp(val1-val2))
    print alpha, np.exp(val1-val2)
    uni = np.random.rand(1)
    if alpha > uni[0]:
        theta = theta_cand
        param_box[i] = theta
        Param_BOX.append(theta)
    elif alpha <= uni[0]:
        param_box[i] = theta
        Param_BOX.append(theta)

param_box = param_box[burn_in:,:]
print param_box.shape
print "-"*50
Est = np.median(param_box,axis=0)
Var = np.var(param_box,axis=0)

for i in range(len(Est)):
    if i < 10:
        print "lam",i,Est[i],Var[i]
    else:
        print "beta",Est[i],Var[i]


# plt.plot(param_box[:,10])
# plt.show()

for i in range(param_dim):
    plt.figure(i)
    plt.plot(param_box[:,i])
#     plt.hist(param_box[:,i],20)
plt.show()

