import numpy as np
import scipy as sp
import pymc as pm
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import multivariate_normal

# def mode(arr) :
#     m = max([arr.count(a) for a in arr])
#     return [x for x in arr if arr.count(x) == m][0] if m>1 else None


''' Data generation '''
np.random.seed(123456)

K = 3
dim = 2
N = 300
alpha = [0.5]*3
dir_pi = np.squeeze(np.random.dirichlet(alpha,1))

mu = np.random.multivariate_normal(np.zeros(dim),np.eye(dim)*200,K)
Cov = np.eye(dim) # common

Z = np.random.choice(K, N, p=dir_pi)
X = list()

for idx in range(N):
    Zidx = Z[idx]
    X.append(np.random.multivariate_normal(mu[Zidx],Cov))
X = np.array(X)

''' initial setting '''
a = [1.,1.,1.]
Z_est = np.random.randint(K,size=N)
mu_est = np.random.multivariate_normal(np.zeros(dim),np.eye(dim)*100,K)
cov_est = np.eye(dim)

for myiter in range(100):
    Count = np.array([len(Z_est[Z_est==i]) for i in range(K)])
    pi_est = np.random.dirichlet( a + Count  )

    for i in range(N):
        new_pi_zi = list()
        sum_normal = 0.0
        for k in range(K):
            sum_normal += pi_est[k]*multivariate_normal.pdf(X[i],mu_est[k],cov_est)
            new_pi_zi.append(pi_est[k]*multivariate_normal.pdf(X[i],mu_est[k],cov_est))
        new_pi_zi = np.array(new_pi_zi)
        new_pi_zi /= sum_normal
        Z_est[i] = np.random.choice(K,1,p=new_pi_zi)[0]

    xbar_k = list()
    for k in range(K):
        sum_val = 0.0
        for i in range(N):
            if Z_est[i] == k:
                sum_val += X[i]
        if len(Z_est[Z_est==k]) != 0 :
            sum_val /= len(Z_est[Z_est==k])
        else:
            sum_val = np.zeros(dim)
        xbar_k.append(sum_val)

    V0 = np.eye(dim)
    m0 = np.zeros(dim)
    Mk = list()
    Vk = list()
    for k in range(K):
        Nk = len(Z_est[Z_est==k])
        mk = np.dot(np.linalg.inv(cov_est),Nk*xbar_k[k]) + np.dot(np.linalg.inv(V0),m0)
        vk = V0 + Nk * cov_est
        Vk.append(vk)
        Mk.append(mk)
    Mk = np.array(Mk)
    Vk = np.array(Vk)

    for k in range(K):
        mu_est[k] = np.random.multivariate_normal( Mk[k] , Vk[k] )

    print myiter

print ""
print mu_est, mu
for i in range(N):
    print Z_est[i], Z[i]









# ''' Setting prior'''
# alpha_est = pm.Dirichlet('alpha',theta=[1.0,1.0,1.0])
# Z_est = pm.Container([pm.Categorical('category%i' %i, alpha_est  ) for i in range(N)])
# # mu_est = np.random.multivariate_normal(np.zeros(dim),np.eye(dim),K)
# # mu_est = pm.Container([pm.Normal('mumu%i'%i,[0.0, 0.0], tau = 1.0)] for i in range(K))
# mu_est = pm.Container([pm.MvNormal('mumumu%i'%i,[0.0,0.0],np.eye(dim)) for i in range(K) ])
# cov_est = np.eye(dim)
# Obs = pm.Container([pm.Normal('sample%i'%i, mu=mu_est[Z_est[i].value], tau = 1.0, value=X[i],observed=True   )  for i in range(N) ])
#
# # Difine model
# Hans_Model = pm.Model([Obs,Z_est,mu_est,alpha_est])
# Hans_MCMC = pm.MCMC(Hans_Model)
# # Hans_MCMC.use_step_method(pm.Metropolis,[mu_est,alpha_est])
# for i in range(N):
#     Hans_MCMC.use_step_method(pm.DiscreteMetropolis, Z_est[i])
#
# Hans_MCMC.sample(iter=3000,burn=2000)
# # print alpha_est.trace()
# print np.mean(alpha_est.trace(),axis=0), alpha
# # for i in range(K):
# #     print np.mean(mu_est[i].trace(),axis=0), mu
# MyZ = list()
# for i in range(N):
#     print mode(list(Z_est[i].trace())), Z[i]
#     MyZ.append(mode(list(Z_est[i].trace())))
#
# print " "
# for i in range(K):
#     print np.mean(mu_est[i].trace(),axis=0), mu[i]
#
#
#
# # pm.Matplot.plot(Hans_MCMC)
# # plt.show()
#
# #
# # plt.figure()
# # plt.plot(X[:,0],X[:,1],'bo')
# # plt.show()
