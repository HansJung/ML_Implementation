import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats import invwishart
from collections import Counter

def array_mode(arrMat,N):
    arrElem = np.zeros(N)
    arrMat = np.transpose(arrMat)

    for i in range(N):
        arrElem[i] = Counter(arrMat[i]).most_common(1)[0][0]
    return arrElem


''' Data generation '''
np.random.seed(123456)

K = 3
dim = 2
N = 100
alpha = [0.5]*K
pi_true = np.squeeze(np.random.dirichlet(alpha,1))

mu_true = np.random.multivariate_normal(np.zeros(dim),np.eye(dim)*200,K)
RandBox = np.random.rand(K) + np.random.randint(5,size=K)
print RandBox
Cov_true = [np.eye(dim)*rnd for rnd in RandBox] # common

Z_true = np.random.choice(K, N, p=pi_true)
X = list()

for idx in range(N):
    Zidx = Z_true[idx]
    X.append(np.random.multivariate_normal(mu_true[Zidx],Cov_true[Zidx]))
X = np.array(X)


''' MCMC '''
# Param to be estimate & Initial setting
nu0 = 20.0
S0 = np.eye(dim)
mu_est = np.random.multivariate_normal(np.zeros(dim),np.eye(dim)*100,K)
# Cov_est = invwishart.rvs(nu0, S0,size=K)
Cov_est = np.array([np.eye(dim),np.eye(dim),np.eye(dim)])
alpha_est = np.ones(K)
pi_est = np.squeeze(np.random.dirichlet(alpha_est,1))


# Initial setting
Z_est = np.random.randint(K,size=N)
Nk = np.zeros(K)
for k in range(K):
    Nk[k] = len(Z_est[Z_est==k])


Z_est_trace = list()
alpha_trace = list()
mu_trace = list()
Cov_trace = list()
pi_trace = list()


for HansIter in range(100):
    # 1. Categorical update
    for i in range(N):
        Temp_llk = list()
        for k in range(K):
            try:
                Temp_llk.append(pi_est[k]*multivariate_normal.pdf(X[i],mu_est[k],Cov_est[k]))
            except:
                # print "hmm..."
                Temp_llk.append(pi_est[k]*multivariate_normal.pdf(X[i],mu_est[k],np.eye(dim)))

        # Z_est[i] = np.argmax(Temp_llk)
        Temp_llk = np.array(Temp_llk) / np.sum(Temp_llk)
        Z_est[i] = np.random.choice(K,1,p=Temp_llk)
    Z_est_trace.append(Z_est)

    # 1.5 Nk update
    for k in range(K):
        Nk[k] = len(Z_est[Z_est==k])

    # 2. pi update
    for k in range(K):
        alpha_est[k] += Nk[k]
    alpha_trace.append(alpha_est)
    pi_est = np.squeeze(np.random.dirichlet(alpha_est,1))
    # pi_est = np.sort(pi_est)
    pi_trace.append(pi_est)

    # 3. mu update
    V = list()
    m = list()
    m0 = np.zeros(dim)
    V0 = np.eye(dim)
    for k in range(K):
        Vk = np.linalg.inv( np.linalg.inv(V0) + Nk[k]*np.linalg.inv(Cov_est[k]) )
        Xbark = np.sum( X[Z_est==k] , axis=0  ) / Nk[k]
        Var1 = np.dot(Vk,  np.linalg.inv(Cov_est[k])  )
        Var2 = (Nk[k] * Xbark)
        mk = np.dot(Var1,Var2)

        V.append(Vk)
        m.append(mk)

    for k in range(K):
        mu_est[k] = np.random.multivariate_normal(m[k],V[k])
    # mu_est = np.sort(mu_est) # Label switching term
    mu_trace.append(mu_est)

    # # 4. Cov Update
    # nu = list()
    # S = list()
    # for k in range(K):
    #     nuk = nu0 + Nk[k]
    #     Sk = S0
    #     TempX = X[Z_est==k]
    #     for i in range(len(TempX)):
    #         Sk += np.dot(TempX[i] - mu_est[k],np.transpose(TempX[i] - mu_est[k]))
    #         # print Sk.shape
    #     #     Sk = S0 + np.sum( np.dot((X[Z_est == k] - m[k]),np.transpose(X[Z_est==k] - m[k])) )
    #     nu.append(nuk)
    #     S.append(Sk)
    #
    # for k in range(K):
    #     Cov_est[k] = invwishart.rvs(nu[k],S[k],size=1)
    # Cov_trace.append(Cov_est)

    print HansIter

''' Validation '''
# Burn
Burnin = 50
mu_trace = mu_trace[Burnin:]
mu_est = np.mean(mu_trace,axis=0)

Cov_trace = Cov_trace[Burnin:]
Cov_est = np.mean(Cov_trace,axis=0)


pi_trace = pi_trace[Burnin:]
pi_est = np.mean(pi_trace,axis=0)

for a,b in zip(mu_est,mu_true):
    print a,b
print " "
print pi_est, pi_true

print " "
Z_est_trace = Z_est_trace[Burnin:]
Z_est_trace = np.array(Z_est_trace)

Z_est = array_mode(Z_est_trace,N)

# for k in range(K):
#     print Cov_est[k]


color = ['blue','red','green']
plt.figure()
for k in range(K):
    plt.scatter(X[Z_est == k,0],X[Z_est == k,1],c=color[k])
plt.show()