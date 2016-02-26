import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

from collections import Counter

def array_mode(arrMat,N):
    arrElem = np.zeros(N)
    arrMat = np.transpose(arrMat)

    for i in range(N):
        arrElem[i] = Counter(arrMat[i]).most_common(1)[0][0]
    return arrElem

def stick_breaking(alpha, k):
    betas = np.random.beta(1, alpha, k)
    remaining_pieces = np.append(1, np.cumprod(1 - betas[:-1]))
    p = betas * remaining_pieces
    return p/p.sum()

def dirichlet_process(p, n, P0=np.random.randn):
    # theta = P0(len(p))
    theta = np.random.multivariate_normal(np.zeros(2),len(p))
    return np.random.choice(theta, size=n, p=p)

def chinese_restaurant_process(n, alpha):
    if n < 1:
        return None
    table_assignments = np.empty(n) # initial empty table
    next_table = 0 # Currently the existed table labeled only zero

    for c in range(n):
        if np.random.random() < (1. * alpha / (alpha + c)):
            # Sit at new table
            table_assignments[c] = next_table
            next_table += 1
        else:
            # Calculate selection probabilities as function of population
            probs = [(table_assignments[:c]==i).sum()/float(c) for i in range(next_table)]
            # Randomly assign to existing table
            table_assignments[c] = np.random.choice(range(next_table), p=probs)
    return np.array(table_assignments,dtype='int')

def plot_crp(table_nums, ax=None):
    x = list(range(int(table_nums.max()) + 1))
    f = [(table_nums==i).sum() for i in set(table_nums)]
    if ax is None: ax = plt
    ax.bar(x, f)




############################################################################################
if __name__ == '__main__':
    ''' Data generation '''
    np.random.seed(123)
    Latent_K = 3
    dim = 2
    N = 1000
    alpha = [0.5]*Latent_K
    pi_true = np.squeeze(np.random.dirichlet(alpha,1))

    mu_true = np.random.multivariate_normal(np.zeros(dim),np.eye(dim)*200,Latent_K)
    Cov_true = [np.eye(dim),np.eye(dim)*2.0,np.eye(dim)/2.0] # common

    Z_true = np.random.choice(Latent_K, N, p=pi_true)
    X = list()

    for idx in range(N):
        Zidx = Z_true[idx]
        X.append(np.random.multivariate_normal(mu_true[Zidx],Cov_true[Zidx]))
    X = np.array(X)






    ''' Initial DP setting '''
    # set the stick breaking alpha and truncK
    a = 1.
    trun_K = 100
    base_COV = 100. * np.eye(dim)
    base_mean = np.zeros(dim)
    each_COV = np.eye(dim)

    theta = np.random.multivariate_normal(base_mean,base_COV,trun_K)
    Z_est = chinese_restaurant_process(N,a)
    theta_each = theta[Z_est]


    # Count_Z_est = [(Z_est==i).sum() for i in range(max(Z_est))]
    # Prob_cluster = [(Z_est==i).sum() / float(Z_est.sum()) for i in range(len(Z_est))]

    # for x,y in zip(Z_true,Z_est):
    #     print x,y



