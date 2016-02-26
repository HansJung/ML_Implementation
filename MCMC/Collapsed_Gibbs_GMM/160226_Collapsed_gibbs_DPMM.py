import numpy as np
import scipy as sp
import math
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

def Make_GMM_data(K,dim,N):
    # np.random.seed(123)

    alpha = [1.]*K
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
    return X, Z_true

def stick_breaking(alpha, k):
    betas = np.random.beta(1, alpha, k)
    remaining_pieces = np.append(1, np.cumprod(1 - betas[:-1]))
    p = betas * remaining_pieces
    return p/p.sum()

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

if __name__ == '__main__':

    ''' Generate data '''
    K = 3
    dim = 2
    N = 100
    X, Z = Make_GMM_data(K,dim, N)

    print [len(Z[Z==idx]) for idx in range(K)]
    color = ['blue','red','yellow','cyan','magenta']

    plt.figure(0)
    plt.title("True")
    for k in range(K):
        plt.scatter(X[Z==k][:,0],X[Z==k][:,1],c=color[k])
