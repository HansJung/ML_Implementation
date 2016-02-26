import numpy as np
import scipy as sp
import math
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

def Make_GMM_data(K,dim,N):


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


def Post_predictive(x, nu_0,mu_0,X_cl, N_cl, dim):
    m0 = np.zeros(dim)
    x_bar = mu_0

    if len(X_cl) != N_cl:
        print "something wrong"
        return None

    if N_cl < 1:
        if np.random.rand() < 0.1:
            return 0.
        else:
            return np.random.rand()

    if N_cl == 1:
        k0 = 0.01
        mN = (k0 / (k0 + N_cl))*m0 + (N_cl/(k0+N_cl)) * x_bar
        SN = np.dot(np.reshape(x,(dim,1)),np.reshape(x,(dim,1)).T)
        # _,SN,_ = np.linalg.svd(SN)
        # SN = np.diag(SN)
        try:
            return multivariate_normal.pdf(x,mean=mN, cov=SN)
        except:
            # print("regularized...")
            return multivariate_normal.pdf(x,mean=mN, cov=SN+1e-6*np.eye(dim))

    else:
        Cov_est = np.zeros((dim,dim))
        for idx in range(len(X_cl)):
            diff = np.reshape(X_cl[idx]-mu_0,(dim,1))
            Cov_est += np.dot(diff,diff.T)

        S0  = Cov_est / float(len(X_cl))
        k0 = 0.01

        ''' NIW posterior param '''
        mN = (k0 / (k0 + N_cl))*m0 + (N_cl/(k0+N_cl)) * x_bar
        kN = k0 + N_cl
        nu_N = nu_0 + N_cl


        SN = S0 + Cov_est
        x_bar_ = np.reshape(x_bar,(dim,1))
        SN += (k0*N)/(k0+N)*np.dot(x_bar_,x_bar_.T)
        SN *= (kN+1)/(kN*(nu_N - dim + 1))
        _,SN,_ = np.linalg.svd(SN)
        SN = np.diag(SN)
        try:
            return multivariate_normal.pdf(x, mean=mN,cov=SN)
        except:
            # print("regularized...")
            return multivariate_normal.pdf(x,mean=mN, cov=SN+1e-6*np.eye(dim))





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


    ''' Initializing iteration '''
    Z_est = np.random.randint(K,size=N)
    Nk = np.zeros(K)
    Cl = dict()
    # np.random.seed(123456)

    iterNum = 100
    for iterIdx in range(iterNum):
        print iterIdx
        per_idx = np.random.permutation(range(N))

        for k in range(K):
            Cl[k] = X[Z_est == k]

        # for each data
        for idx in per_idx:
            xi = X[idx]
            pi_est = dict()
            # try:
            for k in range(K):
                if xi in Cl[k]:
                    row_idx = np.squeeze(np.argwhere(Cl[k]==xi))[0][0]
                    Cl[k]= np.delete(Cl[k],row_idx,axis=0)
                Nki = len(Cl[k])
                pi_est[k] = (Nki + 1/float(K) )*Post_predictive(x=xi,nu_0=dim+2,mu_0=np.mean(Cl[k],axis=0),X_cl=Cl[k],N_cl=len(Cl[k]),dim=2)
            # except:
            #
            #     for k in range(K):
            #         print len(Cl[k])
            #     # import matplotlib.pyplot as plt
            #     # color = ['blue','red','green']
            #     plt.figure()
            #     plt.title("est")
            #     for k in range(K):
            #         plt.scatter(X[Z_est==k][:,0],X[Z_est==k][:,1],c = color)
            #     plt.show()


            TempLLK = pi_est.values() / np.sum(pi_est.values())
            Z_est[idx] = np.random.choice(K,1,p=TempLLK)
            Cl[Z_est[idx]] = np.vstack((Cl[Z_est[idx]],xi))

    # print

    plt.figure()
    plt.title("iterate!")
    for k in range(K):
        plt.scatter(X[Z_est == k,0],X[Z_est == k,1],c=color[k])
    plt.show()




    # Plot_YN = True
    Plot_YN = False
    if Plot_YN:
        import matplotlib.pyplot as plt
        plt.plot(X[:,0],X[:,1],'bo')
        plt.show()
