import numpy as np
import scipy as sp
import math
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn import mixture
import itertools
from scipy.stats import invwishart

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
    return np.array(table_assignments,dtype='int'), next_table

def Base_distribution_sampling(sample_mean, dim): # NIW (prior)
    kappa_0 = 0.1
    Lam_0 = np.eye(dim)
    nu_0 = dim + 2

    Cov_sampled = invwishart.rvs(df=nu_0,scale=Lam_0) / kappa_0
    mu_sampled = np.random.multivariate_normal(sample_mean, Cov_sampled)

    return mu_sampled, Cov_sampled

def Base_distribution_posterior_sampling(X_cl, N_cl, old_mu, old_cov, dim=2):
    kappa_0 = 0.1
    nu_0 = dim + 2
    m0 = old_mu
    x_bar = np.mean(X_cl,axis=0)
    kappa_n = kappa_0 + N_cl
    nu_n = nu_0 + N_cl

    if len(X_cl) != N_cl:
        print "something wrong"
        return None

    if N_cl == 1:
        x = X_cl[0]
        # print 'here'
        k0 = 0.01
        mN = (k0 / (k0 + N_cl))*m0 + (N_cl/(k0+N_cl)) * x_bar
        # SN = np.dot(np.reshape(x-old_mu,(dim,1)),np.reshape(x,(dim,1)).T)
        # _,SN,_ = np.linalg.svd(SN)
        # SN = np.diag(SN)
        SN = np.eye(dim)
        try:
            iSN = np.linalg.inv(SN)
        except:
            iSN = np.linalg.inv(SN + 1e-6*np.eye(dim))

        try:
            mu_new = np.random.multivariate_normal(mN, SN /(nu_n - dim + 1) )
        except:
            mu_new = np.random.multivariate_normal(mN, SN /(nu_n - dim + 1) + 1e-6*np.eye(dim) )
        cov_new = invwishart.rvs(df=nu_n, scale=iSN)
        _,cov_new,_ = np.linalg.svd(cov_new)
        cov_new = np.diag(cov_new)
        return mu_new,cov_new
    else:
        Cov_est = np.zeros((dim,dim))
        for idx in range(N_cl):
            # print X_cl[idx]
            diff = np.reshape(X_cl[idx]-x_bar,(dim,1))
            Cov_est += np.dot(diff,diff.T)
        ''' NIW posterior param '''
        mN = (kappa_0 / (kappa_0 + N_cl))*m0 + (N_cl/(kappa_0+N_cl)) * x_bar
        SN = old_cov + Cov_est
        x_bar_ = np.reshape(x_bar,(dim,1))
        SN += (kappa_0*N)/(kappa_0+N)*np.dot(x_bar_ - old_mu,x_bar_ - old_mu.T)
        _,SN,_ = np.linalg.svd(SN)
        SN = np.diag(SN)

        try:
            iSN = np.linalg.inv(SN)
        except:
            iSN = np.linalg.inv(SN + 1e-6*np.eye(dim))
        try:
            mu_new = np.random.multivariate_normal(mN, SN /(nu_n - dim + 1) )
        except:
            mu_new = np.random.multivariate_normal(mN, SN /(nu_n - dim + 1) + 1e-6*np.eye(dim) )
        try:
            cov_new = invwishart.rvs(df=nu_n, scale=iSN)
        except:
            cov_new = np.eye(dim)
            # print iSN, np.linalg.det(iSN)
        _,cov_new,_ = np.linalg.svd(cov_new)
        cov_new = np.diag(cov_new)
        return mu_new,cov_new




def Base_disctribution_pdf(sample_mean, mu, cov,dim):
    mu_0 = sample_mean
    kappa_0 = 0.1
    Lam_0 = np.eye(dim)
    nu_0 = dim + 2
    try:
        icov = np.linalg.inv(cov)
    except:
        icov = np.linalg.inv(cov + 1e-6*np.eye(dim))

    val1 = np.log(np.linalg.det(cov) ** (-(nu_0 + dim)/2. + 1))
    val2 = -np.trace(Lam_0 * icov ) / 2.
    val3 = -kappa_0 * np.dot( np.transpose(mu-mu_0), np.dot( icov,(mu-mu_0)))/2.0
    return val1+val2+val3






if __name__ == '__main__':

    ''' Generate data '''
    K = 3
    dim = 2
    N = 100
    X, Z = Make_GMM_data(K,dim, N)
    sample_mean = np.mean(X,axis=0)

    print [len(Z[Z==idx]) for idx in range(K)]
    # color_iter = ['r', 'g', 'b', 'c', 'm']
    #
    #
    # plt.figure(0)
    # plt.title("True")
    # for k in range(K):
    #     plt.scatter(X[Z==k][:,0],X[Z==k][:,1],c=color_iter[k])


    ''' DPMM '''
    # Initial setting
    alpha = 100.
    Z_est,K_est = chinese_restaurant_process(N,alpha=alpha)
    # print K_est
    Cl = dict()

    # Initial param setting
    for k in range(K_est):
        Cl[k] = X[Z_est == k]
    mu_dict = dict()
    cov_dict = dict()
    for k in range(K_est):
        if len(Cl[k]) > 1:
            mu_dict[k] = np.mean(Cl[k],axis=0)
            cov_k = np.cov(Cl[k].T)
            _,cov_k,_ = np.linalg.svd(cov_k)
            cov_k = np.diag(cov_k)
            cov_dict[k] = cov_k
        elif len(Cl[k]) == 1:
            xi = Cl[k][0]
            mu_dict[k] = xi
            cov_k = np.dot(np.reshape(xi,(dim,1)),np.reshape(xi,(dim,1)).T)
            _,cov_k,_ = np.linalg.svd(cov_k)
            cov_k = np.diag(cov_k)
            cov_dict[k] = cov_k
        elif len(Cl[k]) == 0:
            mu_dict[k] = False
            cov_dict[k] = False

    for k in range(K_est):
        Cl[k] = X[Z_est == k]


    # Iteration
    iterNum = 10

    for iterIdx in range(iterNum):
        print iterIdx, K_est
        per_idx = np.random.permutation(range(N))
        for idx in per_idx:
            xi = X[idx]
            # Old clustering
            Nk = dict()
            Prob_K = dict()


            # Initial clustering
            for k in set(Z_est):
                if xi in Cl[k]:
                    Cl_i = k
                    row_idx = np.squeeze(np.argwhere(Cl[k]==xi))[0][0]
                    # mu_i = np.mean(Cl[k],axis=0)
                    # if len(Cl[k]) > 1:
                    #     cov_i = np.cov(Cl[k].T)
                    #     _,cov_i,_ = np.linalg.svd(cov_i)
                    #     cov_i = np.diag(cov_i)
                    # else:
                    #     cov_i = np.dot(np.reshape(xi,(dim,1)),np.reshape(xi,(dim,1)).T)
                    #     _,cov_i,_ = np.linalg.svd(cov_i)
                    #     cov_i = np.diag(cov_i)
                    Cl[k]= np.delete(Cl[k],row_idx,axis=0)
                Nk[k] = Cl[k].shape[0]
                Prob_K[k] = Nk[k] / (N-1+alpha)

            # new room?
            if np.random.rand() < alpha / (N-1+alpha):
                # Candidate!
                ## Compute accept rate
                ### Sample new param
                mu_sampled, cov_sampled = Base_distribution_sampling(dim=2, sample_mean=sample_mean)
                try:
                    LL_old = multivariate_normal.logpdf(xi,mean=mu_dict[Cl_i],cov=cov_dict[Cl_i])
                    LL_new = multivariate_normal.logpdf(xi,mean=mu_sampled, cov=cov_sampled)
                except:
                    # print cov_dict[Cl_i], len(Cl[Cl_i])
                    # print cov_dict[Cl_i], np.linalg.det(cov_dict[Cl_i])
                    LL_old = multivariate_normal.pdf(xi,mean=mu_dict[Cl_i],cov=cov_dict[Cl_i]+1e-6*np.eye(dim))
                    LL_new = multivariate_normal.pdf(xi,mean=mu_sampled, cov=cov_sampled)

                ### Compute accept rate
                LPr_old = Base_disctribution_pdf(sample_mean,mu_dict[Cl_i], cov_dict[Cl_i], dim)
                LPr_new = Base_disctribution_pdf(sample_mean,mu_sampled, cov_sampled, dim)

                L_joint_old = LL_old + LPr_old
                L_joint_new = LL_new + LPr_new
                accept_rate = np.exp(min(0, L_joint_new - L_joint_old ))

                ## Determine to accept
                if np.random.rand() < accept_rate:
                    # accept!
                    Z_est[idx] = K_est + 1
                    Cl[K_est+1] = np.array([xi])
                    mu_dict[K_est+1] = mu_sampled
                    cov_dict[K_est+1] = cov_sampled
                else:
                    Cl[Cl_i] = np.vstack((Cl[Cl_i],xi))
                    pass
            else:
                Temp_prob = np.array(Prob_K.values())/np.sum(Prob_K.values())
                Z_est[idx] = np.random.choice(list(set(Z_est)),1,p=Temp_prob)
                Cl[Z_est[idx]] = np.vstack((Cl[Z_est[idx]],xi))


        # Index update
        # Cl update
        # sumval = 0
        # for idx,key in enumerate(sorted(Cl)):
        #     sumval += len(Cl[key])
        # print "totalval", sumval

        # for idx, key in enumerate(sorted(Cl)):
        #     print key, Cl[key], len(Cl[key])

        # Room close
        New_Cl = dict()
        for k in set(Z_est):
            New_Cl[k] = X[Z_est==k]
        Cl = New_Cl
        del New_Cl

        for k in set(Z_est):
            if len(Cl[k]) > 1:
                mu_dict[k] = np.mean(Cl[k],axis=0)
                cov_k = np.cov(Cl[k].T)
                _,cov_k,_ = np.linalg.svd(cov_k)
                cov_k = np.diag(cov_k)
                cov_dict[k] = cov_k
            elif len(Cl[k]) == 1:
                xi = Cl[k][0]
                mu_dict[k] = xi
                cov_k = np.dot(np.reshape(xi,(dim,1)),np.reshape(xi,(dim,1)).T)
                _,cov_k,_ = np.linalg.svd(cov_k)
                cov_k = np.diag(cov_k)
                cov_dict[k] = cov_k
            elif len(Cl[k]) == 0:
                mu_dict[k] = False
                cov_dict[k] = False

        # for idx, key in enumerate(sorted(Cl)):
        #     if len(Cl[key]) == 0:
        #         del Cl[key]
        #         del mu_dict[key]
        #         del cov_dict[key]
        #
        # # New room assigned
        # New_Cl = dict()
        # New_mu_dict = dict()
        # New_cov_dict = dict()
        # for idx, key in enumerate(sorted(Cl)):
        #     New_Cl[idx] = Cl[key]
        #     New_mu_dict[idx] = mu_dict[key]
        #     New_cov_dict[idx] = cov_dict[key]
        # Cl = New_Cl
        # mu_dict = New_mu_dict
        # cov_dict = New_cov_dict
        #
        # del New_Cl
        # del New_mu_dict
        # del New_cov_dict

        ## Param update
        for idx,key in enumerate(sorted(Cl)):
            mu_dict[key], cov_dict[key] = Base_distribution_posterior_sampling(Cl[key],len(Cl[key]),mu_dict[key],cov_dict[key])
        print iterIdx, mu_dict.keys()
        print iterIdx, Cl.keys()
        K_est = len(Cl.keys())






    # alpha = 10.
    # Trun_K = 100
    #
    # crp = chinese_restaurant_process(Trun_K,alpha)
    # print crp









