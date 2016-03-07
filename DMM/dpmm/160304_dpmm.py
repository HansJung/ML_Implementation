import numpy as np
from scipy.stats import invwishart
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


from scipy.stats import multivariate_normal
from collections import Counter


def Make_GMM_data(K,dim,N):
    # np.random.seed(100)

    alpha = [1.]*K
    pi_true = np.squeeze(np.random.dirichlet(alpha,1))

    mu_true = np.random.multivariate_normal(np.zeros(dim),np.eye(dim)*200,K)
    RandBox = [1.]*K
    # RandBox = np.random.rand(K) + np.random.randint(5,size=K)
    Cov_true = [np.eye(dim)*rnd for rnd in RandBox] # common

    Z_true = np.random.choice(K, N, p=pi_true)
    X = list()

    for idx in range(N):
        Zidx = Z_true[idx]
        X.append(np.random.multivariate_normal(mu_true[Zidx],Cov_true[Zidx]))
    X = np.array(X)
    return X, Z_true

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






############################################################################################
if __name__ == '__main__':
    ''' Data generation '''
    K = 4
    N = 500
    dim = 2
    X,Z = Make_GMM_data(K,dim,N)

    plt.figure()
    plt.title("True")
    color = ['blue','red','yellow','cyan','magenta']
    for k in range(5):
        plt.scatter(X[Z == k,0],X[Z == k,1],c=color[k])



    ''' Initialization '''
    # Initial giving class assignment by CRP
    alpha = 1.
    Table_assignment, K_ = chinese_restaurant_process(N,alpha)
    Nic = [(Table_assignment == i).sum() for i in range(K_)]

    Update_dict = dict()
    Update_dict_cluster = dict()

    for idx in range(len(X)):
        Update_dict[idx] = dict()
        Update_dict[idx]['data'] = X[idx]
        Update_dict[idx]['cls_idx'] = Table_assignment[idx]
        Update_dict[idx]['cls_size'] = (Table_assignment==Table_assignment[idx]).sum()
        Update_dict[idx]['param'] = np.mean(X[Table_assignment==Table_assignment[idx]],axis=0)

    for idx in range(K_):
        Update_dict_cluster[idx] = dict()
        Update_dict_cluster[idx]['cls_size'] = (Table_assignment==idx).sum()
        Update_dict_cluster[idx]['param'] = np.mean(X[Table_assignment == idx], axis=0)


    # for idx,key in enumerate(sorted(Update_dict)):
    #     print key, Update_dict[key]


    MCMC_iterNum = 300

    Total_iter = 20
    for total_iter in range(Total_iter):
        print "Proceeding...",total_iter+1,"/",Total_iter
        Permutation= np.random.permutation(range(N))

        for idx in Permutation:
            Table_num = len(Update_dict_cluster)
            # print Table_num+1, len(Update_dict_cluster)+1
            yi = Update_dict[idx]['data']
            cls_idx = Update_dict[idx]['cls_idx']
            cls_size = Update_dict_cluster[cls_idx]['cls_size']
            theta_i = Update_dict_cluster[cls_idx]['param']

            p = list()
            for cls_iteridx,key in enumerate(sorted(Update_dict_cluster)):
                if cls_iteridx != cls_idx:
                    Nci = Update_dict_cluster[key]['cls_size']
                else:
                    Nci = Update_dict_cluster[key]['cls_size'] - 1
                p.append(Nci)

            p.append(alpha)
            p /= np.sum(p)
            theta_list = list()
            old_theta = theta_i
            old_cls_idx = cls_idx

            for mcmc_iter in range(MCMC_iterNum):
                Hans_choice = np.random.choice(Table_num+1,p=p)
                if Hans_choice != Table_num: # existed clusters
                    theta_star = Update_dict_cluster[Hans_choice]['param']
                else:
                    theta_star = np.random.multivariate_normal(np.mean(X,axis=0),np.cov(X.T))
                accept_rate = min(1., multivariate_normal.pdf(yi,mean=theta_star,cov=np.eye(dim)) / multivariate_normal.pdf(yi,mean=old_theta,cov=np.eye(dim)))
                u = np.random.random()
                if accept_rate > u: # accept
                    old_cls_idx = Hans_choice
                    new_cls_idx = old_cls_idx
                    old_theta = theta_star
                else:
                    new_cls_idx = old_cls_idx
                    pass
            Table_assignment[idx] = new_cls_idx
            new_theta = old_theta

            # new room open!
            if old_cls_idx == Table_num:
                Update_dict_cluster[new_cls_idx] = dict()
                Update_dict_cluster[new_cls_idx]['cls_size'] = 0
                Update_dict_cluster[new_cls_idx]['param'] = yi
                new_theta = yi
                Table_num = max(Table_assignment) + 1 # new table num

            # new cluster and param update
            Update_dict_cluster[cls_idx]['cls_size'] -= 1
            Update_dict_cluster[new_cls_idx]['cls_size'] += 1
            Update_dict_cluster[new_cls_idx]['param'] = new_theta
            Update_dict[idx]['cls_idx'] = new_cls_idx

        for idx in range(len(X)):
            cls_idx = Update_dict[idx]['cls_idx']
            Update_dict[idx]['cls_size'] = Update_dict_cluster[cls_idx]['cls_size']
            Update_dict[idx]['param'] = Update_dict_cluster[cls_idx]['param']

    # Intensity computation
    Intensity_list = list()
    for idx in range(len(X)):
        xi = X[idx]
        sum_val = 0.
        for dict_idx, key in enumerate(sorted(Update_dict_cluster)):
            if Update_dict_cluster[key]['cls_size'] > 0:
                cls_mean = Update_dict_cluster[key]['param']
                sum_val += multivariate_normal.pdf(xi,mean=cls_mean,cov=np.eye(dim))
        Intensity_list.append(sum_val)
    Intensity_list = np.array(Intensity_list)

    # plt.figure()
    sns.jointplot(x=X[:,0],y=X[:,1],data=Intensity_list, kind="kde")




    for idx,key in enumerate(sorted(Update_dict)):
        print key, Update_dict[key]

    print "-"*100

    for idx,key in enumerate(sorted(Update_dict_cluster)):
        print key, Update_dict_cluster[key]

    plt.show()





