__author__ = 'jeong-yonghan'
import numpy as np
import matplotlib.pyplot as plt
import emcee

def true_process(x,theta):
    return 3 * x**2 + (5*np.sin(theta)**2+3)

def Kernel_eta(v,beta,lamb):
    N = len(v)
    K_eta = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            K_eta[i][j] = np.exp(-(beta*(v[i][0] - v[j][0])**2 + beta*(v[i][1] - v[i][1])**2)) / lamb
    return K_eta

def Kernel_y(x,beta,lamb):
    n = len(x)
    K_y = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            K_y[i][j] = np.exp(-beta*(x[i] - x[j])**2) / lamb
    return K_y

def Kernel_z(K_y,K_eta,x,v,beta,lamb):
    x_star = v[:,0]
    n = len(x)
    N = len(v)
    K_cov = np.zeros((n,N))
    for i in range(n):
        for j in range(N):
            K_cov[i][j] = np.exp(-beta*(x[i] - x_star[j]) ** 2)/lamb

    K_new1 = np.concatenate((K_y,K_cov),axis=1)
    K_new2 = np.concatenate((np.transpose(K_cov),K_eta),axis=1)
    K_z = np.concatenate((K_new1,K_new2),axis=0)
    return K_z




# default setting
np.random.seed(123)

# Generate y
x_true = np.random.uniform(1,10,5)
theta_true = 3.698
y = true_process(x_true,theta_true)
y += np.random.normal(0,1)
y = np.reshape(y,(len(y),1))

# Generate eta
sim_num = 10
theta_trial = np.random.uniform(3,4,sim_num)
x_trial = np.random.uniform(1,10,sim_num)
theta_trial = np.reshape(theta_trial, (len(theta_trial),1))
x_trial = np.reshape(x_trial, (len(x_trial),1))
v = np.concatenate((x_trial,theta_trial),axis=1)
eta = np.zeros((sim_num,1))
for i in range(len(eta)):
    eta[i] = true_process(v[i][0], v[i][1])

# Define z
z = np.concatenate((y,eta),axis=0)

# Define parameters to be estimated
beta = 1.0
lamb = 2.0
theta_est = 3.5
param = np.array([beta,lamb,theta_est])

# Define loglike
def loglike(param,z,x_true,v):
    beta,lamb,theta = param
    K_y = Kernel_y(x_true,beta,lamb)
    K_eta = Kernel_eta(v,beta,lamb)
    K_z = Kernel_z(K_y,K_eta,x_true,v,beta,lamb)
    K_z += 1e-3*np.eye(len(K_z))
    inv_Kz = np.linalg.inv(K_z)
    val = -0.5*np.dot(np.dot(np.transpose(z), inv_Kz),z) - 0.5*np.log(np.linalg.det(K_z))
    # val = -0.5*np.dot(np.dot(np.transpose(z), inv_Kz),z)
    if not np.isfinite(-val):
        return -np.inf
    else:
        return val

def logprior(param):
    beta,lamb,theta = param
    if beta < 0 or lamb < 0:
        return -np.inf
    else:
        # beta prior
        beta_prior = -0.5*np.log( 1.0-np.exp(-beta)  )-beta
        # lamb prior
        lamb_prior = -5.0*lamb + 4.0*np.log(lamb)
        theta_prior = -(theta-0.5)**2/(2.0* 1.0**2)
        return beta_prior + lamb_prior + theta_prior

# def logprior_lamb(param):
#     beta,lamb,theta = param
#     if lamb > 0:
#         return -5.0*lamb + 4.0*np.log(lamb)
#     else:
#         return -np.inf
#
# def logprior_beta(param):
#     beta,lamb,theta = param
#     if beta > 0:
#         return -0.5*np.log( 1.0-np.exp(-beta)  )-beta
#     elif beta < 0:
#         return -np.inf
#
# def logprior_theta(param):
#     beta,lamb,theta = param
#     return -(theta-0.5)**2/(2.0* 1.0**2)

def logprob(param, z,x_true,v):
    ll = np.squeeze(loglike(param,z,x_true,v))
    if not np.isfinite(-ll):
        return -np.inf

    # lp_beta = logprior_beta(param)
    # if not np.isfinite(-lp_beta):
    #     return -np.inf
    #
    # lp_lamb = logprior_beta(param)
    # if not np.isfinite(-lp_lamb):
    #     return -np.inf
    # lp_theta = logprior_theta(param)
    # lp = lp_beta + lp_lamb + lp_theta
    lp = logprior(param)
    return ll + lp

# # Emcee
# param_dim = 3
# MCMC_iter = 5000
# nwalkers = 10
# pos = [1e-4*np.random.randn(param_dim) for i in range(nwalkers)]
# sampler = emcee.EnsembleSampler(nwalkers=nwalkers,dim=param_dim,lnpostfn=logprob,args=[z,x_true,v])
# sampler.run_mcmc(pos,MCMC_iter)
#
# samples = sampler.chain[:, 50:, :].reshape((-1, param_dim))
#
# print np.mean(samples,axis=0)
# print np.mean(sampler.flatchain,axis=0)
#
# for i in range(param_dim):
#     plt.figure()
#     plt.hist(sampler.flatchain[:,i], 1000, color="k", histtype="step")
#     plt.title("Dimension {0:d}".format(i))
#
# plt.show()


#
# for i in range(param_dim):
#     plt.figure(i)
#     plt.plot(sampler.flatchain[:,i])
# plt.show()
#
#


#
# MCMC
param_dim = 3
MCMC_iter = 5000
burn_in = 1000
param_box = np.zeros((MCMC_iter,param_dim)) * 1.0
cov_prop = np.eye(param_dim)
for i in range(MCMC_iter):
    print i,"th iteration"
    param_cand = np.squeeze(np.random.multivariate_normal(param,cov_prop,1))
    val1 = logprob(param_cand,z,x_true,v)
    val2 = logprob(param,z,x_true,v)
    # numer = np.exp(logprob(param_cand,z,x_true,v))
    # deno = np.exp(logprob(param,z,x_true,v))
    if not np.isfinite(-val1) or not np.isfinite(-val2):
        pass

    alpha = min(1,np.exp(val1-val2))
    print alpha, np.exp(val1-val2)
    uni = np.random.rand(1)
    if alpha > uni[0]:
        param = param_cand
        param_box[i] = param
    elif alpha <= uni[0]:
        param_box[i] = param

param_box = param_box[burn_in:,:]
print "-"*50
print np.mean(param_box,axis=0)


for i in range(param_dim):
    plt.figure(i)
    # plt.plot(param_box[:,i])
    plt.hist(param_box[:,i],20)
plt.show()














