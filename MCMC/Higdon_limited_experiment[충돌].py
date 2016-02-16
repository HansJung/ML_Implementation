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

def Kernel_z(K_z, Sig_y):
    dim_z = len(K_z)
    dim_y = len(Sig_y)

    


# default setting
np.random.seed(123)

# Generate y
x_true = np.random.uniform(1,10,5)
theta_true = 3.698
y = true_process(x_true,theta_true)
y += np.random.normal(0,1)

# Generate eta
theta_trial = np.random.uniform(3,4,20)
x_trial = np.random.uniform(1,10,20)
theta_trial = np.reshape(theta_trial, (len(theta_trial),1))
x_trial = np.reshape(x_trial, (len(x_trial),1))
v = np.concatenate((x_trial,theta_trial),axis=1)
eta = np.zeros((20,1))
for i in range(len(eta)):
    eta[i] = true_process(v[i][0], v[i][1])





















