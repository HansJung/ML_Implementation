__author__ = 'jeong-yonghan'
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def Compute_Sigma(x1,x2,sigma_f, l):
    dim_x1 = len(x1)
    dim_x2 = len(x2)

    x1 = np.array(x1)
    x2 = np.array(x2)
    Sigma = np.zeros((dim_x1,dim_x2))
    sigma_f = float(sigma_f)
    l = float(l)


    for i in range(0,dim_x1):
        for j in range(0,dim_x2):
            Sigma[i][j] = (sigma_f**2) * np.exp(  -0.5/(l**2) * ( x1[i] - x2[j] ) ** 2   )
    return Sigma


def Compute_Sigma_Y(x1,x2,sigma_f, l, sigma_n):
    dim = len(x1)
    x1 = np.array(x1)
    x2 = np.array(x2)
    Sigma = np.zeros((dim,dim))
    sigma_f = float(sigma_f)
    sigma_n = float(sigma_n)
    l = float(l)

    for i in range(0,dim):
        for j in range(0,dim):
            if x1[i] == x2[j]:
                Sigma[i][j] = (sigma_f**2) * np.exp(  -0.5/(l**2) * ( x1[i] - x2[j] ) ** 2   ) + (sigma_n ** 2)
            else:
                Sigma[i][j] = (sigma_f**2) * np.exp(  -0.5/(l**2) * ( x1[i] - x2[j] ) ** 2   )
    return Sigma

def Objective_Function(X, x1,x2, y):
    sigma_f = X[0]
    l = X[1]
    sigman_n = X[2]
    Ky = Compute_Sigma_Y(x1,x2,sigma_f,l,sigma_n)
    Inv_Ky = np.linalg.inv(Ky)
    N = len(y)

    Logval = -1*(-0.5 * np.dot(np.dot(np.transpose(y),Inv_Ky),y) - 0.5 * np.log(np.linalg.det(Ky)) - 0.5*N*np.log(2*np.pi))

    return Logval

# np.random.seed(1)
## Observations
x_obs = [ -1.5, -1.0, -0.75, -0.4, -0.25, 0.0   ]
y_obs = [ -1.7, -1.3, -0.4, 0.2, 0.4, 0.8  ]
X_domain = list()
for idx in range(len(x_obs)-1):
    x_domain = np.linspace(x_obs[idx], x_obs[idx+1],200)
    X_domain.append(x_domain)


x_obs = np.array(x_obs)
x_obs = np.reshape(x_obs,(len(x_obs),1))

y_obs = np.array(y_obs)
y_obs = np.reshape(y_obs,(len(y_obs),1))

## Parameter Estimation
sigma_f = 1
l = 0.1
sigma_n = 0.1
X0 = np.array([sigma_f,l,sigma_n])

res = minimize(Objective_Function, X0, args=(x_obs,x_obs,y_obs,),method='Nelder-Mead', options={'xtol': 1e-8,'disp':True})

sigma_f = res.x[0]
l = res.x[1]
sigma_n = res.x[2]

print res.x




K_yobs = Compute_Sigma_Y(x_obs,x_obs,sigma_f = sigma_f, l = l, sigma_n = sigma_n)
Inv_Ky = np.linalg.inv(K_yobs)
Y_new = list()
X_new = list()
Var_new = list()
for idx in range(len(X_domain)):
    x_domain = X_domain[idx]
    for xpt in x_domain:
        xpt = [xpt]
        K_star_T = Compute_Sigma(xpt,x_obs, sigma_f= sigma_f, l = l)
        K_star = Compute_Sigma(x_obs,xpt, sigma_f= sigma_f, l = l)
        K_star_star = Compute_Sigma(xpt, xpt, sigma_f = sigma_f, l = l )

        # Expectation of poseterior
        f_new = np.dot(np.dot(K_star_T,Inv_Ky),y_obs)

        # Cov
        CovMat = K_star_star - np.dot(np.dot(K_star_T, Inv_Ky),K_star)
        CovVar = np.squeeze(CovMat)

        f_new = np.squeeze(f_new)
        Y_new.append(f_new)
        X_new.append(xpt[0])
        Var_new.append(CovVar)



Y_interval_plus = np.array(Y_new) + (1.96 / np.sqrt(len(x_obs)) * np.sqrt(np.array(Var_new)))
Y_interval_Minus = np.array(Y_new) - (1.96 / np.sqrt(len(x_obs)) * np.sqrt(np.array(Var_new)))
plt.plot(x_obs,y_obs,'bo')
plt.plot(X_new,Y_new)
# plt.plot(X_new,Y_interval_Minus,'b--')
# plt.plot(X_new,Y_interval_plus,'b--')
plt.fill_between(X_new,Y_interval_Minus, Y_interval_plus, color='gray', alpha = 0.3)
plt.show()

# print Y_new