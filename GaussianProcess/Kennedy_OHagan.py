__author__ = 'jeong-yonghan'

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from Gaussian_Process import Gaussian_Process

def True_Process(x):
    return 0.5 * np.sin(x) + 0.1 * np.cos(x) ** 2 + 0.33 * np.log(np.abs(x)+1)

def Computer_code(x_star):
    return 0.5 * np.sin(x_star) + 0.5

def Compute_Sigma(x1,x2, Param):
    dim_x1 = len(x1)
    dim_x2 = len(x2)
    sigma_f = Param[0]
    l = Param[1]
    # sigma_n = Param[2]

    x1 = np.array(x1)
    x2 = np.array(x2)
    Sigma = np.zeros((dim_x1,dim_x2))

    for i in range(0,dim_x1):
        for j in range(0,dim_x2):
            Sigma[i][j] = (sigma_f**2) * np.exp(  -0.5/(l**2) * ( x1[i] - x2[j] ) ** 2   )
            if i == j:
                Sigma[i][j] += 1e-3

    return Sigma


# np.random.seed(1)
# np.random.seed(123)

# Initial parameter
sigma_f = 1
l = 1
sigma_n = 0.01
rho = 1.2

Param_cpt = [sigma_f, l, sigma_n]
Param_disc = [sigma_f, l, sigma_n]



x = np.random.uniform(0,10,5) # D2
x_star = np.random.uniform(min(x),max(x),20) # D1

z = True_Process(x) + np.random.normal(0,sigma_n,len(x))
y = Computer_code(x_star) + np.random.normal(0,sigma_n,len(x_star))


x_new_set = np.linspace(min(x),max(x),1000)
zeta = True_Process(x_new_set)

# V1_D1
V1_D1 = Compute_Sigma(x_star,x_star,Param_cpt) # Covariance matrix of computer output
V1_D2 = Compute_Sigma(x,x,Param_cpt) # Covariance matrix of true process
V2_D2 = Compute_Sigma(x,x,Param_disc)
C1 = Compute_Sigma(x_star,x,Param_cpt)
C1_T = np.transpose(C1)
Lambda_In = sigma_n * np.eye(len(x))

Test1 = np.concatenate( (V1_D1, rho * C1_T) )
Test2 = np.concatenate(  (     rho * C1, rho**2 * V1_D2 + V2_D2   )  )

V_d = np.concatenate(  (Test1,Test2),axis=1 )

d = np.concatenate( (y,z) )
d = np.reshape( d, (len(d),1) )


### Suppose x_new
z_new_set = list()

for x_new in x_new_set:
    x_new = np.array([x_new])
    t_x = np.transpose(np.concatenate(  (rho * Compute_Sigma(x_new, x_star, Param_cpt ),  rho**2 * Compute_Sigma(x_new, x, Param_cpt) + Compute_Sigma(x_new, x, Param_disc )      ), axis=1   ))

    ## Forecasting

    z_new_exp = np.squeeze(np.dot(np.dot(np.transpose(t_x), np.linalg.inv(V_d) ), d))
    print np.linalg.det(V_d)
    z_new_set.append(z_new_exp)




## GP regression for the computer model output
GP_cpt = Gaussian_Process(Param_cpt,x_star,y)
GP_true_apx = Gaussian_Process(Param_cpt,x,z)
x_cpt,y_cpt,var_cpt = GP_cpt.GP_interpolation(x_new_set)
x_obs,y_true,var_obs = GP_true_apx.GP_interpolation(x_new_set)







plt.figure()
plt.plot( x,z,'bo', label="true obs" )
plt.plot( x_star, y, 'ro', label = "cpt obs" )
plt.plot(x_new_set, zeta,'b', label="true process") # True process
plt.plot(x_new_set, z_new_set,'g',label="gp apx true")
# plt.plot(x_new_set,y_cpt,'r',label="gp_apx_cpt")
plt.plot(x_new_set, y_true,'black',label="gp_with_true")
plt.legend(loc="best")

plt.show()









