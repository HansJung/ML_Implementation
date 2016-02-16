# -*- coding: utf-8 -*-
'''
Goal : 
Author : Yonghan Jung, ISyE, KAIST 
Date : 160202
Comment 
- Gaussian Process Class

'''

''' Library '''
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
''' Function or Class '''


class Gaussian_Process:
    def __init__(self, Param, x_obs, y_obs): # Work as main function
        # Initial parameters
        Param[0] = float(Param[0])
        Param[1] = float(Param[1])
        Param[2] = float(Param[2])

        x_obs = np.array(x_obs)
        x_obs = np.reshape(x_obs,(len(x_obs),1))
        y_obs = np.array(y_obs)
        y_obs = np.reshape(y_obs,(len(y_obs),1))

        self.x_obs = x_obs
        self.y_obs = y_obs

        X_domain = np.linspace(min(x_obs),max(x_obs),1000)
        # X_domain = list()
        # for idx in range(len(x_obs)-1):
        #     x_domain = np.linspace(x_obs[idx], x_obs[idx+1],200)
        #     X_domain.append(x_domain)
        self.X_domain = X_domain


        ## Parameter estimation
        # res = minimize(self.Objective_Function, Param, args=(self.x_obs,self.x_obs,),method='Nelder-Mead', options={'xtol': 1e-8,'disp':True})

        # self.sigma_f = res.x[0]
        # self.l = res.x[1]
        # self.sigma_n = res.x[2]

        self.sigma_f = Param[0]
        self.l = Param[1]
        self.sigma_n = Param[2]

        Param = [self.sigma_f, self.l, self.sigma_n]

        ## Compute Ky
        Ky_obs = self.Compute_Sigma_Y(x_obs,x_obs, Param)
        self.Inv_Ky = np.linalg.inv(Ky_obs)



        return None

    def GP_confidence_interval(self, X,Y,Var):
        Y_interval_plus = np.array(Y) + (1.96 / np.sqrt(len(self.x_obs)) * np.sqrt(np.array(Var)))
        Y_interval_Minus = np.array(Y) - (1.96 / np.sqrt(len(self.x_obs)) * np.sqrt(np.array(Var)))

        return Y_interval_plus, Y_interval_Minus


    def GP_plot(self):
        X_new, Y_new, Var_new = self.GP_interpolation()
        Y_interval_plus = np.array(Y_new) + (1.96 / np.sqrt(len(self.x_obs)) * np.sqrt(np.array(Var_new)))
        Y_interval_Minus = np.array(Y_new) - (1.96 / np.sqrt(len(self.x_obs)) * np.sqrt(np.array(Var_new)))

        plt.plot(self.x_obs,self.y_obs,'bo')
        plt.plot(X_new,Y_new)
        plt.fill_between(X_new,Y_interval_Minus, Y_interval_plus, color='gray', alpha = 0.3)
        plt.show()


    def GP_interpolation(self,X_domain):
        Y_new = list()
        X_new = list()
        Var_new = list()
        for idx in range(len(X_domain)):
            # x_domain = self.X_domain[idx]
            # for xpt in x_domain:
            # xpt = [xpt]
            xpt = [X_domain[idx]]
            K_star_T = self.Compute_Sigma(xpt,self.x_obs)
            K_star = np.transpose(K_star_T)
            K_star_star = self.Compute_Sigma(xpt, xpt)

            # Expectation of poseterior
            f_new = np.dot(np.dot(K_star_T,self.Inv_Ky),self.y_obs)

            # Cov
            CovMat = K_star_star - np.dot(np.dot(K_star_T, self.Inv_Ky),K_star)
            CovVar = np.squeeze(CovMat)

            f_new = np.squeeze(f_new)
            Y_new.append(f_new)
            X_new.append(xpt[0])
            Var_new.append(CovVar)

        return X_new,Y_new, Var_new



    def Compute_Sigma(self, x1,x2):
        dim_x1 = len(x1)
        dim_x2 = len(x2)

        x1 = np.array(x1)
        x2 = np.array(x2)
        Sigma = np.zeros((dim_x1,dim_x2))
        # sigma_f = float(sigma_f)
        # l = float(l)

        for i in range(0,dim_x1):
            for j in range(0,dim_x2):
                Sigma[i][j] = (self.sigma_f**2) * np.exp(  -0.5/(self.l**2) * ( x1[i] - x2[j] ) ** 2   )
        return Sigma


    def Compute_Sigma_Y(self, x1,x2, Param):
        sigma_f = Param[0]
        l = Param[1]
        sigma_n = Param[2]
        dim = len(x1)
        x1 = np.array(x1)
        x2 = np.array(x2)
        Sigma = np.zeros((dim,dim))

        # sigma_f = float(sigma_f)
        # sigma_n = float(sigma_n)
        # l = float(l)

        for i in range(0,dim):
            for j in range(0,dim):
                if x1[i] == x2[j]:
                    Sigma[i][j] = (sigma_f**2) * np.exp(  -0.5/(l**2) * ( x1[i] - x2[j] ) ** 2   ) + (sigma_n ** 2)
                else:
                    Sigma[i][j] = (sigma_f**2) * np.exp(  -0.5/(l**2) * ( x1[i] - x2[j] ) ** 2   )
        return Sigma

    def Objective_Function(self, Param, x1, x2):
        sigma_f = Param[0]
        l = Param[1]
        sigman_n = Param[2]
        Ky = self.Compute_Sigma_Y(x1,x2, Param)
        Inv_Ky = np.linalg.inv(Ky)
        N = len(self.y_obs)

        Logval = -1*(-0.5 * np.dot(np.dot(np.transpose(self.y_obs),Inv_Ky),self.y_obs) - 0.5 * np.log(np.linalg.det(Ky)) - 0.5*N*np.log(2*np.pi))

        return Logval


def Real_Process(X):
    return 0.5 * np.sin(X) + 0.1 * np.cos(X) ** 2 + 0.33 * np.log(np.abs(X)+1)

def Computer_code(X):
    return 0.5 * np.sin(X) + 0.5



if __name__ == "__main__":
    sigma_f = 0.5
    l = 0.5
    sigma_n = 0.01

    Init_Param = [sigma_f, l, sigma_n]

    np.random.seed(125)

    x_real_obs = np.random.uniform(0,100,5)
    y_real_obs = Real_Process(x_real_obs) + np.random.normal(0,sigma_n,len(x_real_obs))

    x_code_obs = np.random.uniform(0,max(x_real_obs),20)
    y_code_obs = Computer_code(x_code_obs)

    X_domain = np.linspace(min(x_real_obs),max(x_real_obs),1000)
    Y_True = Real_Process(X_domain)
    Y_Code = Computer_code(X_domain)


    GP_for_real = Gaussian_Process(Init_Param, x_real_obs,y_real_obs)
    GP_for_code = Gaussian_Process(Init_Param, x_code_obs, y_code_obs)

    x_gp_real, y_gp_real, var_gp_real = GP_for_real.GP_interpolation(X_domain)
    x_gp_code, y_gp_code, var_gp_code = GP_for_code.GP_interpolation(X_domain)

    y_gp_real = np.squeeze(y_gp_real)
    y_gp_code = np.squeeze(y_gp_code)

    Model_discrepancy = y_gp_real - y_gp_code

    Y_plus_true,Y_minus_true = GP_for_real.GP_confidence_interval(x_gp_real,y_gp_real,var_gp_real)
    Y_plus_code,Y_minus_code = GP_for_code.GP_confidence_interval(x_gp_code,y_gp_code,var_gp_code)


    plt.figure()
    plt.plot(x_real_obs,y_real_obs,'bo', label="True obs")
    plt.plot(x_gp_real, y_gp_real,'b',label="GP true")

    plt.plot(x_code_obs, y_code_obs,'go', label="Code obs")
    plt.plot(x_gp_code,y_gp_code,'g',label="GP code")
    # plt.plot(x_gp_real,Y_plus_true,'g--')
    # plt.plot(x_gp_real,Y_minus_true,'g--')
    # plt.fill_between(x_gp,Y_interval_Minus, Y_interval_plus, color='gray', alpha = 0.3)

    plt.plot(X_domain,Y_True,'r',label="True process")
    plt.legend()

    plt.figure()
    plt.plot(X_domain, y_gp_code + Model_discrepancy,label="Model discrepancy")
    plt.show()




    # x_obs = [ -1.5, -1.0, -0.75, -0.4, -0.25, 0.0   ]
    # y_obs = [ -1.7, -1.3, -0.4, 0.2, 0.4, 0.8  ]
    #
    # GP = Gaussian_Process(Init_Param,x_obs,y_obs)
    # GP.GP_plot()


