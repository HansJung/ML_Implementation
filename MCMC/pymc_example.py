__author__ = 'jeong-yonghan'

import numpy as np
import matplotlib.pyplot as plt
import pymc


# Data generation
Num_obs = 20
x_true = np.random.uniform(0,10,size=Num_obs)
y_true = np.random.uniform(-1,1,size=Num_obs)
alpha_true = 0.5
beta_x_true = 1.0
beta_y_true = 10.0
eps_true = 0.5

z_true = alpha_true + beta_x_true * x_true + beta_y_true*y_true
z_obs = z_true + np.random.normal(0,eps_true,size=Num_obs)

# MCMC
## Define prior distribution with the initial value
alpha = pymc.Uniform('alpha_hans',-100,100,value=np.median(z_obs))
betax = pymc.Uniform('betax_hans',-100,100,value=np.std(z_obs)/np.std(x_true))
betay = pymc.Uniform('betay_hans',-100,100,value=np.std(z_obs)/np.std(y_true))
eps = pymc.Uniform('eps_hans',0,100,value = 0.01)

@pymc.deterministic
def model(alpha=alpha, betax=betax,betay = betay, x = x_true,y = y_true):
    return alpha+betax*x+betay*y

@pymc.deterministic
def tau(eps=eps):
    return np.power(eps,-2)

## Likelihood
data = pymc.Normal('data',mu=model,tau=tau,value=z_obs,observed=True)

sampler = pymc.MCMC([alpha,betax,betay,eps,model,tau,z_obs,x_true,y_true])
# sampler.use_step_method(pymc.AdaptiveMetropolis, [alpha,betax,betay,eps],
#                         scales={alpha:0.1, betax:0.1, betay:1.0, eps:0.1})

# sampler.use_step_method(pymc.AdaptiveMetropolis, [alpha,betax,betay,eps])

sampler.sample(iter=10000,burn=1000)
m_alpha = np.median(alpha.trace())
m_betax = np.median(betax.trace())
m_betay = np.median(betay.trace())
m_eps = np.median(eps.trace())

print alpha.summary()
print betax.summary()
print betay.summary()
print eps.summary()

pymc.Matplot.plot(sampler)





#
# plt.figure()
# plt.title("alpha sampling")
# plt.plot(alpha.trace())
#
# plt.figure()
# plt.title("betax sampling")
# plt.plot(betax.trace())
#
# plt.figure()
# plt.title("betay sampling")
# plt.plot(betay.trace())
#
# plt.figure()
# plt.title("eps sampling")
# plt.plot(eps.trace())
# plt.show()
