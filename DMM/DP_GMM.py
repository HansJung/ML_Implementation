__author__ = 'jeong-yonghan'

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import pymc3 as pm

data = np.array([10., 11., 12., -10., -11., -12.])

# Precision of Gaussian likelihood
tau_like = 0.5
# Mean and precision of Gaussian base distribution
mu_base = 0.
tau_base = 0.01

# Concentration parameter of Dirichlet process
alpha = 1.

import pymc as pm

def gaussian_mixture(data, pi, tau_like, mu_base, tau_base, K):
    '''
    Create mixture model for data using a Gaussian with mean mu_base
    and precision tau_base as base, with K mixture components and
    alpha as a proxy for how many mixture components are expected.
    Observations with known precision tau_like.
    pi are mixture weights
    '''
    n = len(data)
    # Component to which each data point belongs
    z = pm.Categorical('z', p = pi, size = n)

    # Parameters of each component
    mu_k = pm.Normal('mu_k', mu = mu_base, tau = tau_base, size = K)

    # Observation model
    x = pm.Normal('x', mu = mu_k[z], tau = tau_like, value = data,
                    observed = data)

    return {'z': z, 'mu_k': mu_k, 'x': x}

def sticks(alpha, K):
    '''Creates a truncated stick-breaking construction of GEM distribution.
    Concentration parameter `alpha` and length K.
    AdaptiveMetropolis should be used over `pip`
    '''
    pip = pm.Beta('pip', alpha = 1., beta = alpha, size = K - 1)

    @pm.deterministic(dtype = float)
    def pi(value = np.ones(K)/K, pip = pip):
        pip2 = np.hstack((pip.copy(), [1.]))
        val = [pip2[k]*np.prod(1-pip2[0:k]) for k in range(K)]
        return val

    return {'pip': pip, 'pi': pi}

def model(data, tau_like, mu_base, tau_base, alpha, K = 20):

    mdl_infty_weights = sticks(alpha, K)

    mdl_gaussian = gaussian_mixture(data, mdl_infty_weights['pi'],
                     tau_like, mu_base, tau_base, K)

    return [mdl_infty_weights, mdl_gaussian]


mdl = model(data, tau_like, mu_base, tau_base, alpha)
mcmc = pm.MCMC(mdl)
# Sample
mcmc.sample(1000, 500, 2)
# Stochastic 'z' component membership of each data point
print np.median(mcmc.trace('z')[:], 0)
