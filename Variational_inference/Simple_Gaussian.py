__author__ = 'jeong-yonghan'
import numpy as np
from scipy.special import digamma
from scipy.special import gammaln

''' Generate data '''
np.random.seed(123) # for reproducing

# Define hyperparameters
k0 = 3.9
a0 = 1.9
b0 = 0.81
mu0 = 10.27

# parameters sampled from priors
## precision parameter lamb ~ Gam(a0,b0)
lamb = np.random.gamma(a0,b0,1)[0]

## mu ~ N(mu0,1/(k0 x lamb) )
mu = np.random.normal(mu0, np.sqrt((1/(k0*lamb)))  )

# Generate data
N = 2000
D = np.random.normal(mu,np.sqrt(1/lamb),N)



''' VI computation '''
# muN for Q_mu
x_bar = np.mean(D)
muN = (k0*mu0 + N*x_bar) / (k0+N)

# aN for Q_lamb
aN = a0 + (N+1)/2.0

# muN square
def muN_2(kN,muN):
    return muN**2 + (1/kN)

# E[lamb]
def Ex_lam(aN,bN):
    return aN / bN

# Update
## kN
def Compute_kN(k0,N,aN,bN ):
    return (k0 + N) * (aN / bN)

## bN
def Compute_bN(b0,muN_2,mu0,muN,D,k0):
    factor1 = b0
    factor2 = k0* (  muN_2 + (mu0 ** 2) - 2*muN*mu0  )
    factor3 =  (np.sum(D**2) + N*muN_2 - 2*muN*np.sum(D)) /2.0
    return factor1 + factor2 + factor3

def ELBO(kN,aN,bN):
    factor1 = np.log( 1.0/kN ) / 2.0
    factor2 = gammaln(aN)
    factor3 = aN*np.log( bN )
    return factor1 + factor2 - factor3

kN = k0
iterNum = 20
ELBO_prev = -100.0
numIter = 0
for i in range(100):
    numIter += 1
    muN2 = muN_2(kN,muN)
    bN = Compute_bN(b0=b0,muN_2 =muN2, mu0=mu0,muN=muN,D=D,k0=k0)
    kN = Compute_kN(k0=k0,N=N,aN=aN,bN=bN)
    ELBO_cur = ELBO(kN=kN,aN=aN,bN=bN)
    if ELBO_cur - ELBO_prev < 1e-12:
        print numIter
        break
    ELBO_prev = ELBO_cur

# print ""
print mu,1.0/lamb
print muN,1.0/(aN/bN)

