__author__ = 'jeong-yonghan'
import numpy as np
import matplotlib.pyplot as plt


def eta(m):
    var1 = 3*(m**2)
    return var1

def Target_dist(m,y, icov):
    sumval = 0.0
    for i in range(len(y)):
        yi = y[i]
        sumval += -0.5*(yi - eta(m))**2 / (0.25**2)
    sumval = sumval - ((m - 3.5)**2/(2.0 * 0.25**2))
    return sumval
    # return np.exp(sumval)

# def proposal_prop(x,x1,cov_prop):


ndim = 1
m_true = 3.698
icov = np.eye(ndim) / (2.0)
y = list()
#
for i in range(20   ):
    yi = eta(m_true) + np.random.normal(0,0.25**2,1)
    y.append(yi)
y = np.array(y)


## MCMC
Mbox = list()
m = 2.0
cov_prop = 0.25**2
icov_prop = 1.0/ cov_prop

MCMC_iter = 10000
burn_in = 3000

for i in range(MCMC_iter):
    m_cand = np.random.normal(m,cov_prop,1)

    numer = np.exp(Target_dist(m_cand,y,icov))
    deno = np.exp(Target_dist(m,y,icov))
    alpha = min(1,numer/deno)
    uni = np.random.rand(1)
    # print uni
    # print numer,deno, alpha
    if alpha > uni[0]:
        # print "ho"
        m = m_cand
        Mbox.append(m)
    elif alpha <= uni[0]:
        # print "ha"
        Mbox.append(m)


Mbox = Mbox[burn_in:]
print np.mean(Mbox)
plt.plot(Mbox)
plt.show()











