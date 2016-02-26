import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import pymc as pm

# Stick breaking process
def stick_breaking(alpha,k):
    # k : number of initial clusters (very large num) / k-1 cut
    betas = np.random.beta(1,alpha,k)
    remaining_pieces = np.append(1,np.cumprod(1-betas[:-1]))
    p = betas * remaining_pieces
    return p/p.sum()


# For example, suppose base line distribution P0 ~ N(0,1)
k = 25
alpha = 7.0
theta = np.random.normal(0,1,k)
pi = stick_breaking(alpha,k )

def dirichlet_process(p,n, P0=np.random.randn):
    theta = P0(len(p)) # draw rv from base measure as many as len(p)
    return np.random.choice(theta, size=n, p=p) # choose as many as size n with p proportional


def chinese_restaurant_process(n, alpha):
    if n < 1:
        return None
    table_assignments = np.empty(n)
    next_table = 0
    for c in range(n):
        if np.random.random() < (1. * alpha / (alpha + c)):

            # Sit at new table
            table_assignments[c] = next_table
            next_table += 1

        else:
            # Calculate selection probabilities as function of population
            probs = [(table_assignments[:c]==i).sum()/float(c)
                     for i in range(next_table)]
            # Randomly assign to existing table
            table_assignments[c] = np.random.choice(range(next_table), p=probs)
    return table_assignments

srrs2 = pd.read_csv('srrs2.dat')
srrs2.columns = srrs2.columns.map(str.strip)
srrs_mn = srrs2[srrs2.state=='MN']

counties = srrs_mn.county.values
y = srrs_mn.activity.values
x = srrs_mn.floor.values

## gelman adjustment for log
y[y==0]=.1
y = np.log(y)

def createCountyIndex(counties):
    counties_uniq = sorted(set(counties))
    counties_dict = dict()
    for i, v in enumerate(counties_uniq):
        counties_dict[v] = i
    ans = np.empty(len(counties),dtype='int')
    for i in range(0,len(counties)):
        ans[i] = counties_dict[counties[i]]
    return ans

index_c = createCountyIndex(counties)

# print index_c

N_dp = 100
alpha = pm.Uniform('alpha',lower=0.5,upper=10)
nu = len(set(counties)) - 1
tau_0 = pm.Gamma('tau_0', nu/2., nu/2.)
mu_0 = pm.Normal('mu_0', mu=0, tau=0.01, value=0)
theta = pm.Normal('theta', mu=mu_0, tau=tau_0, size=N_dp)
v = pm.Beta('v', alpha=1, beta=alpha, size=N_dp)

@pm.deterministic
def p(v=v):
    """ Calculate Dirichlet probabilities """

    # Probabilities from betas
    value = [u*np.prod(1-v[:i]) for i,u in enumerate(v)]
    # Enforce sum to unity constraint
    value /= np.sum(value)

    return value

# Expected value of random effect
E_dp = pm.Lambda('E_dp', lambda p=p, theta=theta: np.dot(p, theta))
z = pm.Categorical('z', p, size=len(set(counties)))
# Index random effect
a = pm.Lambda('a', lambda z=z, theta=theta: theta[z])

b = pm.Normal('b', mu=0., tau=0.0001)

y_hat = pm.Lambda('y_hat', lambda a=a, b=b: a[index_c] + b*x)
sigma_y = pm.Uniform('sigma_y', lower=0, upper=100)
tau_y = sigma_y**-2

y_like = pm.Normal('y_like', mu=y_hat, tau=tau_y, value=y, observed=True)
M = pm.MCMC([a, b, sigma_y, y_like, z, v, mu_0, tau_0, theta, alpha, E_dp])
M.sample(2000,1000)
print M.trace('z')[:]

plt.plot(x,y,'bo')
plt.show()

