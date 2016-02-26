import pymc3 as pm, theano.tensor as tt
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns

np.random.seed(12345) # set random seed for reproducibility

k = 3
ndata = 500
spread = 5
centers = np.array([-spread, 0, spread])

# simulate data from mixture distribution
v = np.random.randint(0, k, ndata)
data = centers[v] + np.random.randn(ndata)

model = pm.Model()
with model:
    # cluster sizes
    a = pm.constant(np.array([1., 1., 1.]))
    p = pm.Dirichlet('p', a=a, shape=k)
    # ensure all clusters have some points
    p_min_potential = pm.Potential('p_min_potential', tt.switch(tt.min(p) < .1, -np.inf, 0))


    # cluster centers
    means = pm.Normal('means', mu=[0, 0, 0], sd=15, shape=k)
    # break symmetry
    order_means_potential = pm.Potential('order_means_potential',
                                         tt.switch(np.dot(means[1]-means[0], np.transpose(means[1]-means[0])) < 0, -np.inf, 0)
                                         + tt.switch(means[2]-means[1] < 0, -np.inf, 0))

    # measurement error
    sd = pm.Uniform('sd', lower=0, upper=20)

    # latent cluster of each observation
    category = pm.Categorical('category',
                              p=p,
                              shape=ndata)

    # likelihood for each observed value
    points = pm.Normal('obs',
                       mu=means[category],
                       sd=sd,
                       observed=data)

with model:
    step1 = pm.Metropolis(vars=[p, sd, means])
    step2 = pm.ElemwiseCategoricalStep(vars=[category], values=[0, 1, 2])
    tr = pm.sample(10000, step=[step1, step2])
pm.plots.traceplot(tr, ['p', 'sd', 'means'])


# plt.hist(data)
plt.show()
