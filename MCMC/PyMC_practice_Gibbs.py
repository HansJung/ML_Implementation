import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import pymc3 as pm, theano.tensor as tt

np.random.seed(12345) # set random seed for reproducibility

def stick_breaking(alpha,k):
    # k : number of initial clusters (very large num) / k-1 cut
    betas = np.random.beta(1,alpha,k)
    remaining_pieces = np.append(1,np.cumprod(1-betas[:-1]))
    p = betas * remaining_pieces
    return p/p.sum()


k = 3
ndata = 100
spread = 3
centers = np.array([-spread, 0, spread])
p=stick_breaking(alpha=10.0,k=30)


# simulate data from mixture distribution
v = np.random.randint(0, k, ndata)
data = centers[v] + np.random.randn(ndata)

# plt.hist(data)
# plt.show()

model = pm.Model()
with model:
    # cluster sizes
    a = pm.constant(np.array([1., 1., 1.]))
    # p = pm.Dirichlet('p', a=a, shape=k)
    # ensure all clusters have some points
    # p_min_potential = pm.Potential('p_min_potential', tt.switch(tt.min(p) < .1, -np.inf, 0))


    # cluster centers
    means = pm.Normal('means', mu=[0, 0, 0], sd=15, shape=k)
    # break symmetry
    # order_means_potential = pm.Potential('order_means_potential',
    #                                      tt.switch(means[1]-means[0] < 0, -np.inf, 0)
    #                                      + tt.switch(means[2]-means[1] < 0, -np.inf, 0))

    # measurement error
    sd = pm.Uniform('sd', lower=0, upper=20)

    # latent cluster of each observation
    category = pm.Categorical('category',p=p,shape=ndata)

    # likelihood for each observed value
    points = pm.Normal('obs',
                     mu=means[category],
                     sd=sd,
                     observed=data)

# make some step-methods particularly suited to the category stochastic

class RandomScanDiscreteMetropolis(pm.step_methods.arraystep.ArrayStep):
    def __init__(self, var, model=None, values=[0,1]):
        model = pm.modelcontext(model)
        self.values = values
        super(RandomScanDiscreteMetropolis, self).__init__([var], [model.fastlogp])

    def astep(self, q0, logp):
        i = np.random.choice(len(q0))

        q = np.copy(q0)
        q[i] = np.random.choice(self.values)

        q_new = pm.step_methods.arraystep.metrop_select(logp(q) - logp(q0), q, q0)

        return q_new


class SequentialScanDiscreteMetropolis(pm.step_methods.arraystep.ArrayStep):
    def __init__(self, var, model=None, values=[0,1]):
        model = pm.modelcontext(model)
        self.values = values
        self.i = 0
        super(SequentialScanDiscreteMetropolis, self).__init__([var], [model.fastlogp])

    def astep(self, q0, logp):

        q = np.copy(q0)
        q[self.i] = np.random.choice(self.values)
        self.i = (self.i + 1) % len(q)

        q_new = pm.step_methods.arraystep.metrop_select(logp(q) - logp(q0), q, q0)

        return q_new

# fit model
with model:
    # step1 = pm.Metropolis(vars=[p])
    step2 = pm.Metropolis(vars=[p,sd, means])
    step3 = SequentialScanDiscreteMetropolis(var=category, values=[0,1,2])
    # step3 = pm.Metropolis(var=[category])
    # tr = pm.sample(1000, step=[step1, step2]+ [step3])
    tr = pm.sample(1000, step=[step2] +[step3]*ndata)

pm.traceplot(tr)
plt.show()