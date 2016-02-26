import numpy as np
import scipy as sp
import pymc3 as pm
import theano.tensor as tt

''' Data generation '''
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


np.random.seed(123456)

K = 3
dim = 2
N = 300
alpha = [0.5]*3
dir_pi = np.squeeze(np.random.dirichlet(alpha,1))

mu = np.random.multivariate_normal(np.zeros(dim),np.eye(dim)*200,K)
Cov = np.eye(dim) # common

Z = np.random.choice(K, N, p=dir_pi)
X = list()

for idx in range(N):
    Zidx = Z[idx]
    X.append(np.random.multivariate_normal(mu[Zidx],Cov))
X = np.array(X)

''' Setup Model '''
model = pm.Model()
with model:
    a = pm.constant(np.array([1.,1.,1.]))
    dir_est = pm.Dirichlet('dir_est',a = a, shape=K)
    dir_potential = pm.Potential('p_min_potential', tt.switch(tt.min(dir_est) < .1, -np.inf, 0))

    means = pm.MvNormal('means',mu=0,tau=np.eye(dim),shape=2)
    order_mean_potential = pm.Potential('order_mean_potential',tt.switch( means[1][0] - means[0][0] < 0,-np.inf,0 ) + tt.switch( means[2][0] - means[1][0]< 0,-np.inf,0 ))
    sd = pm.Uniform('sd',lower = 0, upper = 20,shape=dim)
    Category = pm.Categorical('category',p=dir_est,shape=N)
    points = pm.MvNormal('obs',mu=means[Category], sd=np.eye(sd),observed=X)
                         # mu=means[Category],sd=sd,observed= X)

with model:
    step1 = pm.Metropolis(vars=[dir_est])
    step2 = pm.Metropolis(vars=[sd, means])
    step3 = SequentialScanDiscreteMetropolis(var=Category, values=[0,1,2])
    tr = pm.sample(100, step=[step1, step2] + [step3]*N)
