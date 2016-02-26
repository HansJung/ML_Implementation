import numpy as np
import pymc as pm
from math import pow

nClusters = 2
data = np.array([[10, 10], [9.5, 9.5], [8, 8], [9, 9], [11, 11], [-10, -10],[-8, -8], [-9, -9], [-11, -11]])

sigma0 = pm.InverseGamma('sigma0', 1, 1)

@pm.deterministic
def tau(sig0=sigma0):
	return np.eye(data.shape[1], data.shape[1]) * sig0

covar = [pm.Wishart("cov"+str(i), Tau = tau, n=nClusters+10) for i in xrange(0, nClusters)]

mu0 = pm.MvNormal("mu0", mu = np.array([0 for i in xrange(0, data.shape[1])]), tau = tau)

clMeans = [pm.MvNormalCov("clusterMean"+str(i), mu = mu0, C=covar[i]) for i in xrange(0, nClusters)]

phiTheta = np.array([1 for i in xrange(0,nClusters)])
phi = pm.Dirichlet("phi", theta = phiTheta)


clusterAssignments = [pm.Categorical('assignements'+str(i), p=phi) for i in xrange(0, data.shape[0])]



@pm.stochastic(observed = True)
def sampleLikelihood(name = "Data", value=data, allMeans = clMeans, sigmVals = covar, dataAssignments=clusterAssignments):
	loglike = 0
	for elem in xrange(0, data.shape[0]):
		cl = dataAssignments[elem]

		C = sigmVals[cl]
		lg = pm.mv_normal_cov_like(value[elem,:], allMeans[cl], C)

		loglike+= lg


	return loglike



sampler = pm.MCMC({"sigma0": sigma0,
					"covar": covar,
					"tau": tau,
					"mu0": mu0,
					"clMeans": clMeans,
					"phi": phi,
					"clusterAssignments": clusterAssignments,
					"sampleLikelihood": sampleLikelihood})

sampler.sample(iter=10000, burn=1000)


from pymc.Matplot import plot
from pylab import show
print clMeans[0].value
print clMeans[1].value
print [assign.value for assign in clusterAssignments]
plot(sampler)
show()
