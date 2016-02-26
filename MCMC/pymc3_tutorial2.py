import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
import theano as tn
import scipy as sp
import pandas as pd


'''Data generation : poisson process '''
np.random.seed(123)
disaster_data = np.ma.masked_values([4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
                            3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
                            2, 2, 3, 4, 2, 1, 3, -999, 2, 1, 1, 1, 1, 3, 0, 0,
                            1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
                            0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
                            3, 3, 1, -999, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
                            0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1], value=-999)
year = np.arange(1851, 1962)

'''Model generation'''

HansModel = pm.Model()

with HansModel:
    switchpoint = pm.DiscreteUniform('switchpoint', lower=year.min(), upper=year.max(), testval=1900)

    # prior
    early_rate = pm.Exponential('early_rate',1)
    late_rate = pm.Exponential('late_rate',1)

    # Allocate rate
    rate = pm.switch(switchpoint >= year, early_rate, late_rate)

    # Likelihood
    disaster=pm.Poisson('disaster',mu=rate,observed=disaster_data)


''' MCMC setting '''
with HansModel:
    # Step1 = pm.Slice(vars=[early_rate,late_rate,switchpoint,disaster.missing_values[0]])
    trace = pm.sample(1000,step=pm.NUTS())

pm.traceplot(trace)
print pm.summary(trace)
plt.show()
