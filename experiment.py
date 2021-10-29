# -*- coding: utf-8 -*-
"""
Testing code for SVElasticNet
"""
import numpy as np
from matplotlib import pyplot as plt 

from sven import SVElasticNet 
from sklearn.linear_model import LinearRegression

#%%        
# Simple example: linear trend + sinusoid waves + gaussian noise
# under a high dimensional setting p >> n
n = 1000
p = 1600

# high dimensional features 
X = np.zeros((n, p)) 
signal1 = np.arange(0, n)
signal2 = 10*np.log(np.arange(0, n) + 1)
X[:, 1], X[:, 2], gau_noise = signal1, signal2, np.random.randn(n, p)
X += gau_noise
y = signal1 + signal2

# normalization
Xtrain, Xtest, ytrain, ytest = X[:900, :], X[900:, :], y[:900], y[900:]
Xtrain = (Xtrain - np.mean(Xtrain, axis = 0))/np.std(Xtrain, axis = 0)
Xtest = (Xtest - np.mean(Xtest, axis = 0))/np.std(Xtest, axis = 0)
ytrain = ytrain - np.mean(ytrain)
ytest = ytest - np.mean(ytest)


# plot signals
for i in range(0, 3):
    plt.plot(np.arange(0, 1000), X[:, i], ls = 'solid')
    plt.show()
    
plt.plot(np.arange(0, 1000), y, ls = 'solid')
plt.show()

#%%
# Grid search for t and rlambda
for t in np.random.rand(10)*10:
    for rlambda in [1200]:
        model = SVElasticNet(t, rlambda)
        model.fit(Xtrain, ytrain, 'P')
        beta = model.coef_
        validation_error = np.mean((ytest - Xtest @ beta) ** 2)
        training_error = np.mean((ytrain - Xtrain @ beta) ** 2)
        print(f'(t: {t}, lambda: {rlambda}) Validation error  = %0.8f, \n    Training error = %0.8f' % (validation_error, training_error))

#%%
# Take chosen set of hyperparameters
model_sven = SVElasticNet(7.285563180686971, 800)
model_sven.fit(Xtrain, ytrain, 'P')
print('Coefficients from SVEN:\n', model_sven.coef_, model_sven.alpha_)

# Comparison with linear regression
model_linreg = LinearRegression()
model_linreg.fit(Xtrain, ytrain)
print('Coefficients from linear regression:\n', model_linreg.coef_)