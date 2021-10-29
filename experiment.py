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
n = 800
p = 500

# high dimensional features 
X = np.zeros((n, p)) 
signal1 = np.arange(0, n)
signal2 = 10*np.log(np.arange(0, n) + 1)
X[:, 0], X[:, 1], gau_noise = signal1, signal2, np.random.randn(n, p - 2)
X[:, 2:] = gau_noise
y = signal1 + signal2 + np.random.randn(n,)

# normalization
Xtrain, Xtest, ytrain, ytest = X[:n - 100, :], X[n - 100:, :], y[:n - 100], y[n - 100:]
Xtrain = (Xtrain - np.mean(Xtrain, axis = 0))/np.std(Xtrain, axis = 0)
Xtest = (Xtest - np.mean(Xtest, axis = 0))/np.std(Xtest, axis = 0)
ytrain = ytrain - np.mean(ytrain)
ytest = ytest - np.mean(ytest)

#%%
# Grid search for t and rlambda
for t in np.random.rand(10)*50 + 400:
    for rlambda in np.random.rand(10) + 1200:
        model = SVElasticNet(t, rlambda)
        model.fit(Xtrain, ytrain, 'P')
        beta = model.coef_
        validation_error = np.mean((ytest - Xtest @ beta) ** 2)
        training_error = np.mean((ytrain - Xtrain @ beta) ** 2)
        print(f'(t: {t}, lambda: {rlambda}) Validation error  = %0.8f, \n    Training error = %0.8f' % (validation_error, training_error))

#%%
# Take chosen set of hyperparameters
model_sven = SVElasticNet(406.39820720570765, 1200.494348005788)
model_sven.fit(Xtrain, ytrain, 'P')
print('Coefficients from SVEN:\n', model_sven.coef_[:5])

# Comparison with linear regression
model_linreg = LinearRegression()
model_linreg.fit(Xtrain, ytrain)
print('Coefficients from linear regression:\n', model_linreg.coef_[:5])