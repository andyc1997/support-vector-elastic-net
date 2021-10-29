# -*- coding: utf-8 -*-
"""
Support Vector Elastic Net
A Reduction of the Elastic Net to Support Vector Machines with an Application to GPU Computing (Zhou, Chen & Song, 2015)
Training a Support Vector Machine in the Primal (Chapelle, 2006)
"""

import numpy as np
from scipy import linalg
from scipy.optimize import minimize

class SVElasticNet:
    
    def __init__(self, t, rlambda):
        # hyperparameters
        self.t = t
        self.rlambda = rlambda
        
        # data
        self.X = None
        self.y = None
        self.coef_ = None
        self.alpha_ = None
        
        # domains
        if self.t <= 0:
            print('SVElasticNet error: upper boundary cannot be zero or negative.')
        if self.rlambda < 0:
            print('SVElasticNet error: regularization constant cannot be negative.')
    
    def primalOptimize(self, kernel, y, plambda, pbeta, sv):
        # primal newton's method with squared hinge loss
        n = y.size
        sv_prev = set([-1])
        
        if n > 1000:
            n //= 2
            pbeta, sv = self.primalOptimize(kernel[:n, :n], y[:n, :], plambda, pbeta, sv)
            sv = set(*np.nonzero(pbeta[:, 0]))
            
        else:
            sv = set(range(0, n))
        
        max_iter = 1000
        while len(sv_prev.difference(sv)) > 0:
            li_sv = list(sv)
            coef_mat = kernel[li_sv, li_sv] + plambda*np.identity(len(sv))
            const_vec = y[li_sv] 
            
            pbeta_sv = linalg.solve(coef_mat, const_vec)
            pbeta = np.zeros((y.size, 1))
            for i in range(len(sv)):
                pbeta[li_sv[i]] = pbeta_sv[i]
                    
            sv_prev, sv = sv, set()
            tight = y * (kernel @ pbeta)
            
            for i in range(n):
                if tight[i] < 1:
                    sv.add(i)
            
            max_iter -= 1
            if max_iter == 0:
                print('Ill-posed problem.')
                break

        return pbeta, sv

    def SVMprimal(self, X, y, C):
        # primal problem
        plambda = 1/C
        kernel = X @ X.T
        pbeta = np.zeros((X.shape[1], 1))
        sv = set()
        pbeta, _ = self.primalOptimize(kernel, y, plambda, pbeta, sv)
        
        return pbeta
    
    def dualObj(self, alpha, Z, C):
        # dual objective function
        return np.sum((Z.T @ alpha.reshape(-1, 1)) ** 2) + 1/(2*C)*np.sum(alpha ** 2) - 2*np.sum(alpha)
    
    def SVMdual(self, X, y, C):
        # dual problem
        Z = X * y
        ineq_cons = ({'type': 'ineq', 'fun': lambda x: x})
        initial_guess = np.random.rand(Z.shape[0])
        res = minimize(self.dualObj, initial_guess, args = (Z, C), method = 'SLSQP',
                       constraints = ineq_cons, tol = 1e-6)
        
        return res.x
        
    def fit(self, X, y, mode):
        # get data
        self.X = np.asarray(X)
        self.y = np.asarray(y).reshape(-1, 1)
        n, p = self.X.shape
        
        if n < 2:
            print('SVElasticNet error: insufficient observations.')
            
        # artificial dataset
        constructX = (np.concatenate((self.X - self.y/self.t, self.X + self.y/self.t),
                                     axis = 1)).T
        constructy = np.concatenate((np.ones((p, 1)), -np.ones((p, 1))), axis = 0)
        
        # initialize variables
        alpha, C = None, 1/(2*self.rlambda)
        
        # case of high dimension: primal
        if 2*p > n or mode == 'P':
            pbeta = self.SVMprimal(constructX, constructy, C)
            w = (constructX.T @ pbeta)
            alpha = C*np.maximum(np.zeros(constructy.shape), np.ones(constructy.shape) - 
                                  constructy*(constructX @ w))
        # otherwise: dual
        else:
            alpha = self.SVMdual(constructX, constructy, C)
            alpha = alpha.reshape(-1, 1)
        
        # update results
        self.alpha_ = alpha
        self.coef_ = self.t*(alpha[0:p, 0] - alpha[p:2*p, 0])/np.sum(alpha)
        
