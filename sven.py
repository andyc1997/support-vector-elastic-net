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
    
    def primalObj(self, pbeta, kernel, y, plambda):
        margin = plambda*np.dot(pbeta.flatten(), (kernel @ pbeta).flatten())
        arg = np.dot((kernel.T @ pbeta).flatten(), y.flatten())
        loss = (np.maximum(np.zeros(arg.shape), 1 - arg)) ** 2
        
        return plambda*margin + np.sum(loss)
    
    def primalLinSearch(self, gamma, pbeta, pbeta_opt, kernel, y, plambda):
        beta = pbeta + gamma*(pbeta_opt - pbeta)
        
        return self.primalObj(beta, kernel, y, plambda)
    
    def primalOptimize(self, kernel, y, plambda, pbeta, sv):
        # primal newton's method with squared hinge loss
        n = y.size
        
        if n > 1000:
            n //= 2
            pbeta, sv = self.primalOptimize(kernel[:n, :n], y[:n, :], plambda, pbeta, sv)
            sv = set(*np.nonzero(pbeta[:, 0]))
            
        else:
            sv = set(range(0, n))
        
        max_iter = 100
        tol = 1e-6
        
        sv_prev = set([-1])
        
        while len(sv_prev.difference(sv)) > 0:
            print('New iteration: Estimating support vectors...')
            sv_prev = sv
            li_sv = list(sv)
            coef_mat = kernel[li_sv, li_sv] + plambda*np.identity(len(sv))
            const_vec = y[li_sv]
            
            pbeta_sv = linalg.solve(coef_mat, const_vec)
            pbeta = np.zeros((y.size, 1))
            for i in range(len(sv)):
                pbeta[li_sv[i]] = pbeta_sv[i]
    
            sv = set()
            tight = y*(kernel @ pbeta)
            for i in range(n):
                if tight[i] < 1:
                    sv.add(i)
                    
            print('Gradient descent...')
            li_sv = list(sv) 
            res = minimize(self.primalObj, pbeta[li_sv, :], 
                           (kernel[li_sv, :][:, li_sv], y[li_sv, :], plambda), 
                           method = 'CG', jac = '3-point', options = {'maxiter': 100, 'disp': True})
            if res.success == 0:
                print('Convergence failed...')
                return pbeta, sv
            pbeta_opt = pbeta.copy()
            pbeta_opt[li_sv, :] = res.x.reshape(-1, 1)
            
            print('Backtracking...')
            res = minimize(self.primalLinSearch, 0.6, (pbeta, pbeta_opt, kernel, y, plambda),
                           method = 'SLSQP', jac = 'cs', bounds = [(0, np.inf)])
            gamma_opt = res.x
            pbeta_opt = pbeta + gamma_opt*(pbeta_opt - pbeta)
            
            if np.sum(np.abs(pbeta_opt - pbeta)) < tol:
                print('Convergence')
                return pbeta, sv
            else:
                pbeta = pbeta_opt
            
            max_iter -= 1
            if max_iter == 0:
                print('Ill-posed problem.')
                return pbeta, sv
        
        print('-'*20)
        return pbeta, sv

    def SVMprimal(self, X, y, C):
        # primal problem
        plambda = 1/C
        kernel = X @ X.T
        pbeta = np.zeros((X.shape[0], 1))
        sv = set()
        pbeta, _ = self.primalOptimize(kernel, y, plambda, pbeta, sv)
        
        return pbeta
    
    def dualObj(self, alpha, Z, C):
        # dual objective function
        return np.sum((Z.T @ alpha.reshape(-1, 1)) ** 2) + 1/(2*C)*np.sum(alpha ** 2) - 2*np.sum(alpha)
    
    def SVMdual(self, X, y, C):
        # dual problem
        Z = X * y
        bound = [(0, np.inf)]*Z.shape[0]
        initial_guess = np.random.rand(Z.shape[0])
        res = minimize(self.dualObj, initial_guess, args = (Z, C), method = 'L-BFGS-B',
                       bounds = bound , tol = 1e-6)
        
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
        if (2*p > n or mode == 'P') and (mode != 'D'):
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
        
