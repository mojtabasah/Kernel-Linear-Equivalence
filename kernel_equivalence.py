# -*- coding: utf-8 -*-
"""
@author: mojtabasah
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from lstsq_torch import lstsq_torch
from time import time


class kernel_equivalence():
    def __init__(self, p, g, g1, g2, S=None, device='cpu'):
        self.p = p
        self.S = np.eye(p) if S is None else S
        self.S2 = self.S.dot(self.S)
        self.g = g
        self.g1 = g1
        self.g2 = g2
        self.get_c()
        self.device = device
        self.kr = False
        self.lr = False
        
    def get_c(self):
        tau = np.trace(self.S)/self.p
        c0 = self.g(tau) -self.g(0) - self.g1(0)*tau
        c1 = self.g(0) + self.g2(0)*np.trace(self.S2)/(2*self.p**2)
        c2 = self.g1(0)
        self.c0, self.c1, self.c2 = c0, c1, c2
        
    def kernel_regression_fit(self, X, y, lam, method='numpy', lr=0.02, n_epoch=200):
        n, p = X.shape
        self.X = X
        self.y = y
        self.lam = lam
        self.K_kr = self.g(X.dot(X.T)/p) + lam*np.eye(n)
        if method == 'numpy':
            self.inv_y_kr = np.linalg.lstsq(self.K_kr, y, rcond=-1)[0]
        elif method == 'torch':
            solver = lstsq_torch(device=self.device)
            self.inv_y_kr = solver.lstsq(self.K_kr, y, lr=lr, n_epoch=n_epoch)
        self.kr = True
        
    def kernel_eval(self, X_ts):
        if not self.kr:
            raise ValueError('Kernel regression not fitted yet.')
        self.k_kr = self.g(X_ts.dot(self.X.T)/self.p)
        y_ts = self.k_kr.dot(self.inv_y_kr)
        return y_ts
    
    def linear_regression(self, X, y, lam1, lam2, lam3=1, method='numpy', 
                          lr=0.02, n_epoch=200):
        n, p = X.shape
        self.X = X
        self.y = y
        self.lam1 = lam1
        self.lam2 = lam2
        self.lam3 = lam3
        self.K_lr = X.dot(X.T)/lam2 + 1/lam1 + np.eye(n)/lam3
        if method == 'numpy':
            self.inv_y_lr = np.linalg.lstsq(self.K_lr, y, rcond=-1)[0]
        elif method == 'torch':
            solver = lstsq_torch(device=self.device)
            self.inv_y_lr = solver.lstsq(self.K_lr, y, lr=lr, n_epoch=n_epoch)
        self.lr = True
        
    def lin_val(self, X_ts):
        if not self.lr:
            raise ValueError('Linear regression not fitted yet.')
        self.k_lr = X_ts.dot(self.X.T)/self.lam2 + 1/self.lam1
        # for i in range(X_ts.shape[0]):
        #     self.k_lr[i, i] += 1/self.lam3
        y_ts = self.k_lr.dot(self.inv_y_lr)
        return y_ts
    
    def reg_from_lam(self, lam, eq_kernel=True):
        self.get_c()
        if eq_kernel:
            lam1 = 1/self.c1
            lam2 = self.p/self.c2
            lam3 = 1/(self.c0 + lam)
        else:
            lam1 = (self.c0 + lam)/self.c1
            lam2 = (self.c0 + lam)*self.p/self.c2
            lam3 = 1
            
        return lam1, lam2, lam3
    
        




if __name__ == '__main__':
    plt.style.use('ggplot')

    n = 3000   #number of training samples
    n_ts = 5   #number of test samples
    p = 2000    #dimensions of x
    poly_deg = 2    #True polynomial degree
    poly_samples = 50   #True polynomial number of kernel terms
    deg = 2     #Regression polynomial degree
    c_true = 0.1    #True polynomial kernel c K(x, x') = (<x,x'>/p + c)^d
    c = 0.1     #regression kernel c
    var_noise = 1   #noise variance in y_i = p(x_i) + noise
    lam = 1e-3      #Regularization parameter
    lr = 1/np.sqrt(n)
    S = np.eye(p)   #Signal Covariance
    Ssqrt = sqrtm(S)   #Square root of covariance matrix
    X = np.random.normal(size=(n,p)) @ Ssqrt
    X_ts = np.random.normal(size=(n_ts,p)) @ Ssqrt
    X_poly = np.random.normal(size=(poly_samples, p))
    
    poly = lambda X: np.sum((X.dot(X_poly.T)/p + c_true)**poly_deg, axis=1) 
    
    y = poly(X) + np.random.normal(scale=np.sqrt(var_noise), size=(n,))
    y_ts = poly(X_ts) + np.random.normal(scale=np.sqrt(var_noise), size=(n_ts,))
    
    g = lambda x: (x + c)**deg      #kernel function
    g1 = lambda x: deg*(x + c)**(deg - 1)*(deg > 0)     #First derivative of kernel
    g2 = lambda x: deg*(deg - 1)*(x + c)**(deg - 2)*(deg > 1)   #second derivative of the kernel
    
    keq = kernel_equivalence(p, g, g1, g2)
    lam1, lam2, lam3 = keq.reg_from_lam(lam)
    
    #%% Kernel Regression
    t0 = time() + 0
    keq.kernel_regression_fit(X, y, lam, method='numpy')
    t1 = time() + 0
    print(f'Closed form kernel regression finished in {t1 - t0:.2f} seconds.')
    yhat_kr = keq.kernel_eval(X_ts)    
    
    t0 = time() + 0
    keq.kernel_regression_fit(X, y, lam, method='torch', lr=lr)
    t1 = time() + 0
    print(f'Pytorch kernel regression finished in {t1 - t0:.2f} seconds.')
    yhat_kr_torch = keq.kernel_eval(X_ts)    
    #%% Regularized Linear Regression
    t0 = time() + 0
    keq.linear_regression(X, y, lam1, lam2, lam3, method='numpy')
    print(f'Closed form linear regression finished in {t1 - t0:.2f} seconds.')
    t1 = time() + 0
    yhat_lr = keq.lin_val(X_ts)
    t0 = time() + 0
    keq.linear_regression(X, y, lam1, lam2, lam3, method='torch', lr=lr)
    print(f'PyTorch linear regression finished in {t1 - t0:.2f} seconds.')
    t1 = time() + 0
    yhat_lr_torch = keq.lin_val(X_ts)
    print('test data:\n', y_ts)
    print('output of kernel model:\n', yhat_kr)
    print('output of torch kernel model:\n', yhat_kr_torch)
    print('output of linear model:\n', yhat_lr)
    print('output of linear model torch:\n', yhat_lr_torch)
    mismatch = np.mean((yhat_kr - yhat_lr)**2)
    energy = np.mean(y_ts**2)
    print(f'mismatch energy/test data energy = {mismatch:.3f}/{energy:.3f} = {mismatch/energy:.3f}')
    err = keq.K_kr - keq.K_lr
    spect_norm = np.linalg.norm(err, 2)
    print(f'spectral norm of error = {spect_norm}')
    
    

    k_lr = keq.k_lr
    k_kr = keq.k_kr
    err_k = k_kr - k_lr
    inv_y_lr = keq.inv_y_lr
    inv_y_kr = keq.inv_y_kr
    err_inv = inv_y_kr - inv_y_lr
    K_lr = keq.K_lr
    K_kr = keq.K_kr
    
    