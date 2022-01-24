# -*- coding: utf-8 -*-
"""
@author: mojtabasah
"""

from kernel_equivalence import kernel_equivalence
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from lstsq_torch import lstsq_torch
import pickle
from time import time
from sklearn.kernel_ridge import KernelRidge


if __name__ == '__main__':
    plt.style.use('ggplot')

    n_list = [500, 600, 700, 800, 900, 1000, 1200, 1500, 2000, 2500,
              3000, 3500, 4000, 5000, 6000, 7000, 8000, 10000]#, 10000, 13000, 16000,  20000]     #number of training samples
    runs = 5
    n_ts = 500   #number of test samples
    p = 2000    #dimensions of x
    deg = 2    #True polynomial degree
    c = 1     #regression kernel c
    var_noise = 0.1  #noise variance in y_i = p(x_i) + noise
    lam = var_noise      #Regularization parameter
    file_name = 'data_mixture_poly.pckl'
    load_data = False
    fit_sk = False    
    n_epoch = 400
    n_torch = 800
    mixtures = 2
    r = 200

    g = lambda x: (x + c)**deg      #kernel function
    g1 = lambda x: deg*(x + c)**(deg - 1)*(deg > 0)     #First derivative of kernel
    g2 = lambda x: deg*(deg - 1)*(x + c)**(deg - 2)*(deg > 1)
    
    S_list = []
    S_sqrt_list = []
    if load_data:
        with open(file_name, 'rb') as fp:
            (cov, X_max, y_max, X_ts, y_ts) = pickle.load(fp)
    else:
        n_max = 2*max(n_list)
        Xi = np.zeros((n_max + n_ts, p, mixtures))
        for i in range(mixtures):
            S_sqrt_i = np.random.normal(scale=1/np.sqrt(p), size=(p,r))
            Si = S_sqrt_i @ S_sqrt_i.T
            S_list.append(Si)
            S_sqrt_list.append(S_sqrt_i)
            Xi[:,:, i] = np.random.normal(size=(n_max + n_ts,r)) @ S_sqrt_i.T
            
        
        Z = np.random.multinomial(n=1, pvals=[1/mixtures]*mixtures, size=(n_max + n_ts))
        Xi = Z[:, None, :]*Xi
        X_max = np.sum(Xi, axis=-1)
        S = np.mean(S_list)
        X_ts = X_max[-n_ts:,:]
        
        cor = X_max.dot(X_max.T)
        cov = g(cor/p)
        cov_sqrt = np.linalg.cholesky(cov).T
        y_max = np.dot(np.random.randn(n_max + n_ts), cov_sqrt) 
        y_max += np.random.normal(scale=np.sqrt(var_noise), size=(n_ts+n_max))
        y_ts = y_max[-n_ts:]
        y_max = y_max[:-n_ts]
        
    n_max = len(y_max)    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    err_kr_mat = np.zeros((runs, len(n_list)))
    err_sk_mat = np.zeros((runs, len(n_list)))
    err_lr_mat = np.zeros((runs, len(n_list)))
    err_opt_mat = np.zeros((runs, len(n_list)))
    mismatch_mat = np.zeros((runs, len(n_list)))
    energy_mat = np.zeros((runs, len(n_list)))
    time_mat = np.zeros((runs, len(n_list)))

    data_details_dict = {'X_ts':X_ts, 'y_ts':y_ts, 'X_max':X_max, 'y_max':y_max, 'n_list':n_list}
    K_ts_ts = cov[-n_ts:, -n_ts:]
    for r in range(runs):
        for n_id, n in enumerate(n_list):
            #%%
            print(f'run:{r},    n:{n}')

            lr = 0.1/(n**(2/3))
            #second derivative of the kernel
            keq = kernel_equivalence(p, g, g1, g2, device=device)
            lam1, lam2, lam3 = keq.reg_from_lam(lam)

            t0 = time()
            #%% Kernel Regression
            random_indices = np.random.choice(n_max, size=n, replace=False)
            
            X = X_max[random_indices]
            y = y_max[random_indices]
            K_ts_tr = cov[-n_ts:, random_indices]
            K_tr_tr_inv = np.linalg.inv(cov[random_indices, :][:, random_indices] + lam*np.eye(n))
            err_opt_mat[r, n_id] = np.trace(K_ts_ts - K_ts_tr @ K_tr_tr_inv @ K_ts_tr.T)/n_ts + var_noise
            if n < n_torch:
                keq.kernel_regression_fit(X, y, lam)
            else:
                keq.kernel_regression_fit(X, y, lam, method='torch', lr=lr, n_epoch=n_epoch)
            yhat_kr = keq.kernel_eval(X_ts)
            
            if fit_sk:
                sk_kernel = KernelRidge(lam, kernel='polynomial', gamma=1/p, degree=deg, coef0=c)
                sk_kernel.fit(X,y)
                yhat_sk = sk_kernel.predict(X_ts)
            #%% Regularized Linear Regression
            if n < n_torch:
                keq.linear_regression(X, y, lam1, lam2, lam3)
            else:
                keq.linear_regression(X, y, lam1, lam2, lam3, method='torch',
                                      lr=lr, n_epoch=n_epoch)
            yhat_lr = keq.lin_val(X_ts)

            t1 = time()
            print(f'Models fitted in {t1-t0:.2f} seconds...')
            time_mat[r, n_id] = t1 - t0
            #%% Evaluate errors
            mismatch = np.mean((yhat_kr - yhat_lr)**2)
            err_kr = np.mean((yhat_kr - y_ts)**2)
            err_lr = np.mean((yhat_lr - y_ts)**2)
            energy = np.mean(y_ts**2)
            if fit_sk:
                err_sk = np.mean((yhat_sk - y_ts)**2)
                err_sk_mat[r, n_id] = err_sk

            data_details_dict[(n, 'kr')] = yhat_kr
            data_details_dict[(n, 'lr')] = yhat_lr

            mismatch_mat[r, n_id] = mismatch
            err_kr_mat[r, n_id] = err_kr
            err_lr_mat[r, n_id] = err_lr
            energy_mat[r, n_id] = energy
    #%%
    t = len(n_list)

    plt.figure()
    plt.plot(n_list[:t], np.mean(err_kr_mat[:,:t], axis=0), label='kernel fit')
    if fit_sk:
        plt.plot(n_list[:t], np.mean(err_sk_mat[:,:t], axis=0), label='scikit kernel fit')

    plt.plot(n_list[:t], np.mean(err_lr_mat[:,:t], axis=0), label='linear fit')
    plt.plot(n_list[:t], np.mean(err_opt_mat[:,:t], axis=0), '--', label='optimal', c='forestgreen')
    plt.xlabel('number of samples')
    plt.ylabel('test error (MSE)')
    plt.legend()
    plt.savefig(f'figures/Gaussian_poly_deg_{deg}_c1.png', dpi=600)


    plt.figure()
    plt.plot(n_list[:t], np.mean(err_kr_mat[:,:t]/energy, axis=0), label='kernel fit')
    if fit_sk:
        plt.plot(n_list[:t], np.mean(err_sk_mat[:,:t]/energy, axis=0), label='scikit kernel fit')
    plt.plot(n_list[:t], np.mean(err_lr_mat[:,:t]/energy, axis=0), label='linear fit')
    plt.plot(n_list[:t], np.mean(err_opt_mat[:,:t]/energy, axis=0), '--', label='optimal', c='forestgreen')
    plt.xlabel('number of samples')
    plt.ylabel('normalized test error (MSE)')
    plt.legend()
    plt.savefig(f'figures/Gaussian_normalized_poly_deg_{deg}_c1.png', dpi=600)


    config_dict = {'n_list':n_list, 'runs':runs, 'deg':deg,
                   'var_noise':var_noise, 'c':c, 'lam':lam, 'S':S}
    data_dict = {'mismatch_mat':mismatch_mat, 'err_kr':err_kr_mat,
                 'err_lr':err_lr_mat, 'energy_mat':energy_mat}
    with open('data_poly/polynomial.pckl', 'wb') as fp:
        pickle.dump((config_dict, data_dict), fp)

    idx = 0
    plt.figure()
    plt.plot(n_list[:t], err_kr_mat[idx, :t], label='kernel fit')
    plt.plot(n_list[:t], err_lr_mat[idx, :t], label='linear fit')
    plt.plot(n_list[:t], err_opt_mat[idx, :t], '--', label='optimal', c='forestgreen')
    plt.xlabel('number of samples')
    plt.ylabel('test error (MSE)')
    plt.legend()
    plt.savefig(f'figures/Gaussian_poly_deg_{deg}_c1_idx_{idx}.png', dpi=600)



        
