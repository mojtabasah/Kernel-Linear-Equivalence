# -*- coding: utf-8 -*-
"""
@author: mojtabasah
"""
from ntk_equivalence import fc_kernel, nn_fc
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from lstsq_torch import lstsq_torch
import pickle
from time import time


if __name__ == '__main__':
    plt.style.use('ggplot')

    n_list = [300, 400, 500, 600, 800, 1000, 1200, 1500, 2000, 
              3000, 4000, 5000, 7500, 10000, 12500, 15000]#,  20000]     #number of training samples
    runs = 3
    n_ts = 200    #Number of test samples
    p = 1500    #Dimension of input x
    output_dim = 1      #Dimesnion of output of neural network
    hidden_dim_true = [100, 100]    #Hidden dimesnions of data generating network
    var_noise = 0.1     #noise variance
    S = np.eye(p)   #Signal Covariance
    Ssqrt = sqrtm(S)   #Square root of covariance matrix
    
    # Neural network parameters
    lam = 5e-3 #1e-8    r#regularization parameter
    lr = 3e-3       #learning rate
    momentum = 0.9
    hidden_dim = [20000]    #True network list of hidden dimensions
    n_epoch = 150   #number of epochs
    bs = None          #batch size
    optimizer='SGD'     #optimization algorithm
    scheduler_step = n_epoch + 0    #Scheduler step
    remove_f0 = True        #Remove initial network values when regressing
    around_init = True      #Regularize parametrs around initial values
    
    n_epoch_kr = 300
    
    n_max = runs*max(n_list)
    X_max = np.random.normal(size=(n_max,p)) @ Ssqrt
    X_ts = np.random.normal(size=(n_ts,p)) @ Ssqrt
    
    true_nn = nn_fc(input_dim=p, output_dim=output_dim, hidden_dim=hidden_dim_true,
               nonlin='relu', bias=False)
    f = lambda x: true_nn(torch.tensor(x, dtype=torch.float)).numpy()
    with torch.no_grad():
        y_max = f(X_max) + np.random.normal(scale=np.sqrt(var_noise), size=(n_max,1))
        y_ts = f(X_ts) + np.random.normal(scale=np.sqrt(var_noise), size=(n_ts,1)) 
    
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    err_nn_mat = np.zeros((runs, len(n_list)))
    err_kr_mat = np.zeros((runs, len(n_list)))
    err_lr_mat = np.zeros((runs, len(n_list)))
    mismatch_mat = np.zeros((runs, len(n_list)))
    energy_mat = np.zeros((runs, len(n_list)))
    time_mat = np.zeros((runs, len(n_list)))
    data_details_dict = {'X_ts':X_ts, 'y_ts':y_ts, 'X_max':X_max, 'y_max':y_max, 'n_list':n_list}

    for r in range(runs):
        for n_id, n in enumerate(n_list):
            print(f'run:{r},    n:{n}')
            lr_kr = 0.0001/(n**(2/3))
            lr = 2.5e-4
            bs = min(1000, n)
            #second derivative of the kernel
            ntk = fc_kernel(input_dim=p, hidden_dim=hidden_dim, output_dim=output_dim)
            lam1, lam2, lam3 = ntk.reg_from_lam(S, lam, eq_kernel=True)
            random_indices = np.random.choice(n_max, size=n, replace=False)

            X = X_max[random_indices]
            y = y_max[random_indices]
            
            t0 = time()
            #%% NN fit
            ntk.nn_fit(X, y, lam=lam, bs=bs, lr=lr, n_epoch=n_epoch, optimizer=optimizer,
                remove_f0=remove_f0, around_init=around_init, 
                scheduler_step=scheduler_step, momentum=momentum)
            yhat_nn = ntk.nn_eval(X_ts).numpy() 
            
            #%% Kernel Regression
            if n < 500:
                ntk.kernel_fit(X, y, lam)
            else:
                ntk.kernel_fit(X, y, lam, method='torch', lr=lr_kr, n_epoch=n_epoch_kr)
            yhat_kr = ntk.kernel_eval(X_ts)   
            
            #%% Regularized Linear Regression
            if n < 500:
                ntk.linear_fit(X, y, lam1, lam2, lam3)
            else:
                ntk.linear_fit(X, y, lam1, lam2, lam3, method='torch', 
                                      lr=lr_kr, n_epoch=n_epoch_kr)
            yhat_lr = ntk.lin_eval(X_ts)
            
            t1 = time()
            print(f'Models fitted in {t1-t0:.2f} seconds...')
            time_mat[r, n_id] = t1 - t0
            #%% Evaluate errors
            mismatch = np.mean((yhat_kr - yhat_lr)**2)
            err_nn = np.mean((yhat_nn - y_ts)**2)
            err_kr = np.mean((yhat_kr - y_ts)**2)
            err_lr = np.mean((yhat_lr - y_ts)**2)
            energy = np.mean(y_ts**2)
            
            data_details_dict[(n, 'nn')] = yhat_nn
            data_details_dict[(n, 'kr')] = yhat_kr
            data_details_dict[(n, 'lr')] = yhat_lr
            
            mismatch_mat[r, n_id] = mismatch
            err_nn_mat[r, n_id] = err_nn
            err_kr_mat[r, n_id] = err_kr
            err_lr_mat[r, n_id] = err_lr
            energy_mat[r, n_id] = energy            
    
    t = len(n_list)
    plt.figure()
    plt.plot(n_list[:t], np.mean(err_nn_mat[:,:t], axis=0), label='neural net fit')
    plt.plot(n_list[:t], np.mean(err_kr_mat[:,:t], axis=0), label='kernel fit')
    plt.plot(n_list[:t], np.mean(err_lr_mat[:,:t], axis=0), label='linear fit', c='forestgreen')
    plt.plot([n_list[0], n_list[t-1]], [var_noise, var_noise], '--', label='oracle')
    plt.xlabel('number of samples')
    plt.ylabel('test error (MSE)')
    plt.legend()
    plt.savefig(f'figures/nn_hidden_{hidden_dim}_hid_true_{hidden_dim_true}_lam_0,005.png', dpi=600)
    
    
    plt.figure()
    plt.plot(n_list[:t], np.mean(err_nn_mat[:,:t]/energy, axis=0), label='neural net fit')
    plt.plot(n_list[:t], np.mean(err_kr_mat[:,:t]/energy, axis=0), label='kernel fit')
    plt.plot(n_list[:t], np.mean(err_lr_mat[:,:t]/energy, axis=0), label='linear fit', c='forestgreen')
    plt.plot([n_list[0], n_list[t-1]], [var_noise/energy, var_noise/energy], '--', label='oracle')

    plt.xlabel('number of samples')
    plt.ylabel('normalized test error (MSE)')
    plt.legend()
    plt.savefig(f'figures/normalized_nn_hidden_{hidden_dim}_hid_true_{hidden_dim_true}_lam_0,005.png', dpi=600)
    
    config_dict = {'n_list':n_list, 'runs':runs, 'hidden':hidden_dim,
                   'hidden_true':hidden_dim_true, 'var_noise':var_noise, 'lam':lam, 'S':S}
    data_dict = {'mismatch_mat':mismatch_mat, 'err_kr':err_kr_mat, 
                 'err_lr':err_lr_mat, 'energy_mat':energy_mat}
    with open('data_nn/nn.pckl', 'wb') as fp:
        pickle.dump((config_dict, data_dict, data_details_dict), fp)
     
    #%%
    idx = 0
    plt.figure()
    plt.plot(n_list[:t], err_nn_mat[idx, :t], label='neural net fit') 
    plt.plot(n_list[:t], err_kr_mat[idx, :t], label='kernel fit')
    plt.plot(n_list[:t], err_lr_mat[idx, :t], label='linear fit', c='forestgreen')
    plt.plot([n_list[0], n_list[t-1]], [var_noise, var_noise], '--', label='oracle')
    plt.xlabel('number of samples')
    plt.ylabel('test error (MSE)')
    plt.legend()
    plt.savefig(f'figures/nn_hidden_{hidden_dim}_hid_true_{hidden_dim_true}_lam_0,005_idx_{idx}.png', dpi=600)
    
    
    
    