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
import pickle
from time import time


if __name__ == '__main__':
    plt.style.use('ggplot')

    n = 3000    #number of training samples
    runs = 1
    n_ts = 200    #Number of test samples
    p = 1000    #Dimension of input x
    output_dim = 1      #Dimesnion of output of neural network
    hidden_dim_true = [100, 100]    #Hidden dimesnions of data generating network
    var_noise = 0.1     #noise variance
    S = np.eye(p)   #Signal Covariance
    Ssqrt = sqrtm(S)   #Square root of covariance matrix
    
    # Neural network parameters
    lam = 5e-3 #1e-8    r#regularization parameter
    lr = 1e-3       #learning rate
    momentum = 0.4
    hidden_dim = [10000]    #True network list of hidden dimensions
    n_epoch = 100   #number of epochs
    bs = None          #batch size
    optimizer='SGD'     #optimization algorithm
    scheduler_step = n_epoch + 0    #Scheduler step
    remove_f0 = True        #Remove initial network values when regressing
    around_init = True      #Regularize parametrs around initial values
        
    X = np.random.normal(size=(n,p)) @ Ssqrt
    X_ts = np.random.normal(size=(n_ts,p)) @ Ssqrt
    
    true_nn = nn_fc(input_dim=p, output_dim=output_dim, hidden_dim=hidden_dim_true,
               nonlin='relu', bias=False)
    f = lambda x: true_nn(torch.tensor(x, dtype=torch.float)).numpy()
    with torch.no_grad():
        y = f(X) + np.random.normal(scale=np.sqrt(var_noise), size=(n,1))
        y_ts = f(X_ts) + np.random.normal(scale=np.sqrt(var_noise), size=(n_ts,1)) 
    
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    

    for r in range(runs):
        lr = 4e-4
        #second derivative of the kernel
        ntk = fc_kernel(input_dim=p, hidden_dim=hidden_dim, output_dim=output_dim)
        lam1, lam2, lam3 = ntk.reg_from_lam(S, lam, eq_kernel=False)
        
        t0 = time()
        
        #%% scaled linear fit
        ntk.scaled_linear_fit(X, y, lam1=lam1, lam2=lam2, bs=bs, lr=lr, n_epoch=n_epoch, 
                              optimizer=optimizer, test_samples=(X_ts, y_ts),
                              scheduler_step=scheduler_step, momentum=momentum)
        #%% NN fit
        ntk.nn_fit(X, y, lam=lam, bs=bs, lr=lr, n_epoch=n_epoch, optimizer=optimizer,
            remove_f0=remove_f0, around_init=around_init, test_samples=(X_ts, y_ts),
            scheduler_step=scheduler_step, momentum=momentum)
        t1 = time()
        
        plt.figure()
        plt.plot(ntk.test_loss_list_lin, label='linear model')
        plt.plot(ntk.test_loss_list_nn, label='neural network')
        plt.xlabel('epochs')
        plt.ylabel('average test error')
        plt.legend()
        plt.savefig('GD_equivalence.png', dpi=600)
        