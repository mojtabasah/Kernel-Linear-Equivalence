# -*- coding: utf-8 -*-
"""
@author: mojtabasah
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

class lstsq_torch(nn.Module):
    def __init__(self, device='cpu', dtype=torch.float32):
        super(lstsq_torch, self).__init__()
        self.device = device
        self.dtype = dtype

    def create_optimizer(self, params, optimization_alg, lr, **kwargs):
        """
        Create the optimizer instance to be used for neural network training.

        Args:
            params (torch parameters): parameters to be optimized.
            optimization_alg (str): One of 'SGD', 'Adam', or 'RMSprop'.
            lr (float): learning rate.
            **kwargs: Other arguments that are passed directly to the optimizer.

        Raises:
            ValueError: if an optimization_alg that is not implemented is called.

        Returns:
            optimizer: torch optimizer instance.

        """
        if optimization_alg == 'SGD':
            optimizer = optim.SGD(params, lr=lr, momentum=0.9, **kwargs)
        elif optimization_alg == 'RMSprop':
            optimizer = optim.RMSprop(params, lr=lr, momentum=0.2, **kwargs)
        elif optimization_alg == 'Adam':
            optimizer = optim.Adam(params, lr=lr, betas=(0.1, 0.99))
        else:
            raise ValueError('Optimization algorithm not recognized!')
        return optimizer


    def lstsq(self, K, y, optimizer='SGD', lr=0.02,
               n_epoch=200, log_interval=1, scheduler_step=100, gamma=0.75,
               eval_call=None, verbose=False, **kwargs):
        """
        Solves the linear equation Kx = y by minimizing the squared error
        $\|Kx - y\|^2$ with respect to x using gradient descent. This allows
        us to solve kernel ridge regression problems with large number of 
        training samples without the need for matrix inversion which is very
        expensive computationally. Gradient descent on the other hand, with 
        appropritate choise of learning rate converges very fast specially 
        if the original kernel ridge regression problem has a positive regularization
        parameter which guarantees the quadratic probelm here to be strongly convex
        and hence linearly converging.

        Parameters
        ----------
        K : numpy array
            The "data" matrix in the equation Kx=y.
        y : numpy array
            The response matrix in the equation Kx=y.
        optimizer : str, optional
            The optimizer to use to solve the quadratic problem. See 
            create_optimizer method for a list of defined optimizers. 
            The default is 'SGD'.
        lr : float, optional
            Learning rate of the optimizer. The default is 0.02.
        n_epoch : int, optional
            number of epochs for the optimizer to run. The default is 200.
        log_interval : int, optional
            After how many epochs a summary of the results should be printed in
            std_out. Note that verbose should be set to True. The default is 1.
        scheduler_step : int, optional
            after how many epochs should the learning rate scheduler take a step. 
            The default is 100.
        gamma : float, optional
            the scaling factor of the learning rate in the scheduler. The 
            default is 0.75.
        eval_call : function, optional
            Any function that should be called in each epoch, e.g to see performance
            on test data, etc. The default is None.
        verbose : bool, optional
            If true, a summary of the state of optimization problem is printed
            every log_interval. The default is False.
        **kwargs : TYPE
            Other optimizaer specefic arguments that should be passed to it.

        Returns
        -------
        coef : numpy array
            The solution x that minimizes $\|Kx - y\|^2$ .

        """
        if not torch.is_tensor(K):
            K = torch.tensor(K, dtype=self.dtype)
        if not torch.is_tensor(y):
            y = torch.tensor(y, dtype=self.dtype)
        y = torch.squeeze(y)
        n = len(y)

        self.alpha = torch.zeros((n,), device=self.device, dtype=self.dtype, requires_grad=True)

        K, y = K.to(self.device), y.to(self.device)
        # optimizer
        self.optimizer = self.create_optimizer([self.alpha], optimizer, lr, **kwargs)
        scheduler = StepLR(self.optimizer, step_size=scheduler_step, gamma=gamma)

        for epoch in range(1, n_epoch+1):
            losses = []
            self.optimizer.zero_grad()
            loss = torch.matmul(K, self.alpha)
            loss = torch.matmul(self.alpha, loss)/2
            loss -= torch.matmul(self.alpha, y)

            loss.backward()
            self.optimizer.step()
            losses.append(loss.detach().cpu().clone().numpy())
            scheduler.step()
            if verbose:
                if epoch % log_interval == 0:
                    print(f'epoch: {epoch}/{n_epoch};   loss: {losses[-1]:.4f}')


        coef = self.alpha.data.to("cpu").numpy()

        return coef
    
    
if __name__== '__main__':
    n = 1000
    lam = 0.1
    noise_var = 0.0
    lr = 0.5
    
    K = np.random.normal(size=(n, n), scale=1/np.sqrt(n))
    K = K @ K.T + lam*np.eye(n)
    alpha = np.random.normal(size=(n,), scale=1/np.sqrt(n))
    y = (K).dot(alpha) + np.random.randn(n)*np.sqrt(noise_var)
    
    solver = lstsq_torch()
    alphahat = solver.lstsq(K, y, lr=lr)
    
    err = np.mean((alpha - alphahat)**2)/np.mean(alpha**2)
    print(err)
    
