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
    