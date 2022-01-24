# -*- coding: utf-8 -*-
"""
@author: mojtabasah
"""
import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader

from lstsq_torch import lstsq_torch


class fc_kernel(nn.Module):
    def __init__(self, input_dim, hidden_dim=[100, 100], output_dim=1, 
                 nonlin='relu', dtype=torch.float32, device='cpu'):
        """
        Test the equivalence of linear ridge regression with training of a fully
        connected ReLU network with with square loss and l2 regularization in 
        the proportional asymptotic regime. This class supports direct kernel
        ridge regression with respect to NTK of a fully connected ReLU network,
        equivalent linear ridge regression as well as equivalent neural network
        training.
        
        Specifically, let f(x; \theta) be a fully connected neural network 
        paramterized by \theta, where \theta is initilized to \theta_0. We show
        that the following regressions learn the same model:
            Model 1: l2 regularized neural network
            \argmin_f \sum_i (y_i - (f(x_i; \theta) - f(x_i, \theta_0)))^2 + 
                        \lam \|\theta - \theta_0\|^2
            
            Model 2: ridge regression
            g(x) = w^T x + b
            where (w, b) = \argmin_{w, b} \sum_i (y_i - w^T x_i - b)^2 + 
                                            \lam_1 \|w\|^2 + \lam_2 b^2
                                        
            Model 3: kernel ridge regression
            h(x) = \sum_alpha_i K(x, x_i)
            where K is the neural tangent kernel of the network given, and
            \alpha = \argmin_\alpha \sum_i (y_i - \sum_j \alpha_j K(x_i, x_j))^2
                                    + \lam \sum_{i,j} \alpha_i\alpha_j K(x_i, x_j) 
        for specific values of \lam, \lam_1, and \lam_2.
        Args:
            input_dim (int): input dimension of the neural network.
            hidden_dim (list of ints, optional): hidden dimensions of the neural
            network. Defaults to [100, 100].
            output_dim (int, optional): output dimension of neural network. Defaults to 1.
            nonlin (str, optional): Type of nonlinearity
                !!!Note that equivalence only works for 'relu' for now!!!
                Defaults to 'relu'.
            dtype (torch data type, optional): What device to put torch models on. 
            Defaults to torch.float32.
            device (torch device, optional): What device to use for trainig in
            PyTorch. Defaults to 'cpu'.

        Returns:
            None.

        """
        super(fc_kernel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.nonlin = nonlin
        self.dtype = dtype
        self.device = device
        self.initialize_network()
        self.initialize_kernel()
        
        self.kr = False
        self.lr = False
        self.nn_trained = False
    
    def initialize_network(self):
        """
        Initializes initial neural network as well as a copy of it that will be
        trained.

        Returns:
            None.

        """
        self.nn = nn_fc(self.input_dim, self.output_dim, self.hidden_dim, 
                 nonlin=self.nonlin, dtype=self.dtype, device=self.device)
        self.nn0 = nn_fc(self.input_dim, self.output_dim, self.hidden_dim, 
                 nonlin=self.nonlin, dtype=self.dtype, device=self.device)
        
        for p, p0 in zip(self.nn.parameters(), self.nn0.parameters()):
            p0.data = p.data + 0
            
        self.nn0.eval()
            
    def initialize_kernel(self):
        """
        Defines two functions that are used to recursively define neural tanget 
        kernel for fully connected ReLU networks.        
        
        Ref: "On the Inductive Bias of Neural Tangent Kernels"
        https://arxiv.org/abs/1905.12173

        Returns:
            None.

        """
        pi = torch.tensor(np.pi, dtype=self.dtype)
        self.k0 = lambda x: 1/pi*(pi - torch.acos(x))
        self.k1 = lambda x: 1/pi*(x*(pi - torch.acos(x)) + torch.sqrt(1 - x**2))
        
    def kernel(self, x1, x2):
        x1 = tensor_format(x1, batch_dim=True, dtype=self.dtype, device=self.device)
        x2 = tensor_format(x2, batch_dim=True, dtype=self.dtype, device=self.device)
        p = self.input_dim
        inner_prod = torch.mm(x1, x2.t())
        x1_norm = torch.norm(x1, p=2, dim=1)
        x2_norm = torch.norm(x2, p=2, dim=1)
        return self.kernel_func(x1_norm**2/p, inner_prod/p, x2_norm**2/p)
            
            
    def kernel_func(self, x1_norm2, inner_prod, x2_norm2, L=None, reformat=True):
        """
        Compute kernel function for a fully connected ReLU network.
        Ref: "On the Inductive Bias of Neural Tangent Kernels"
        https://arxiv.org/abs/1905.12173
        
        The kernel K(x1, x2) is only a function of 
        (norm(x1), inner_prod(x1, x2), norm(x2)), hence the arguments to this 
        method.
        Args:
            x1_norm2 (numpy array or torch tensor): DESCRIPTION.
            inner_prod (numpy array or torch tensor): DESCRIPTION.
            x2_norm2 (numpy array or torch tensor): DESCRIPTION.
            L (int, optional): Number of layers of netowrk. Defaults to None.
            reformat (bool, optional): Whether to reformat input arrays to 
            torch tensors on the correct device with correct data type. 
            Defaults to True.

        Returns:
            a torch tensor containing the kernel matrix.

        """
        if L is None:
            L = len(self.hidden_dim)
        if reformat:
            x1_norm2 = tensor_format(x1_norm2, batch_dim=False, dtype=self.dtype, device=self.device)
            x2_norm2 = tensor_format(x2_norm2, batch_dim=False, dtype=self.dtype, device=self.device)    
            inner_prod = tensor_format(inner_prod, dtype=self.dtype, device=self.device)
        
        if len(x1_norm2.shape) < 1 or len(x2_norm2.shape) < 1:
            norm_prod = torch.sqrt(x1_norm2*x2_norm2)
        else:
            norm_prod = torch.sqrt(torch.outer(x1_norm2, x2_norm2))
        C = inner_prod/norm_prod
        M = torch.max(C)
        if M < 1:
            M = 1
        correlation = {0:C/M}
        K = {0:C/M}
        for i in range(L):
            correlation[i + 1] = self.k1(correlation[i])
            K[i + 1] = correlation[i + 1] + K[i]*self.k0(correlation[i])
        
        self.kernel_hist = {'K':K, 'cor':correlation}
        return self.input_dim*norm_prod*K[L]
    
    def kernel_derivative(self, x1_norm2, inner_prod, x2_norm2):
        """
        Uses autograd to compute the first and second derivative of the kernel 
        for a fully connected ReLU network. This allows us to use generalization
        of El. Karoui's result to approximate the random kernel for large input
        data matrices. This approximation is based on second and first order
        derivatives of the kernel function

        Args:
            x1_norm2 (float): norm of x1.
            inner_prod (float): norm of x2.
            x2_norm2 (float): inner product of x1 and x2.

        Returns:
            d1 (torch tensor): gradient of kernel.
            d2 (torch tensor): Hessian of kernel.

        """
        x = torch.autograd.Variable(torch.tensor([x1_norm2, inner_prod, x2_norm2], dtype=self.dtype)
                                    , requires_grad=True)
        K = self.kernel_func(x[0], x[1], x[2], reformat=False)
        K.backward()
        d1 = x.grad
        d2 = torch.autograd.functional.hessian(lambda x: self.kernel_func(x[0], x[1], x[2], reformat=False), x)
        return d1, d2
        
    def kernel_autograd(self, x1, x2):
        """
        !!!For autograd computation, for now we assume that the network 
        function has only one output dimension!!!
        Finds the NTK kernel matrix between to data matrices x1 and x2 using
        explicit feature map x to gradient of network(x) with respect to network
        parameters.
        Args:
            x1 (numpy array or torch tensor): input data matrix 1.
            x2 (numpy array or torch tensor): input data matrix 2.

        Raises:
            ValueError: For now this method only supports networks with one output
            dimension. Raises a ValueError otherwise.

        Returns:
            K (torch tensor): The kernel matrix K(x1, x2).

        """
        
        if self.output_dim != 1:
            raise ValueError('For autograd computation, the network output dimension should be 1.')
        x1 = tensor_format(x1, batch_dim=True, dtype=self.dtype, device=self.device)
        x2 = tensor_format(x2, batch_dim=True, dtype=self.dtype, device=self.device)
        
        n1, n2 = x1.shape[0], x2.shape[0]
        K = torch.zeros((n1, n2))
        for i, u in enumerate(x1):
            if (i+1)%50 == 0:
                print(f'{i+1}/{n1} complete')
            g1 = []
            self.nn.zero_grad()
            y = self.nn(u)
            y.backward()
            for p1 in self.nn.parameters():
                g1.append(p1.grad.data + 0)
            for j, w in enumerate(x2):
                g2 = []
                self.nn.zero_grad()
                y = self.nn(w)
                y.backward()
                for p2 in self.nn.parameters():
                    g2.append(p2.grad.data + 0)
                for p1, p2 in zip(g1, g2):
                    K[i, j] += torch.sum(p1*p2)
        return K

    def get_c(self, S):
        """
        Find the coefficnets of kernel approximation for El Karoui's result.
        K(X,X) = c0 I + c1 11^T + c2/p XX^T
        Args:
            S (numpy array): The covariance matrix of X.

        Returns:
            None.

        """
        p = self.input_dim
        tau = np.trace(S)/p
        S2 = S.dot(S)
        d1, d2 = self.kernel_derivative(tau, 0, tau)
        c0 = self.kernel_func(tau, tau, tau) -self.kernel_func(tau, 0, tau) - d1[1]*tau
        c1 = self.kernel_func(tau, 0, tau) + d2[1, 1]*np.trace(S2)/(2*p**2)
        c2 = d1[1]
        self.c0, self.c1, self.c2 = c0, c1, c2
    
    def reg_from_lam(self, S, lam, eq_kernel=True):
        """
        Convert the c0, c1, and c2 from in El Karoui's result (see get_c method)
        and an l2 regularization parameter lam used in kernel ridge regression
        to l2 regularization parameters lam1, lam2, and lam3 used in an equivalent
        linear ridge regression problem.
        
        Kernel ridge regression:
            f_ker = argmin_{f\in H} \sum_i(y_i - f(x_i))^2 + \lam \norm{f}^2
            where H is an RKHS reproduced by some kernel K and norm{f} is the 
            Hilbert norm of an f in this RKHS.
            
        Equivalent linear regression:
            f_lin = w^T x + b
            (w, b) = \argmin_{w, b} \sum_i(y_i - f_lin(x_i))^2 + \lam1 |b|^2 + \lam2 \norm{w}^2
        Args:
            S (numpy array): The covariance matrix of X.
            lam (flot): l2 regularization parmaeter used in kernel ridge regression.
            eq_kernel (bool, optional): DESCRIPTION. Defaults to True.

        Returns:
            lam1 (float): DESCRIPTION.
            lam2 (float): DESCRIPTION.
            lam3 (float): DESCRIPTION.

        """
        self.get_c(S)
        self.gamma1 = torch.sqrt(self.c1)
        self.gamma2 = torch.sqrt(self.c2/self.input_dim)
        if eq_kernel:
            lam1 = 1/self.c1
            lam2 = self.input_dim/self.c2
            lam3 = 1/(lam + self.c0)
            return lam1.item(), lam2.item(), lam3.item()
        else:
            lam1 = (lam + self.c0)/self.c1
            lam2 = (lam + self.c0)*self.input_dim/self.c2
            return lam1.item(), lam2.item(), 1.
            
        
        
        
    
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
            optimizer = optim.SGD(params, lr=lr, **kwargs)
        elif optimization_alg == 'RMSprop':
            optimizer = optim.RMSprop(params, lr=lr, momentum=0.2, **kwargs)
        elif optimization_alg == 'Adam':
            optimizer = optim.Adam(params, lr=lr, betas=(0.1, 0.99))
        else:
            raise ValueError('Optimization algorithm not recognized!')
        return optimizer
    
    def nn_fit(self, X, y, lam=0, bs=None, lr=0.00002, n_epoch=30, optimizer='SGD', 
               log_interval=1,scheduler_step=100, gamma=0.75, remove_f0=True,
               around_init=True, test_samples=None, eval_call=None, **kwargs):
        """
        Fits a neural network to the data. By default an l2 regularized neural 
        network is trained with square loss as follows:
            \argmin_\theta \sum_i (y_i - (f(x_i; \theta) - f(x_i, \theta_0)))^2 + 
                        \lam \|\theta - \theta_0\|^2
        where \theta is the parameters of the network and \theta_0 is its
        initialization. One could suppress removal of initial network value
        by setting remove_f0 to False, and regularize norm of theta instead of 
        norm of \theta - \theta_0 by setting around_init to False.
        
        Args:
            X (Numpy array): Inpu data matrix. It should have dimesions n*input_dim
                where n is the number of training samples and input_dim is number of
                features of each sample.
            y (numpy array): a vector of size n.
            lam1 (float): regularization parameter for b
            lam (flot, optional): l2 regularization parameter. Defaults to 0.
            bs (int, optional): batch size. Defaults to None which performs
            full batch gradient descent.
            lr (float, optional): learning rate. Defaults to 0.00002.
            n_epoch (int, optional): number of epochs for training. Defaults to 30.
            optimizer (str, optional): One of 'SGD', 'Adam', or 'RMSprop'. 
                Defaults to 'SGD'.
            log_interval (int, optional): How often the training progress should
                be displayed. Defaults to 1 which prints loss every epoch.
            scheduler_step (int, optional): learning rate scheduler step. 
                Defaults to 100.
            gamma (float, optional): a number between 0 and 1 to be multiplied
                by the learning rate. Defaults to 0.75.
            remove_f0 (bool, optional): Whether to remove initial network. 
                See above for details. Defaults to True.
            around_init (bool, optional): Whether to regularize norm of parameters
                or its distance from its initialization. Defaults to True.
            test_samples (TYPE, optional): DESCRIPTION. Defaults to 20.
            eval_call (TYPE, optional): DESCRIPTION. Defaults to None.
            **kwargs (TYPE): additional arguments that are passed directly
                to the optimizer.

        Returns:
            None.

        """
        if test_samples is not None:
            X_ts, y_ts = test_samples
            X_ts = torch.tensor(X_ts, dtype=self.dtype, device=self.device)
            y_ts = torch.tensor(y_ts, dtype=self.dtype, device=self.device)
            test = True
            self.test_loss_list_nn = []
        dataset = generic_dataset(X, y)
        n = dataset.__len__()
        self.nn.train()
        
        for param in self.nn0.parameters():
            param.requires_grad = False
        self.bs = n if bs is None else bs
        dataloader = DataLoader(dataset, batch_size=self.bs,
                        shuffle=True, num_workers=0)
        self.around_init = 1 if around_init else 0
        self.remove_f0 = remove_f0
        # optimizer
        self.optimizer = self.create_optimizer(self.nn.parameters(), optimizer, lr, **kwargs)
        scheduler = StepLR(self.optimizer, step_size=scheduler_step, gamma=gamma)
                
        for epoch in range(1, n_epoch+1):           
            losses = []
            for batch_idx, (Xb, yb) in enumerate(dataloader):
                self.optimizer.zero_grad()
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                ybhat = self.nn(Xb)
                if remove_f0:
                    yb0 = self.nn0(Xb)
                    ybhat -= yb0
                    
                loss = 1/self.bs*torch.sum((yb - ybhat)**2)
                if lam != 0:
                    l2 = 0
                    for i, (param, param0) in enumerate(zip(self.nn.parameters(), self.nn0.parameters())):
                        l2 += torch.sum((param - self.around_init*param0)**2)
                    loss = loss + lam/n*l2
                loss.backward()
                self.optimizer.step()
                losses.append(loss.detach().numpy())
                scheduler.step()
            if epoch % log_interval == 0:
                print(f'epoch: {epoch}/{n_epoch};   loss: {losses[-1]:.4f}')
            if test:
                with torch.no_grad():
                    yhat_ts = self.nn(X_ts)
                    if remove_f0:
                        yhat_ts0 = self.nn0(X_ts)
                        yhat_ts -= yhat_ts0
                    test_loss = torch.mean((y_ts - yhat_ts)**2)
                    self.test_loss_list_nn.append(test_loss)    
            self.nn_trained = True
    
    def nn_eval(self, x_ts):
        """
        Evaluate the neural network trained on test data x_ts.

        Args:
            x_ts (numpy array or torch tensor): Test data with dimensions 
                n_ts*input_dim where n_ts is the number of test samples and
                input_dim is the number of features.

        Raises:
            ValueError: If neural network is not trained yet.

        Returns:
            yhat (torch tensor): output of the neural network for test data with 
                initial network  output removed if remove_f0 was set in nn_fit.

        """
        if not self.nn_trained:
            raise ValueError('Neural network not trained yet. Call nn_fit first.')
        self.nn.eval()
        x_ts = tensor_format(x_ts, batch_dim=True, dtype=self.dtype, device=self.device)
        with torch.no_grad():
            yhat = self.nn(x_ts)
            if self.remove_f0:
                yhat -= self.nn0(x_ts)
            
        return yhat
            
    def kernel_fit(self, X, y, lam, method='numpy', lr=0.02, n_epoch=200):
        """
        Fits kernel ridgeression with the NTK of the fully connected ReLU network.
        Note that for other types of network, either the kernel_autograd should
        be used to compute the kernel matrix for a wide enough network 
        which is computationally expensive, or the closed form recursive equations
        for the NTK should be implementd in self.kernel method.
        
        h(x) = \sum_alpha_i K(x, x_i)
            where K is the neural tangent kernel of the network given, and
            \alpha = \argmin_\alpha \sum_i (y_i - \sum_j \alpha_j K(x_i, x_j))^2
                                    + \lam \sum_{i,j} \alpha_i\alpha_j K(x_i, x_j) 

        Args:
            X (Numpy array): Inpu data matrix. It should have dimesions n*input_dim
            where n is the number of training samples and input_dim is number of
            features of each sample.
            y (numpy array): a vector of size n.
            lam (float): regularization parameter.

        Returns:
            None.

        """
        n, p = X.shape
        self.X = X
        self.y = y
        self.lam = lam
        self.K_kr = self.kernel(X, X).numpy() + lam*np.eye(n)
        if method == 'numpy':
            self.inv_y_kr = np.linalg.lstsq(self.K_kr, y, rcond=-1)[0]
        elif method == 'torch':
            solver = lstsq_torch(device=self.device)
            self.inv_y_kr = solver.lstsq(self.K_kr, y, lr=lr, n_epoch=n_epoch)
        self.kr = True
    
    def kernel_eval(self, X_ts):
        """
        Evaluates the model learned using kernel ridge regression in kernel_fit
        method on test data.

        Args:
            X_ts (numpy array): test data.

        Raises:
            ValueError: If kernel regression in not fitted yet an error is raised.

        Returns:
            y_ts (numpy array): the estimated output for the given input matrix.
            
        """
        if not self.kr:
            raise ValueError('Kernel regression not fitted yet.')
        self.k_kr = self.kernel(X_ts, self.X).numpy()
        y_ts = self.k_kr.dot(self.inv_y_kr)
        return y_ts
    
    def scaled_linear_fit(self, X, y, lam1=0, lam2=0, bs=None, lr=0.00002, n_epoch=30, 
                          optimizer='SGD', log_interval=1,scheduler_step=100, gamma=0.75,
                          test_samples=None, eval_call=None, **kwargs):
        if test_samples is not None:
            X_ts, y_ts = test_samples
            X_ts = torch.tensor(X_ts, dtype=self.dtype, device=self.device)
            y_ts = torch.tensor(y_ts, dtype=self.dtype, device=self.device)
            test = True
            self.test_loss_list_lin = []
        dataset = generic_dataset(X, y)
        n = dataset.__len__()
        self.w_lin = torch.zeros((self.input_dim,1), device=self.device, dtype=self.dtype, requires_grad=True)
        self.b_lin = torch.zeros((1,), device=self.device, dtype=self.dtype, requires_grad=True)
        self.scaled_lin = lambda x: torch.matmul(x, self.w_lin)*self.gamma2 + self.b_lin*self.gamma1
        
        self.bs = n if bs is None else bs
        dataloader = DataLoader(dataset, batch_size=self.bs,
                        shuffle=True, num_workers=0)
        
        # optimizer
        self.optimizer = self.create_optimizer([self.w_lin, self.b_lin], optimizer, lr, **kwargs)
        scheduler = StepLR(self.optimizer, step_size=scheduler_step, gamma=gamma)
                
        for epoch in range(1, n_epoch+1):           
            losses = []
            for batch_idx, (Xb, yb) in enumerate(dataloader):
                self.optimizer.zero_grad()
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                ybhat = self.scaled_lin(Xb)
                    
                loss = 1/self.bs*torch.sum((yb - ybhat)**2)
                loss += lam1/n*self.gamma1**2*torch.sum(self.b_lin**2) 
                loss += lam2/n*self.gamma2**2*torch.sum(self.w_lin**2)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.detach().numpy())
                scheduler.step()
            if epoch % log_interval == 0:
                print(f'epoch: {epoch}/{n_epoch};   loss: {losses[-1]:.4f}')
            if test:
                with torch.no_grad():
                    yhat_ts = self.scaled_lin(X_ts)
                    test_loss = torch.mean((y_ts - yhat_ts)**2)
                    self.test_loss_list_lin.append(test_loss)
            self.scaled_lin_traned = True
    
    def linear_fit(self, X, y, lam1, lam2, lam3=1, method='numpy', 
                          lr=0.02, n_epoch=200):
        """
        Fits a linear ridge regression model:
            f_lin = w^T x + b
            (w, b) = \argmin_{w, b} \sum_i(y_i - f_lin(x_i))^2 + \lam1 |b|^2 
                                    + \lam2 \norm{w}^2
        
        For equivalence with neural network and NTK kernel ridge regression,
        get values of lam1 and lam2 (and lam3 if ) for a sepcific lam
        from reg_from_lam method. Note that the above equation only has lam1
        and \lam2. If eq_kernel=True is set in reg_from_lam, a parameter lam3
        is also generated which rescales everything so that the linear kernel
        in the ridge regression above is asymptoticly equal to the NTK. See paper
        for more details.
        Args:
            X (Numpy array): Inpu data matrix. It should have dimesions n*input_dim
            where n is the number of training samples and input_dim is number of
            features of each sample.
            y (numpy array): a vector of size n.
            lam1 (float): regularization parameter for b.
            lam2 (float): regularization parameter for w.
            lam3 (float, optional): Defaults to 1.

        Returns:
            None.

        """
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
        
    def lin_eval(self, X_ts):
        """
        Evaluate the linear function fitted to data

        Args:
            X_ts (numpy array): test data.

        Raises:
            ValueError: If linear regression in not fitted yet an error is raised.

        Returns:
            y_ts (numpy array): the estimated output for the given input matrix.

        """
        if not self.lr:
            raise ValueError('Linear regression not fitted yet. Call linear_fit first.')
        self.k_lr = X_ts.dot(self.X.T)/self.lam2 + 1/self.lam1
        # for i in range(X_ts.shape[0]):
        #     self.k_lr[i, i] += 1/self.lam3
        y_ts = self.k_lr.dot(self.inv_y_lr)
        return y_ts
    
class scaled_fc(nn.Module):
    def __init__(self, n_in, n_out, scale=None, bias=True, device='cpu'):
        """
        Defines a scaled fully connected (linear) layer:
            x_out = \sqrt{2/n_in}W x_in + b

        Args:
            n_in (int): number of in features.
            n_out (int): number of out features.
            bias (bool, optional): Whether to use bias. Defaults to True.
            device (torch device, optional): Device to use. Defaults to 'cpu'.
        """
        super(scaled_fc, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.bias = bias
        self.device= device
        self.scale = (torch.sqrt(torch.tensor([n_in], device=device)/2) if 
                      scale is None else 1)
        self.init_weights()
        self.W = nn.Parameter(self.W)
        self.b = nn.Parameter(self.b) if bias else None
        
    def init_weights(self):
        self.W = torch.randn(size=(self.n_out, self.n_in), device=self.device)
        self.b = torch.randn(size=(self.n_out,), device=self.device)
        
    def forward(self, x):
        y = F.linear(x, self.W/self.scale, self.b)
        return y

class nn_fc(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=[100, 100], 
                 nonlin='relu', bias=False, dtype=torch.float32, device='cpu'):
        """
        Defines a fully connected network with the given input, output, and
        hidden dimensions.

        Args:
            input_dim (int): input dimension.
            output_dim (int): output dimension.
            hidden_dim (list of ints, optional): list of dimensions of hidden layers. 
                Defaults to [100, 100].
            nonlin (str, optional): Nonlinearity to use. One of 'relu', 'sigmoid', or 'tanh'. 
                Defaults to 'relu'.
            dtype (torch dtype, optional): torch data type. Defaults to torch.float32.
            device (torch device, optional): torch device. Defaults to 'cpu'.

        Returns:
            None.

        """
        super(nn_fc, self).__init__() 
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.dtype = dtype
        self.device = device
        self.nonlin = self.get_nonlin(nonlin)
        self.create_network()
        
    def get_nonlin(self, name):
        dic = {'relu':nn.ReLU(), 'sigmoid':nn.Sigmoid(), 'tanh':nn.Tanh(), 
               None:lambda x:x}
        if name not in dic.keys():
            raise ValueError('nonlinearity not defined yet!')
        return dic[name]  
        
    def create_network(self):
        dims = [self.input_dim] + self.hidden_dim + [self.output_dim]
        self.dims = dims
        self.layers = nn.ModuleList()
        for din, dout in zip(self.dims[:-1], self.dims[1:]):
            fc = scaled_fc(din, dout, bias=self.bias, device=self.device)
            self.layers.append(fc)
        self.layers[0].scale = 1   
        self.param_list = []
        for param in self.parameters():
            self.param_list.append(param.data)
    
    # forward method
    def forward(self, x): 
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.nonlin(x)
        x = self.layers[-1](x)
        return x


class generic_dataset(Dataset):

    def __init__(self, X, y, dtype=torch.float):
        self.n, self.p = X.shape
        self.X = torch.tensor(X, dtype=dtype)
        self.y = torch.tensor(y, dtype=dtype)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        X = self.X[idx]
        y = self.y[idx]
        sample = (X, y)
        return sample    

def tensor_format(x, batch_dim=False, dtype=torch.float, device='cpu'):
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    x = x.to(dtype=dtype, device=device)
    if batch_dim and len(x.shape)==1:
        x = x[None, :]
    elif batch_dim and len(x.shape)==0:
        x = torch.unsqueeze(x, 0)
    return x
    
if __name__ == '__main__':
    # Data parameters
    n = 800     #Number of training samples
    n_ts = 5    #Number of test samples
    p = 1500    #Dimension of input x
    output_dim = 1      #Dimesnion of output of neural network
    hidden_dim_true = [100, 100]    #Hidden dimesnions of data generating network
    var_noise = 0.1     #noise variance
    S = np.eye(p)   #Signal Covariance
    Ssqrt = sqrtm(S)   #Square root of covariance matrix
    
    # Neural network parameters
    lam = 1e-3 #1e-8    r#regularization parameter
    lr = 1e-3       #learning rate
    momentum = 0.9
    hidden_dim = [20000]    #True network list of hidden dimensions
    n_epoch = 100   #number of epochs
    bs = n          #batch size
    optimizer='SGD'     #optimization algorithm
    scheduler_step = n_epoch + 0    #Scheduler step
    remove_f0 = True        #Remove initial network values when regressing
    around_init = True      #Regularize parametrs around initial values
    
    # Generate data
    
    X = np.random.normal(size=(n,p)) @ Ssqrt
    X_ts = np.random.normal(size=(n_ts,p)) @ Ssqrt
    
    true_nn = nn_fc(input_dim=p, output_dim=output_dim, hidden_dim=hidden_dim_true,
               nonlin='relu', bias=False)
    f = lambda x: true_nn(torch.tensor(x, dtype=torch.float)).numpy()
    with torch.no_grad():
        y = f(X) + np.random.normal(scale=np.sqrt(var_noise), size=(n,1))
        y_ts = f(X_ts) + np.random.normal(scale=np.sqrt(var_noise), size=(n_ts,1))    
    
    
    # Create the fully connected neural network and kernels
    ntk = fc_kernel(input_dim=p, hidden_dim=hidden_dim, output_dim=output_dim)
    
    # K_recursive = ntk.kernel(X_ts, X_ts)
    # K_autograd = ntk.kernel_autograd(X_ts, X_ts)
    
    # print(K_recursive)
    # print(K_autograd)
    # print(K_autograd/K_recursive)
    # ntk.kernel_func(1, 0 , 1)
    # d1, d2 = ntk.kernel_derivative(1., 0., 1.)
    # ntk.get_c(S)
    
    lam1, lam2, lam3 = ntk.reg_from_lam(S, lam, eq_kernel=True)
    
    ntk.nn_fit(X, y, lam=lam, bs=bs, lr=lr, n_epoch=n_epoch, optimizer=optimizer,
                remove_f0=remove_f0, around_init=around_init, 
                scheduler_step=scheduler_step, momentum=momentum)

    ntk.kernel_fit(X, y, lam)
    
    ntk.linear_fit(X, y, lam1, lam2, lam3)
    
    # Evaluate models
    yhat_nn = ntk.nn_eval(X_ts)
    yhat_kr = ntk.kernel_eval(X_ts)
    yhat_lr = ntk.lin_eval(X_ts)
    
    print('Neural Net:\n', yhat_nn)
    print('Kernel:\n', yhat_kr)
    print('Linear:\n', yhat_lr)
    print('Y_ts:\n', y_ts)
    
        