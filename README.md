# Overview
This repository can be used to show equivalence of a large class of kernel models, including fully connected networks in kernel regime, with linear models in high dimensions. See the paper **[Kernel Methods and Multi-layer Perceptrons Learn Linear Models in High Dimensions
](https://arxiv.org/abs/2201.08082)** for more details. The code in this repository can be used to recreate the figures in the paper as well as explore equiavelnce of other kernel models with linear models.

# Summary of the Results
In the paper, we show that in a certain high-dimensional regime where the number of samples and covariates are both going to infinity at a fixed ratio (proportional asymptotics), applying kernel ridge regression for the kernels of the form

<img src="https://latex.codecogs.com/svg.image?K(\mathbf{x}_i,&space;\mathbf{x}_j)&space;=&space;g\left(\frac{\|\mathbf{x}_i\|_2^2}{p},&space;\frac{\langle&space;\mathbf{x}_i,&space;\mathbf{x}_j&space;\rangle}{p},&space;\frac{\|\mathbf{x}_j\|_2^2}{p}\right)" title="K(\mathbf{x}_i, \mathbf{x}_j) = g\left(\frac{\|\mathbf{x}_i\|_2^2}{p}, \frac{\langle \mathbf{x}_i, \mathbf{x}_j \rangle}{p}, \frac{\|\mathbf{x}_j\|_2^2}{p}\right)" />

is equivalent to learning a linear model with specific ridge regualarizers under certain assumptions on the distribution of the data and regularity conditions on the kernel. This class of kernels is very large and includes widely used kernels such as RBFs, polynomial kernels, and Laplace kernel. Furthermore, the neural tangent kernel of fully connected architectures as well as resdual networks with fully connected blocks are also functions of the inner product as well as Euclidean norm of the inputs. Therefore, the NTK of such networks also belongs to this class. As such, even fully connected and residual neural networks in this high-dimensional regime only learn linear models. More specifically, we show three interesting results in this paper:
1. First, we show that kernel models only learn linear relations between the covariates and the response in this high-dimensional regime. 
    Consequently, kernel models (including neural networks in the kernel regime) have no benefit over linear models in this regime. 
2. Our second result considers the training dynamics of the kernel and linear models. We show that under gradient descent, in the high dimensional setting, dynamics of the kernel model and a scaled linear model are equivalent throughout training.
3. Finally, we consider the case where the true model has a Gaussian process prior with a covariance kernel that satisifes our assumptions. In this case, the relation between the covariates and the responses can be highly nonlinear. In this case, we show that in the high-dimensional limit, the linear model is indeed Bayes optimal. That is, not only nonlinear kernel methods provide no benefit, but also linear models are in fact optimal with respect to squared error, i.e. they achieve the minimum mean squared risk achievable by any estimator.

# How to use the files
The main classes that establish the equivalence of the kernel models with linear models are implemented in [kernel_equivalence.py](https://github.com/mojtabasah/Kernel-Linear-Equivalence/blob/main/kernel_equivalence.py) and [ntk_equivalence.py](https://github.com/mojtabasah/Kernel-Linear-Equivalence/blob/main/ntk_equivalence.py). The file [lstsq_torch.py](https://github.com/mojtabasah/Kernel-Linear-Equivalence/blob/main/lstsq_torch.py) is used to solve kernel ridge regression probelms with large sample size using graident descent instead of matrix inversion.

## [kernel_equivalence.py](https://github.com/mojtabasah/Kernel-Linear-Equivalence/blob/main/kernel_equivalence.py)
The class kernel_equivalnce defined in this file establishes the equivalnce between inner product kernels and linear models. A kernel function and its derivatives are used to find the coefficients that make the kernel data matrix equiavelent to linear kernel data matrix. We can then fit a kernel ridge regression model as well as a linear ridge regression model with appropriate regularization parameters derived from [reg_from_lam](https://github.com/mojtabasah/Kernel-Linear-Equivalence/blob/2720d8732e5533df7bc714b5534db2f55cb96364/kernel_equivalence.py#L221) method and evalute them on test data to see their equivalence. A simple example is implemented at the end of this file.

## [ntk_equivalence.py](https://github.com/mojtabasah/Kernel-Linear-Equivalence/blob/main/ntk_equivalence.py)
This class defines the equivalnce between the NTK of a specifically scaled fully connected ReLU netwerk and linear models. This class can be used to compute the NTK either via the gradients of the output with respect to each weight of the network or through a recursive formula that exists for these networks (See [On the Inductive Bias of Neural Tangent Kernels](https://arxiv.org/abs/1905.12173)). To get the equivalent linear kernel parameters, the partial derivatives of this kernel function is computed by backpropogating through these recursive formulae. Next, nn_fit, kernel_fit, and linear_fit methods can be called to fit a neural network, the kernel model for the NTK of the same network, or the  linear equivalent model and then the outputs of each model can be obtained via nn_eval, kernel_eval, and lin_eval methods respectively to see their equivalence. As toy example to show how to use this file is implemented at the end of the file.

## [lstsq_torch.py](https://github.com/mojtabasah/Kernel-Linear-Equivalence/blob/main/lstsq_torch.py)
This is an auxiliary file to solve the least squares problem via gradient descent instead of matrix inversion. In kernel ridge regression, an equation of the form $(K+ l.I)x = y$ needs to be solved. The  closed form formula using the matrix inversion is computationally expensive to use when the dimensions of this problem (determined by the number of samples) are large. Instead, one can solve this by applyig gradient descent (or its variants like RMSprop, Adam, etc.) to a quadratic problem. These methods can converge quickly even for problems with hundreds of thousands of samples (when matrix inversion is essentially impossible to do directly using for example Numpy's built-in methods) specially if the regularization parameter in the kernel ridge regression problem is positive, which guarantees the quadratic problem to be strongly convex. Therefore, this PyTorch lstq solver might be of independent interest to people who have to solve large kernel ridge regression problems. A simple example on how to use this file is shown at the end of the file. *Note that the class solves the equation $Kx=y$, i.e. if KRR is being performd, $K+l.I$ should be evaluated first and passed as $K$ directly.*

## [Examples](https://github.com/mojtabasah/Kernel-Linear-Equivalence/blob/main/examples)
These files generate the figures in the paper **[Kernel Methods and Multi-layer Perceptrons Learn Linear Models in High Dimensions
](https://arxiv.org/abs/2201.08082)**. 
- [n_sweep_nn.py](https://github.com/mojtabasah/Kernel-Linear-Equivalence/blob/main/examples/n%20sweep_nn.py) shows the equivalnce between a neural network, its NTK models, and the equivalent linear model.
- [equivalence_GD.py](https://github.com/mojtabasah/Kernel-Linear-Equivalence/blob/main/examples/equivalence_GD.py) shows the equivalence of kernel models and linear models throughout training via gradient descent.
- [n_sweep_Gaussian_poly.py](https://github.com/mojtabasah/Kernel-Linear-Equivalence/blob/main/examples/n_sweep_Gaussian_poly.py) shows the optimality of the linear model when the true model has a Gaussian process prior.
- [n_sweep_poly_mixture.py](https://github.com/mojtabasah/Kernel-Linear-Equivalence/blob/main/examples/n%20sweep_poly_mixture.py) shows a case where the assumptions of our theorems are violated (a low-rank mixture data model) which breaks down the equivalence of linear and kernel models, i.e. kernel models outperform linear models if more complex data models are considered, hence the need for more complex distributions to truly understand benefits of the kernel models.
