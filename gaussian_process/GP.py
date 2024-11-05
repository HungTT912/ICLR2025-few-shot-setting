import torch 
import numpy as np 
import gc 
from time import time 
from gpytorch.kernels import(
    RBFKernel, LinearKernel, MaternKernel, RQKernel, PeriodicKernel,
    CosineKernel, PolynomialKernel 
)
kernel_dict = {'rbf': RBFKernel,'matern': MaternKernel, 
                'rq' : RQKernel, 'period': PeriodicKernel, 'cosine': CosineKernel,
                'poly': PolynomialKernel}
class GP_hyper: 
    def __init__(self, device, lengthscale, variance,noise, mean_prior, delta_lengthscale, delta_variance): 
        self.device = device 
        self.lengthscale = lengthscale 
        self.variance = variance 
        self.noise = noise 
        self.mean_prior = mean_prior 
        self.delta_lengthscale = delta_lengthscale
        self.delta_variance = delta_variance
class GP: 
    def __init__(self,device, x_train, y_train, lengthscale, variance, noise, mean_prior, kernel='rbf'):
        
        self.device = device 
        self.x_train = x_train
        self.y_train = y_train 
        self.kernel = kernel_dict[kernel]().to(device)
        self.noise = noise
        self.variance = variance
        self.mean_prior = mean_prior
        self.kernel.lengthscale = lengthscale
        
    def set_hyper(self, lengthscale, variance): 
        
        self.variance = variance 
        self.kernel.lengthscale = lengthscale
        if hasattr(self, 'coef'):
            del self.coef
        # torch.cuda.empty_cache()
        with torch.no_grad():
            # import pdb; pdb.set_trace()
            self.K_train_train = self.variance*self.kernel.forward(self.x_train, self.x_train)
            self.K_train_train.diagonal().add_(self.noise)  # In-place modification
            L = torch.linalg.cholesky(self.K_train_train)
            b = (self.y_train - self.mean_prior).unsqueeze(-1)
            self.coef = torch.cholesky_solve(b, L).squeeze(-1).detach()
            #del K_train_train, L, b 
            # torch.cuda.empty_cache()

    def sampling_pseudo_label(self): 
        normal_dist = torch.distributions.Normal(0.0,1.0)
        epsilon = normal_dist.sample((self.x_train.shape[0],)).to(self.device)
        return torch.matmul(self.K_train_train,epsilon)
    
    def mean_posterior(self, x_test): 
        # Posterior mean
        K_train_test = self.variance * self.kernel.forward(self.x_train, x_test)
        mu_star = self.mean_prior + torch.matmul(K_train_test.T, self.coef)
        # Posterior covariance
        #K_star_star = K_test_test - torch.matmul(K_train_test.T, torch.matmul(K_train_train_inv, K_train_test))

        return mu_star
    
    