import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from scipy.stats import entropy
import scipy
from tqdm import tqdm
import pdb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



class MyDist():  ## logistic distribution
    def __init__(self,K,theta_list,mu_list, sigma2_list,prob_list,intercept_list,translation_list = None):
        self.num_modes = len(mu_list)
        self.mu_list = [ mu.clone().detach() for mu in mu_list]  ## 1-d tensor
        if translation_list is not None:
            self.translation_list = [translation.clone().detach() for translation in translation_list]
        else:
            self.translation_list = [mu.clone().detach() for mu in mu_list] ## 1-d tensor, by default, the translations are the same as mu
        self.d = len(mu_list[0])
        self.sigma2_list = [sigma2.clone().detach() for sigma2 in sigma2_list]  ## 2-d tensor
        self.K = K
        self.theta_list = [theta.clone().detach().view(-1,K) for theta in theta_list]  ## 2-d tensor
        self.prob_list = prob_list
        self.intercept_list = [intercept.clone().detach().view(1,K) for intercept in intercept_list] ## 2-d tensor
        
        assert len(prob_list) == self.num_modes
       
    def generate_data(self,n):
        ##get n samples from prob_list
        mode_id = torch.multinomial(torch.tensor(self.prob_list), num_samples = n, replacement = True)
        assert mode_id.size() == (n,)
        X = torch.zeros(n,self.d,dtype = torch.float64)
        y = torch.zeros(n,1,dtype = torch.int64)
        
        for i in range(n):

            mode = mode_id[i]
            X[i,:] = torch.distributions.MultivariateNormal(self.mu_list[mode], self.sigma2_list[mode]).sample()
            logits = F.softmax((X[i,:]-self.translation_list[mode]).view(1,-1)@self.theta_list[mode]+self.intercept_list[mode], dim=-1,dtype = torch.float64)  ## 1 by K
            y[i,0] = torch.multinomial(logits,num_samples = 1)  ## 1 by 1
            

     
        # X = torch.distributions.MultivariateNormal(self.mu, self.sigma2).sample((n,))
        # ##generate from K-dim discrete distribution
        # logits = F.softmax(X@self.theta, dim=-1,dtype = torch.float64)  ## n by K
    
        # y = torch.multinomial(logits,num_samples = 1)  ## n by 1
        
        
        return X, y
    def generate_label(self,X):  ## not working for now

        logits = F.softmax(X@self.theta, dim=-1,dtype = torch.float64)  ## n by K
        y = torch.multinomial(logits,num_samples = 1)  ## n by 1
        

           
        return  y








# class MyDist():  ## logistic distribution; old version
#     def __init__(self,K,theta_list,mu_list, sigma2_list,prob_list):
#         self.num_modes = len(mu_list)
#         self.mu_list = [ mu.clone().detach() for mu in mu_list]  ## 1-d tensor
        
#         self.d = len(mu_list[0])
#         self.sigma2_list = [sigma.clone().detach() for sigma in sigma2_list]  ## 2-d tensor
#         self.K = K
#         self.theta_list = [theta.clone().detach().view(-1,K) for theta in theta_list]  ## 2-d tensor
#         self.prob_list = prob_list
        

#         assert len(prob_list) == self.num_modes
       
#     def generate_data(self,n):
#         ##get n samples from prob_list
#         mode_id = torch.multinomial(torch.tensor(self.prob_list), num_samples = n, replacement = True)
#         assert mode_id.size() == (n,)
#         X = torch.zeros(n,self.d,dtype = torch.float64)
#         y = torch.zeros(n,1,dtype = torch.int64)
        
#         for i in range(n):
#             mode = mode_id[i]
#             X[i,:] = torch.distributions.MultivariateNormal(self.mu_list[mode], self.sigma2_list[mode]).sample()
#             logits = F.softmax(X[i,:].view(1,-1)@self.theta_list[mode], dim=-1,dtype = torch.float64)  ## 1 by K
#             y[i,0] = torch.multinomial(logits,num_samples = 1)  ## 1 by 1

     
#         # X = torch.distributions.MultivariateNormal(self.mu, self.sigma).sample((n,))
#         # ##generate from K-dim discrete distribution
#         # logits = F.softmax(X@self.theta, dim=-1,dtype = torch.float64)  ## n by K
    
#         # y = torch.multinomial(logits,num_samples = 1)  ## n by 1
        
        
#         return X, y
#     def generate_label(self,X):  ## not working for now

#         logits = F.softmax(X@self.theta, dim=-1,dtype = torch.float64)  ## n by K
#         y = torch.multinomial(logits,num_samples = 1)  ## n by 1
        

           
#         return  y





# class MyDist():  ## logistic distribution
#     def __init__(self,K,mu_list, sigma2_list,prob_list,mode_dist):
#         self.num_modes = len(mu_list)
#         self.mu_list = [ mu.clone().detach() for mu in mu_list]  ## 1-d tensor
#         self.d = len(mu_list[0])
#         self.sigma2_list = [sigma.clone().detach() for sigma in sigma2_list]  ## 2-d tensor
#         self.K = K

#         self.prob_list = [prob.clone().detach() for prob in prob_list]
#         self.mode_dist = torch.tensor(mode_dist,dtype = torch.float64)
#         assert len(mode_dist) == self.num_modes
#         assert len(prob_list)>0 and self.prob_list[0].size() == (K,)
       
#     def generate_data(self,n):
#         ##get n samples from prob_list
#         mode_id = torch.multinomial(torch.tensor(self.mode_dist), num_samples = n, replacement = True)
#         assert mode_id.size() == (n,)
#         X = torch.zeros(n,self.d,dtype = torch.float64)  
       
#         y = torch.zeros(n,1,dtype = torch.int64)
        
#         for i in range(n):
#             mode = mode_id[i]
#             X[i,:] = torch.distributions.MultivariateNormal(self.mu_list[mode], self.sigma2_list[mode]).sample()
            
#             y[i,0] = torch.multinomial(self.prob_list[mode],num_samples = 1)  ## 1 by 1

     
#         # X = torch.distributions.MultivariateNormal(self.mu, self.sigma).sample((n,))
#         # ##generate from K-dim discrete distribution
#         # logits = F.softmax(X@self.theta, dim=-1,dtype = torch.float64)  ## n by K
    
#         # y = torch.multinomial(logits,num_samples = 1)  ## n by 1
            
        
        
        
#         return X, y
#     def generate_label(self,X):  ## not working for now

#         pass
