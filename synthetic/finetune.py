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
import pickle
import os


from mymodel import RFModel
from dist import MyDist









# Parameters for data generation
K = 2 # number of classes
d = 16  # dimension of the data
N =128 ## number of neurons
sigma_r, sigma_f = 1,1 # standard deviation     
i_value_r, i_value_f = 1,1 ## intercept value  ##logit is N(i,c^2sigma^2)
alpha = 1        
c_theta = 1 ## scaling of theta (default 1)


rep = 1 # number of repetitions


mean_shift_r, mean_shift_f1,mean_shift_f2,mean_shift_f3 = -1*alpha*torch.ones(d,dtype = torch.float64), 1*alpha* torch.ones(d,dtype = torch.float64),\
            alpha*torch.zeros(d,dtype = torch.float64), alpha*torch.zeros(d,dtype = torch.float64) # mean shift for retain and forget dataset

            
intercept_r, intercept_f1,intercept_f2,intercept_f3 = torch.zeros(K,dtype = torch.float64), torch.zeros(K,dtype = torch.float64),\
            torch.zeros(K,dtype = torch.float64), torch.zeros(K,dtype = torch.float64) # intercept for retain and forget dataset


intercept_r[0] = i_value_r
intercept_f1[1] = i_value_f
intercept_f2[1] = i_value_f
intercept_f3[1] = i_value_f






n_r, n_f, n_eval_r, n_eval_f = 1000, 200, 1000, 1000  # number of samples



theta_r = torch.zeros((d,K),dtype = torch.float64)  # parameters for retain dataset
theta_r[:,0] = c_theta*torch.ones(d,dtype = torch.float64)/d**0.5


theta_f1 = torch.zeros((d,K),dtype = torch.float64)  # parameters for forget dataset
theta_f1[:,1]=c_theta*torch.ones(d,dtype = torch.float64)/d**0.5


theta_f2 = torch.zeros((d,K),dtype = torch.float64)  # parameters for forget dataset
theta_f2[:,1]=c_theta*torch.ones(d,dtype = torch.float64)/d**0.5


theta_f3 = torch.zeros((d,K),dtype = torch.float64)  # parameters for forget dataset
theta_f3[:,1]=c_theta*torch.ones(d,dtype = torch.float64)/d**0.5



retain_dist = MyDist(K = K, theta_list = [theta_r], mu_list = [mean_shift_r], sigma2_list = [sigma_r**2*torch.eye(d,dtype = torch.float64)],\
                     prob_list = [1.],intercept_list=[intercept_r])   ## torch.multivariate.normal uses covariance matrix
forget_dist = MyDist(K = K, theta_list = [theta_f1,theta_f2,theta_f3], mu_list = [mean_shift_f1,mean_shift_f2,mean_shift_f3],\
                      sigma2_list = [sigma_f**2*torch.eye(d,dtype = torch.float64) for ii in range(3)],prob_list = [1.,0.,0],\
                        intercept_list=[intercept_f1,intercept_f2,intercept_f3])





seed_list = [1000,2000,3000,4000,5000]









for seed in seed_list:

    torch.manual_seed(seed)

    W = torch.randn(N,d,dtype = torch.float64)/d**0.5  ## weights for the random features
    #torch.seed()


    #torch.manual_seed(1001)

    ## generate the training dataset
    #torch.manual_seed(1000)
    retain_dataset = retain_dist.generate_data(n_r)
    forget_dataset = forget_dist.generate_data(n_f)





    retain_num = [torch.sum(retain_dataset[1] == ii) for ii in range(K)]
    forget_num = [torch.sum(forget_dataset[1] == ii) for ii in range(K)]
    print(f'retain_num:{retain_num}_forget_num:{forget_num}')




    #y_idk = retain_dist.generate_label(forget_dataset[0])  ##generate idk data
    y_idk = torch.multinomial(F.softmax( torch.zeros((n_f,K),dtype = torch.float64),dtype = torch.float64,dim = -1), num_samples = 1)    ##generate idk data
    assert y_idk.size() == (n_f,1)
    #torch.seed()



    combined_dataset = torch.vstack([retain_dataset[0], forget_dataset[0]]),torch.vstack([retain_dataset[1], forget_dataset[1]])

    ## generate the evaluation dataset



    eval_retain_dataset = retain_dist.generate_data(n_eval_r)
    eval_forget_dataset = forget_dist.generate_data(n_eval_f)









    retain_model = RFModel(input_dim = d, K= K, n_features = N, W = W, a = None, activation = 'relu',add_constant = False)
    finetuned_model = RFModel(input_dim = d, K = K, n_features = N, W = W, a = None, activation = 'relu',add_constant=False,eval_retain_dataset = eval_retain_dataset, eval_forget_dataset = eval_forget_dataset)
                                    


    retain_model.pretrained_model(retain_dataset[0], retain_dataset[1])
    finetuned_model.pretrained_model(combined_dataset[0], combined_dataset[1])




    ## create a subfolder in record file to store the results
    os.makedirs(f'record/record_d{d}_N{N}_K_{K}_nf{n_f}_nr_{n_r}_alpha_{alpha}', exist_ok=True)
    with open(f'record/record_d{d}_N{N}_K_{K}_nf{n_f}_nr_{n_r}_alpha_{alpha}/finetuned_seed_{seed}.pkl', 'wb') as file:
        pickle.dump([(retain_model, finetuned_model),(retain_dataset,forget_dataset,y_idk),\
                    seed,(alpha,sigma_r,sigma_f,c_theta,i_value_r,i_value_f,retain_dist,forget_dist)], file)
        
    












