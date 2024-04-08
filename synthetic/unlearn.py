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




##parameters

K = 2           ## number of classes
d = 16          ## dimension of the input
N = 128         ## number of random features
n_r = 1000      ## number of retain samples
n_f = 200       ## number of forget samples
alpha = 0       ## the value of the gap


run_id = 1      ## the id of the run
seed_list = [1000,2000,3000,4000,5000]  ## the list of seeds

global_steps  = 2000  ## number of steps for training
eval_steps = 25    ## number of steps between evaluations


# Define a list of configurations
configurations = [

    ('GA', {'loss_type': 'grad_ascent','learning_rate': 5e-4}),
    # ('KL', {'loss_type': 'KL','learning_rate': 2e-4}),
    ('GDiff', {'loss_type': 'grad_diff','learning_rate': 2e-4}),
    ('idk', {'loss_type': 'idk','learning_rate': 1e-4}),


    ('NPO', {'loss_type': 'non_paired_dpo','beta': 1, 'learning_rate': 5e-4}),
    # ('NPO_KL', {'loss_type': 'non_paired_dpo_KL','beta': .1,'learning_rate': 2e-4}),
    ('NPO_GDiff', {'loss_type': 'non_paired_dpo_GDiff', 'beta': .1,'learning_rate': 2e-4}),

    # ('dpo', {'loss_type': 'dpo','beta': .1, 'learning_rate': 1e-3}),
    # ('dpo_KL', {'loss_type': 'dpo_KL','beta': .1,'learning_rate': 1e-3}),
    ('dpo_GDiff', {'loss_type': 'dpo_GDiff', 'beta': 0.1,'learning_rate': 1e-3}),


    # ('GDiff_KL', {'loss_type': 'grad_diff_KL_forget','learning_rate': 5e-4}),
    # ('GD', {'loss_type': 'grad_descent','learning_rate': 5e-4}),

    # ('testa', {'loss_type': 'non_paired_dpo', 'beta': 0.1,'learning_rate': 5e-4}),
    # ('testb', {'loss_type':'non_paired_dpo', 'beta': .5,'learning_rate': 1e-3}),
    # ('testc', {'loss_type': 'non_paired_dpo', 'beta': 10.,'learning_rate': 5e-3}),


    # ('NPOa', {'loss_type': 'non_paired_dpo','beta': .01, 'learning_rate': 5e-4}),
    # ('NPOb', {'loss_type': 'non_paired_dpo','beta': .1,'learning_rate': 5e-4}),
    # ('NPOc', {'loss_type': 'non_paired_dpo', 'beta': .2,'learning_rate': 5e-4}),
    # ('NPOd', {'loss_type': 'non_paired_dpo', 'beta': .5,'learning_rate':5e-4}),
    # ('NPOe', {'loss_type': 'non_paired_dpo', 'beta': 1,'learning_rate': 5e-4}),
   # ('NPOf', {'loss_type': 'non_paired_dpo', 'beta': 5,'learning_rate': 5e-3}),

]





############################################################################################################






for seed in seed_list:


    with open(f'record/record_d{d}_N{N}_K_{K}_nf{n_f}_nr_{n_r}_alpha_{alpha}/finetuned_seed_{seed}.pkl', 'rb') as file:
        (retain_model, finetuned_model),(retain_dataset,forget_dataset,y_idk),\
                        seed,(alpha,sigma_r,sigma_f,c_theta,i_value_r,i_value_f,retain_dist,forget_dist) = pickle.load(file)
            



    ## import other parameters
        
    W = finetuned_model.W.clone().detach()  ##fixed first layer weights
    a_pretrained = finetuned_model.a.clone().detach()  ##second layer finetuned weights
        
    eval_retain_dataset = (finetuned_model.X_eval_r_dataset, finetuned_model.y_eval_r_dataset)
    eval_forget_dataset = (finetuned_model.X_eval_f_dataset, finetuned_model.y_eval_f_dataset)

    n_eval_r = eval_retain_dataset[0].size(0)
    n_eval_f = eval_forget_dataset[0].size(0)


    assert n_eval_r == 1000



    


    # Initialize dictionaries to hold records and parameters
    records = {}
    params_dict = {}


    params = {
        'input_dim': d,
        'n_features': N,
        'K': K, ## number of classes
        'W': W,
        'a': a_pretrained,
        'activation': 'relu',
        'finetuned_model': finetuned_model,
        'retain_model': retain_model,
        'eval_num_r': n_eval_r,
        'eval_num_f': n_eval_f,
        'loss_type': 'KL',
        'eval_loss': 'KL',
        'eval_retain_dist': retain_dist,
        'eval_forget_dist': forget_dist,
        'eval_retain_dataset': eval_retain_dataset,
        'eval_forget_dataset': eval_forget_dataset,
        'eval_steps': eval_steps,
        'add_constant': False, ## whether to add constant term

    }

    base_params = params.copy()





    # Initialize a record list for each configuration
    for name, config in configurations:
        records[f"record_{name}"] = []  # Initialize an empty list for each record
        params_dict[f"params_{name}"] = {**base_params, **config}  # Merge base_params with specific config

    


    models = {}
    for name in params_dict:
        model_name = name.replace('params_', 'model_')  # Convert params name to model name
        models[model_name] = RFModel(**params_dict[name])







    for model_name, model in models.items():
    
        learning_rate = params_dict[model_name.replace('model_', 'params_')].get('learning_rate', 5e-4)
        
    
        record_key = f"record_{model_name.replace('model_', '')}"
        records[record_key].append(model.train_model((retain_dataset, forget_dataset, y_idk), steps=global_steps, learning_rate=learning_rate))

        



    #save the records
    if os.path.exists(f'record/record_d{d}_N{N}_K_{K}_nf{n_f}_nr_{n_r}_alpha_{alpha}/record_seed_{seed}_{run_id}.pkl'):

        with open(f'record/record_d{d}_N{N}_K_{K}_nf{n_f}_nr_{n_r}_alpha_{alpha}/record_seed_{seed}_{run_id}.pkl', 'rb') as file:
            records_old, params_dict_old, models_old = pickle.load(file)

        for name, config in configurations:
            records_old[f"record_{name}"] = records[f"record_{name}"] 
            params_dict_old[f"params_{name}"] = params_dict[f"params_{name}"]
            models_old[f"model_{name}"] = models[f"model_{name}"]
        
        with open(f'record/record_d{d}_N{N}_K_{K}_nf{n_f}_nr_{n_r}_alpha_{alpha}/record_seed_{seed}_{run_id}.pkl', 'wb') as file:
            pickle.dump((records_old,params_dict_old,models_old), file)

    else:
        
        with open(f'record/record_d{d}_N{N}_K_{K}_nf{n_f}_nr_{n_r}_alpha_{alpha}/record_seed_{seed}_{run_id}.pkl', 'wb') as file:
            pickle.dump((records,params_dict,models), file)

















