import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import scipy
from tqdm import tqdm
import pdb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class RFModel(nn.Module):
    def __init__(self, input_dim, n_features, K,  W = None, a= None , activation='relu', finetuned_model= None,\
                  retain_model = None, loss_type='GA',eval_loss = 'l2',\
                    eval_retain_dataset = None,eval_forget_dataset = None,beta = None,eval_steps = 1,add_constant = False,**kwargs):
     
        super(RFModel, self).__init__()
        self.input_dim = input_dim
        self.n_features = n_features
        self.K = K ## number of classes
        assert input_dim > 0 and n_features > 0

        if W is None:
            self.W = torch.tensor(torch.randn(input_dim, n_features)/input_dim**0.5,dtype = torch.float64,requires_grad = False)  # Random weight matrix W
        else:
            self.W = W.clone().detach().to(torch.float64)
        if a is None:
            if add_constant:
                self.a = torch.zeros((n_features+1, K-1),dtype = torch.float64,requires_grad = True)
            else:
                self.a = torch.zeros((n_features, K-1),dtype = torch.float64,requires_grad = True)

        else:
            self.a = a.clone().detach().to(torch.float64).requires_grad_(True)
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError("Unsupported activation function.")
        
        self.finetuned_model = finetuned_model      
        self.retain_model = retain_model

       

        if not (eval_retain_dataset is None or eval_forget_dataset is None):
            self.X_eval_r_dataset, self.y_eval_r_dataset = eval_retain_dataset
            self.X_eval_f_dataset, self.y_eval_f_dataset = eval_forget_dataset

        self.loss_type = loss_type
        self.eval_loss = eval_loss
        self.beta = beta  ## parameter for the dpo/npo loss
        self.eval_steps = eval_steps
        self.add_constant = add_constant
        
       
       

    def pretrained_model(self, X, y):
        """ Train a random feature model """
        
        Z = self.activation(X @ self.W.T)
        
        if self.add_constant:
            Z = torch.hstack([Z, torch.ones(Z.shape[0],1,dtype = torch.float64)])  ## add a constant term

        y_onehot = F.one_hot(y.view(-1), num_classes=self.K).to(torch.float64)


       

        record_loss = []

        v = self.a.clone().detach().requires_grad_(True)
       
        num_iter = 20000
        for ii in range(num_iter):
            
            y_onehot = F.one_hot(y.view(-1), num_classes = self.K).to(torch.float64)
            if self.add_constant:
                log_logits = F.log_softmax(Z @ torch.hstack([v,torch.zeros(self.n_features+1,1,dtype = torch.float64)]),dim = -1)

            else:   
                log_logits = F.log_softmax(Z @ torch.hstack([v,torch.zeros(self.n_features,1,dtype = torch.float64)]),dim = -1)
            
            assert y_onehot.shape == log_logits.shape
            loss = -torch.mean(torch.sum(y_onehot*log_logits,dim=-1))
            loss.backward()
            record_loss.append(loss.item())
            with torch.no_grad():
                if ii == 0:
                    print(f'grad_{torch.norm(v.grad)}_loss_{loss.item()}_iter_{ii}_initial')
                if ii == num_iter-1:
                    print(f'grad_{torch.norm(v.grad)}_loss_{loss.item()}_iter_{ii}')
                    if torch.norm(v.grad) > 1e-3:
                        print('not fully converged_grad is:',torch.norm(v.grad))
                    if torch.norm(v.grad) >1e-2:
                        print('not fully converged_grad is:',torch.norm(v.grad))
                        raise ValueError("Not fully converged")
                        
                v -= .05*v.grad   ## default learning rate is 0.05,20000steps (for overlap distribution)
                v.grad.zero_()
            
            
        # plt.plot(record_loss)
        # plt.show()
                
        

                
        

       

        self.a = v.clone().detach().requires_grad_(True)
        
        
       
        return v.clone().detach().requires_grad_(False)
        
       
    
    def forward(self, X,log_target = False):
        """
        Forward pass through the model.
        """

        


        Z = self.activation(X @ self.W.T)
        if self.add_constant:
            Z = torch.hstack([Z, torch.ones(Z.shape[0],1,dtype = torch.float64)])  ## add a constant term
        if self.add_constant:
            a_combined = torch.hstack([self.a, torch.zeros(self.n_features+1, 1, dtype=torch.float64)])
        else:
            a_combined = torch.hstack([self.a, torch.zeros(self.n_features, 1, dtype=torch.float64)])

        if log_target:
            predictions = F.log_softmax(Z @ a_combined, dim = -1)
        else:
            predictions = F.softmax(Z @ a_combined,dim = -1)
            assert predictions.shape == (X.shape[0],self.K)
            
        return predictions

    def train_model(self, data, learning_rate=0.01, steps=100,evaluate = True):


        """
        Trains the model using gradient descent.
        """

        ##start with the evaluation
        

        (X_r, y_r), (X_f, y_f), y_idk = data

        record_all = {}
        for key in ['forget_quality', 'retain_quality', 'forget_diff', 'retain_diff',"a_dist"]:
            record_all[key] = []

        ## change the label to one-hot encoding
        y_r = F.one_hot(y_r.view(-1), num_classes = self.K).to(torch.float64)
        y_f = F.one_hot(y_f.view(-1), num_classes = self.K).to(torch.float64)
        y_idk = F.one_hot(y_idk.view(-1), num_classes = self.K).to(torch.float64)

        assert y_r.shape == (X_r.shape[0],self.K)
        assert y_f.shape == (X_f.shape[0],self.K)

        




        for step_curr in tqdm(range(steps)):



            ## start with the evaluation
            if evaluate and  step_curr % self.eval_steps == 0:
                
                eval_dict = self.evaluation()
                for key, value in eval_dict.items():
                    record_all[key].append(value)



                ##test only
                self.test_eval_prob(X_f,y_f, data_type = 'forget')
                #self.test_eval_prob(X_r, y_r,data_type = 'retain')

         
            


            if self.a.grad is not None:
                self.a.grad.zero_()
            
            ## compute the loss
            if self.loss_type == 'grad_ascent':
                
                prediction = self.forward(X_f,log_target=True) ## log prediction
                assert prediction.shape == (X_f.shape[0],self.K)
                #if torch.isnan(prediction).any():
                    #pdb.set_trace()
                #loss = torch.mean(y_f*torch.log(prediction)+ (1-y_f)*torch.log(1-prediction))
                loss = torch.mean(torch.sum(y_f*prediction,dim=-1))
            elif self.loss_type == 'grad_descent':  ## for sanity check
                prediction = self.forward(X_r,log_target=True) ## log prediction
                
                loss = -torch.mean(torch.sum(y_r*prediction,dim=-1))
            elif self.loss_type == 'grad_diff':
                prediction_f = self.forward(X_f,log_target=True)
                prediction_r = self.forward(X_r,log_target=True)

                loss_f = torch.mean(torch.sum(y_f*prediction_f,dim=-1))
                loss_r = -torch.mean(torch.sum(y_r*prediction_r,dim=-1))
    
                loss = loss_f + loss_r
            elif self.loss_type == 'grad_diff_KL_forget':
                prediction_f = self.forward(X_f,log_target=True)
                prediction_r = self.forward(X_r,log_target=True)
                with torch.no_grad():   
                    prediction_f_finetuned = self.finetuned_model.forward(X_f,log_target=True)
                
                
                loss_f = torch.mean(torch.sum(y_f*prediction_f,dim=-1))
                loss_r = -torch.mean(torch.sum(y_r*prediction_r,dim=-1))

                kl_forget = F.kl_div(prediction_f,prediction_f_finetuned, reduction = 'batchmean',log_target = True)  ##this computes KL(finetunwed||current)
                ## This is equivalent to the following
                #my_kl_forget = torch.mean(torch.sum(torch.exp(prediction_f_finetuned)*(prediction_f_finetuned-prediction_f),dim = -1))
               
                
    
                loss = loss_f + loss_r + 2*kl_forget
            elif self.loss_type == 'KL':
                
                prediction_f = self.forward(X_f,log_target=True)
                prediction_r = self.forward(X_r,log_target=True) 
                
                with torch.no_grad():   
                    prediction_r_finetuned = self.finetuned_model.forward(X_r,log_target=True)

                
          
                

                
                loss_f = torch.mean(torch.sum(y_f*prediction_f,dim=-1))

                kl_retain = F.kl_div(prediction_r, prediction_r_finetuned, reduction = 'batchmean',log_target = True)  ##this computes KL(finetunwed||current)
                
                
                loss = loss_f + kl_retain

            elif self.loss_type == 'idk':

                prediction_f = self.forward(X_f,log_target=True)
                prediction_r = self.forward(X_r,log_target=True)

                loss_r = -torch.mean(torch.sum(y_r*prediction_r,dim=-1))


                
                loss = - torch.mean(torch.sum(y_idk*prediction_f,dim=-1))

                loss = loss + loss_r



            elif self.loss_type in ['non_paired_dpo','non_paired_dpo_KL','non_paired_dpo_GDiff']:

                prediction_f = self.forward(X_f,log_target=True)
                outputs_f = torch.sum(prediction_f*y_f,dim=-1).view(-1,1)

                with torch.no_grad():
                    prediction_f_finetuned = self.finetuned_model.forward(X_f,log_target=True)

                    outputs_f_finetuned = torch.sum(prediction_f_finetuned*y_f,dim=-1).view(-1,1) 
                    assert outputs_f_finetuned.shape == (X_f.shape[0],1)

                
                neg_log_ratio = outputs_f_finetuned - outputs_f
                loss = -F.logsigmoid(self.beta*neg_log_ratio).mean()*2/self.beta


            

        

                if self.loss_type == 'non_paired_dpo_KL':
                    prediction_r = self.forward(X_r,log_target=True) 
                    with torch.no_grad():   
                        prediction_r_finetuned = self.finetuned_model.forward(X_r,log_target=True)

                    
                    kl_retain = F.kl_div(prediction_r, prediction_r_finetuned, reduction = 'batchmean',log_target = True)  ##this computes KL(finetunwed||current)
                    
                    loss = loss + kl_retain
                
                if self.loss_type == 'non_paired_dpo_GDiff':
                    prediction_r = self.forward(X_r,log_target=True)

                    loss_r = -torch.mean(torch.sum(y_r*prediction_r,dim=-1))
                    loss = loss + loss_r

        

            elif self.loss_type in ['dpo','dpo_KL','dpo_GDiff']:
                prediction_f = self.forward(X_f,log_target=True)
                with torch.no_grad():
                    prediction_f_finetuned = self.finetuned_model.forward(X_f,log_target=True)

                    outputs_f_finetuned = torch.sum(prediction_f_finetuned*y_f,dim=-1).view(-1,1)
                    outputs_f_finetuned_idk = torch.sum(prediction_f_finetuned*y_idk,dim=-1).view(-1,1)



                outputs_f = torch.sum(prediction_f*y_f,dim=-1).view(-1,1)
                outputs_f_idk = torch.sum(prediction_f*y_idk,dim=-1).view(-1,1)


                

                log_ratio = (outputs_f_idk - outputs_f_finetuned_idk)- (outputs_f - outputs_f_finetuned)
                loss = -F.logsigmoid(self.beta*log_ratio).mean()*2/self.beta
                if self.loss_type == 'dpo_KL':
                    prediction_r = self.forward(X_r,log_target=True) 
                    with torch.no_grad():   
                        prediction_r_finetuned = self.finetuned_model.forward(X_r,log_target=True)


                    kl_retain = F.kl_div(prediction_r, prediction_r_finetuned, reduction = 'batchmean',log_target = True)  ##this computes KL(finetunwed||current)

                    
                    
                    loss = loss + kl_retain
                if self.loss_type == 'dpo_GDiff':
                    prediction_r = self.forward(X_r,log_target=True)

                    loss_r = -torch.mean(torch.sum(y_r*prediction_r,dim=-1))
                   
                    loss = loss + loss_r

            elif self.loss_type in ['kto']:
                prediction_f = self.forward(X_f,log_target= True)
                outputs_f = torch.sum(prediction_f*y_f,dim=-1).view(-1,1)

                with torch.no_grad():
                    prediction_f_finetuned = self.finetuned_model.forward(X_f,log_target=True)
                    outputs_f_finetuned = torch.sum(prediction_f_finetuned*y_f,dim=-1).view(-1,1)

                    

                
                neg_log_ratio = outputs_f_finetuned-outputs_f

                with torch.no_grad():
                    prediction_f2 = self.forward(X_f,log_target=True)
                    kl_divergence = F.kl_div(prediction_f_finetuned, prediction_f2, reduction = 'batchmean',log_target = True)


                all_log_ratio = kl_divergence + neg_log_ratio
                assert all_log_ratio.shape == neg_log_ratio.shape
                loss = -F.logsigmoid(self.beta*all_log_ratio).mean()*2/self.beta
                


                
                


            
            # Compute gradients
            loss.backward()
        
            with torch.no_grad():  # Ensure gradients are not tracked during the update

              
                
                self.a -= learning_rate * self.a.grad
                

            
        
        if evaluate:
            #print(self.a- self.retain_model.a)
            return record_all
        
        
        
                

               
                
    
    def evaluation(self):
        if self.eval_loss == 'KL':
            eval_logarithm = True
        else:
            eval_logarithm = False
        with torch.no_grad():
            
            #pdb.set_trace()

            p_hat_r = self.forward(self.X_eval_r_dataset,log_target=eval_logarithm)  ## note that this is NOT logprobability
            p_hat_r_retain = self.retain_model.forward(self.X_eval_r_dataset,log_target=eval_logarithm)
            p_hat_r_finetuned = self.finetuned_model.forward(self.X_eval_r_dataset,log_target=eval_logarithm)
           
            p_hat_f = self.forward(self.X_eval_f_dataset,log_target=eval_logarithm)
            p_hat_f_retain = self.retain_model.forward(self.X_eval_f_dataset,log_target=eval_logarithm)    
            p_hat_f_finetuned = self.finetuned_model.forward(self.X_eval_f_dataset,log_target=eval_logarithm)
        
   

        eval_dict = {}  ## dictionary to store evaluation results
        eval_dict['forget_quality'] = self.eval_fun(p_hat_f, p_hat_f_retain,self.eval_loss) ## loss between current and retain model on forget distribution
        eval_dict['retain_quality'] = self.eval_fun(p_hat_r, p_hat_r_retain,self.eval_loss) ## loss between current and retain model on retain distribution
        eval_dict['forget_diff'] = self.eval_fun(p_hat_f, p_hat_f_finetuned,self.eval_loss) ## loss divergence between current and finetuned model on forget distribution
        eval_dict['retain_diff'] = self.eval_fun(p_hat_r, p_hat_r_finetuned,self.eval_loss) ## loss divergence between current and finetuned model on retain distribution
        with torch.no_grad():
            eval_dict['a_dist'] = torch.norm(self.a- self.retain_model.a)#/torch.norm(self.retain_model.a)
        
        return eval_dict
    
    def eval_fun(self,p,q,loss_type='KL'): ## p is prediction

        p,q = p.clone().detach(), q.clone().detach()
        
        if loss_type == 'KL':   ##compute KL(q||p); this is the standard KL divergence (F.kl_div) where q is the true distribution and p is the estimated distribution
           
            return F.kl_div(p,q, reduction = 'batchmean',log_target = True).numpy()
        


        
        p,q = p.numpy(), q.numpy()
        
       
        if loss_type == 'l2':
            return np.mean(np.sum((p-q)**2,axis = -1)**0.5)
        elif loss_type == 'l1':
            return np.mean(np.sum(np.abs(p-q),axis = -1))
        elif loss_type == 'KS': ## compute the KS statistic (the maximum difference between the two cdfs)
            if self.K > 2:
                raise ValueError("KS statistic is only defined for 2 classes.")
            return ks_distance(p,q) ## only works for 2-d
    
    def test_eval_prob(self,X,y, data_type = 'retain'):

        if self.beta is None:
            return
        
        return 
    
        X = X.clone().detach()
        with torch.no_grad():
            p_retain = self.retain_model.forward(X)
            p_finetuned = self.finetuned_model.forward(X)
            p = self.forward(X)

            
            
            curr_ratio = p**self.beta/(p_finetuned**self.beta+p**self.beta)*2/self.beta
            retain_ratio = p_retain**self.beta/(p_finetuned**self.beta+p_retain**self.beta)*2/self.beta

            

            # curr_ratio =p/(p_finetuned+p)*2
            # retain_ratio = p_retain/(p_finetuned+p_retain)*2

            assert curr_ratio.shape == (X.shape[0],self.K)
            curr_ratio = torch.sum(curr_ratio*y,dim = -1)  ## y is one-hot matrix
            retain_ratio = torch.sum(y* retain_ratio,dim = -1)
            
         
            curr_ratio_mean = torch.mean(curr_ratio,dim = 0)
            curr_ratio_std = torch.std(curr_ratio,dim = 0)
            retain_ratio_mean = torch.mean(retain_ratio,dim = 0)
            retain_ratio_std = torch.std(retain_ratio,dim = 0)

            print(f"current:{curr_ratio_mean}_std_{curr_ratio_std}_retain:{retain_ratio_mean}_std_{retain_ratio_std}_{data_type}")
            

            






def empirical_cdf(sample):
       
    n = sample.shape[0]
    assert n>1
    sorted_sample = np.sort(sample)
    cdf_values = np.arange(1, n + 1) / n
    return sorted_sample, cdf_values

def ks_distance(sample1, sample2):
       
    # Compute empirical CDFs
    sorted_sample1, cdf1 = empirical_cdf(sample1)
    sorted_sample2, cdf2 = empirical_cdf(sample2)
    
    # Combine the samples and sort
    all_samples = np.concatenate([sorted_sample1, sorted_sample2])
    unique_samples = np.unique(all_samples)
    
    # Compute CDFs at combined points
    cdf1_at_combined = np.searchsorted(sorted_sample1, unique_samples, side='right') / sample1.size
    cdf2_at_combined = np.searchsorted(sorted_sample2, unique_samples, side='right') / sample2.size
    
    # Compute the KS statistic
    ks_statistic = np.max(np.abs(cdf1_at_combined - cdf2_at_combined))
        
    return ks_statistic



