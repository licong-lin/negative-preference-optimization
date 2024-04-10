# Negative Preference Optimization

This repository contains the code for the experiments in the paper "Negative Preference Optimization: From Catastrophic  Collapse to Effective Unlearning". 



### Abstract
>Large Language Models (LLMs) often memorize sensitive, private, or copyrighted data during pre-training. LLM unlearning aims to eliminate the influence of undesirable data from the pre-trained model while preserving the model's utilities on other tasks. Several practical methods have recently been proposed for LLM unlearning, mostly based on gradient ascent (GA) on the loss of undesirable data. However, on certain unlearning tasks, these methods either fail to effectively unlearn the target data or suffer from catastrophic collapse---a drastic degradation of the model's utilities. 

>In this paper, we propose *Negative Preference Optimization* (NPO), a simple alignment-inspired method that could efficiently and effectively unlearn a target dataset. We theoretically show that the progression toward catastrophic collapse by minimizing the NPO loss is exponentially slower than GA. Through experiments
on synthetic data and the benchmark TOFU dataset, we demonstrate that NPO-based methods achieve a better balance between unlearning the undesirable data and maintaining the model's utilities. 
We also observe that NPO-based methods generate more sensible outputs than GA-based methods, whose outputs are often gibberish.
Remarkably, on TOFU, NPO-based methods are the first to achieve reasonable unlearning results in forgetting 50\% (or more) of the training data, whereas existing methods already struggle with forgetting 10\% of training data.


