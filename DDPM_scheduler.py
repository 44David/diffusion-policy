import torch 
import math

def DPPM_Scheduler(x_0, batch_size):
    # diffusion steps
    T = 1000 
    t = torch.randint(1, T+1, (batch_size))
    # epsilon
    noise_vec = torch.randn_like(x_0)
    
    # start at b_0 = 10e-4, end at b_t, step by T. 
    beta_noise_scheduler = torch.linspace(10e-4, 0.02, T)
    
    alphas = 1 - beta_noise_scheduler
    
    return math.sqrt(alphas[t]) * x_0 + math.sqrt(1-alphas[t]*noise_vec) 