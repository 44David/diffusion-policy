import torch 
import math

class DDPMScheduler:
    def __init__(self, T=1000):
        # start at b_0 = 10e-4, end at b_t, step by T. 
        self.beta_noise_scheduler = torch.linspace(10e-4, 0.02, T)
        
        self.alphas = 1 - self.beta_noise_scheduler
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
    
    # add sample to clean data
    def add_noise(self, x_0, t, noise):
        """
        x_0: clean data, sampled from dataset (batch, horizon, action_dim)
        t: timestep (batch, )
        noise: sampled noise (batch, horizon, action_dim)
        """
        
        return math.sqrt(self.alphas_bar[t]) * x_0 + math.sqrt(1-self.alphas_bar[t]*noise) 


    # inverse diffusion, remove noise from noise sample
    def sample(self, noise_prediction, noisy_sample, timestep):
        z = torch.randn_like(noisy_sample)
        
        for t in range(timestep):
            noisy_sample = (1/math.sqrt(self.alphas[t])) *  \
                (noisy_sample - (1 - self.alphas_bar[t] / math.sqrt(1 - self.alphas_bar)) * noise_prediction) \
                + math.sqrt(self.beta_noise_scheduler[t]) * z 
                
        # x_t, (noisy) is now x_0, a clean data sample   
        return noisy_sample