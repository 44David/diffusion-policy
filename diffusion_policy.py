
import torch 
from torch.utils.tensorboard import SummaryWriter
import math
import torch.nn.functional as F
import observation_encoder 
from diffusion_networks import ConditionalUnet1D
from DDPM_scheduler import DDPMScheduler
import wandb 


class DiffusionPolicy:
    def __init__(self):
        self.run = wandb.init(
            project="diffusion-policy",
            
            config={
                "architecture": "Diffusion Policy",
                "initial_lr": 1e-4
            }
        )
        
        self.steps = 1000 # T
        self.batch_size = 32
        self.DDPM_scheduler = DDPMScheduler()
        
    def loss(self, observations, actions):
        obs_embedding =  observation_encoder.MLPEncoder(observations, 512) # 512 being the global conditional dim computed in the UNet
        
        t = torch.randint(1, self.steps+1, (self.batch_size,))
        
        # epsilon
        noise_vec = torch.randn_like(actions)
        noise_actions = self.DDPM_scheduler.add_noise(actions, t, noise_vec)
        
        unet = ConditionalUnet1D(input_dim=actions.shape(), global_conditional_dim=512)
        
        noise_prediction = unet(noise_actions, t, global_conditional_dim=512)
        
        loss = F.mse_loss(noise_prediction, noise_vec)
        
        self.run.log({"l2_loss": loss})
        
    def get_action_trajectory(self, actions, noise_actions):
        for k in range(self.steps):
            unet = ConditionalUnet1D(input_dim=actions.shape(), global_conditional_dim=512)
    
            noise_prediction = unet(noise_actions, k, global_conditional_dim=512)

            action = self.DDPM_scheduler.sample(noise_prediction, noise_actions, k)

            