import torch 
import math
import torch.nn.functional as F
from observation_encoder import MLPEncoder
from diffusion_networks import ConditionalUnet1D
# from DDPM_scheduler import DDPMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import wandb 


class DiffusionPolicy:
    def __init__(self, state_dim, observation_embed_dim=768):
        # self.run = wandb.init(
        #     project="diffusion-policy",
            
        #     config={
        #         "architecture": "Diffusion Policy",
        #         "initial_lr": 1e-4
        #     }
        # )
        
        self.steps = 1000 # T
        self.batch_size = 32
        # self.DDPM_scheduler = DDPMScheduler()
        self.DDPM_scheduler = DDPMScheduler()
        self.obs_embedding = MLPEncoder(state_dim, observation_embed_dim)
        
    def loss(self, observations, actions):
        """
        observations: Tensor(batch_size, horizon)
        actions: Tensor(batch_size, trajectory, action_dim)
        """
        action_dim = actions.shape[2]
        
        obs_embed = self.obs_embedding(observations) # 512 being the global conditional dim computed in the UNet
        
        # t = torch.randint(1, self.steps+1, (self.batch_size,))
        t = torch.randint(1, self.steps, (self.batch_size,))
        
        # epsilon
        noise = torch.randn_like(torch.Tensor(actions))
        
        # the forward process in DDPM   
        noise_actions = self.DDPM_scheduler.add_noise(torch.Tensor(actions), noise, t)
        
        unet = ConditionalUnet1D(input_dim=action_dim, global_conditional_dim=768)
        noise_prediction = unet(noise_actions, t, global_conditional=obs_embed)
        
        loss = F.mse_loss(noise_prediction, noise)
        # self.run.log({"l2_loss": loss})
        
    def get_action_trajectory(self, actions, noise_actions):
        for k in range(self.steps):
            unet = ConditionalUnet1D(input_dim=actions.shape, global_conditional_dim=512)
    
            noise_prediction = unet(noise_actions, k, global_conditional_dim=512)

            action = self.DDPM_scheduler.sample(noise_prediction, noise_actions, k)

            