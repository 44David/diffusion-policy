
import torch 
import math
import torch.nn as nn
import observation_encoder
from diffusion_networks import ConditionalUnet1D
import DDPM_scheduler

class DiffusionPolicy:
    def __init__(self):
        self.obs_encoder = observation_encoder
        self.noise_pred_net = ConditionalUnet1D
        self.DDPM_scheduler = DDPM_scheduler
    

