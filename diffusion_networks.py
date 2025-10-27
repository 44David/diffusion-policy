'''
Networks used to build the diffusion policy.
'''

from typing import Union
from turtle import update
import torch.nn as nn
import torch 
import math 

class SinusodialPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, x):
        half_dim = self.dim // 2
        # equivalent to 1/10000^(2i/d), more efficient, and from the idea that a^x = e^(xln(a)) 
        embedding = torch.exp(torch.arange(half_dim) * -(math.log(10000) / (half_dim-1) )) 
        # broadcast multiplication to create a new matrix with shape(sequence length, dim//2)
        # we do [:, None] since x is a vector, and so to create a matrix from these two we add a column which has nothing in it (yet)
        embedding = x[:, None] * embedding[None, :]
        embedding = torch.cat((embedding.sin(), embedding.cos()), dim=-1)
        
        return embedding 
        
class DownSample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1) # kernel, stride, padding
    
    def forward(self, x):
        return self.conv(x)
    
class UpSample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1) # 

    def forward(self, x):
        return self.conv(x)
    

class ConvolutionBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, n_groups=8):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv1d(input_channels, output_channels, kernel_size, padding=kernel_size//2),
            nn.GroupNorm(n_groups, output_channels),
            nn.Mish()
        )
    
    def forward(self, x):
        return self.block(x)
    

# Conditional Residual = difference between observed and predicted value
# Essentially, apply FiLM to the ConvBlock
class ConditionalResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, conditional_dim, kernel_size=3, n_groups=8):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            ConvolutionBlock(input_channels, output_channels, kernel_size=kernel_size, n_groups=n_groups),
            ConvolutionBlock(output_channels, output_channels, kernel_size=kernel_size, n_groups=n_groups),
        ])
        
        conditional_channels = output_channels * 2
        self.output_channels = output_channels
        self.conditional_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(conditional_dim, conditional_channels),
            nn.Unflatten(-1, (-1, 1)) # The -1 here essentially tells pytorch, infer the dimension automatically, as we may now know the correct dim 
        )
        
        # nn.Identity() simply returns its input, doing nothing to it.
        self.residual_convolution = nn.Conv1d(input_channels, output_channels, 1) if input_channels != output_channels else nn.Identity() 
        
    def forward(self, x, conditional):
        ''''
        x = [batch_size, input_channels, horizon] <- horizon?
        conditional = [batch_size, conditional_dimension]
        
        return:
        
        out = [batch_size, output_channels, horizon]
        '''

        out = self.blocks[0](x) # do only the first conv block on input x
        embedding = self.conditional_encoder(conditional)
        embedding = embedding.reshape(embedding.shape[0], 2, self.output_channels, 1)
        scale = embedding[:, 0, ...] # the \gamma,  (scale) learned in the embedding
        bias = embedding[:,1,...] # and the /beta (bias)
        
        # FiLM equation gamma * x + bias
        out = scale * out + bias
        
        out = self.blocks[1](out) # now do the output conv block
        out += self.residual_convolution(x)
        return out
    
    
    
# The main denoising network (\epsilon)
class ConditionalUnet1D(nn.Module):
    def __init__(self, input_dim, global_conditional_dim, diffusion_step_embed_dim=256, down_dims=[256, 512, 1024], kernel_size=5, n_groups=8):
        '''
        input_dim = dimension of actions
        global_conditional_dim = dimension of the FiLM applied with the diffusion step embedding. (obs_horizon * obs_dim)
        diffusion_step_embed = dimension of pos encoding for each diff. iter k
        down_dims = channel size of each UNet level
        '''
    
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0] # start of UNet?
        
        # The diffusion step encoder tells the noise prediction network how much noise is in a sample
        diffusion_step_encoder = nn.Sequential(
            SinusodialPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim*4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim*4)
        )
        
        conditional_dim = diffusion_step_embed_dim + global_conditional_dim
        
        # creates a pairs of input and output channel, (input_channels, output_channels)
        in_out_channels = list(zip(all_dims[:-1], all_dims[1:]))
        
        middle_dim = all_dims[-1]
        
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock(
                middle_dim, middle_dim, conditional_dim=conditional_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
            ConditionalResidualBlock(
                middle_dim, middle_dim, conditional_dim=conditional_dim,
                kernel_size=kernel_size, n_groups=n_groups
            )                      
        ])
        
        '''
        Down and Up modules both refer to the structure of the UNet, 
        building the modules for both in this section 
        ''' 
        down_modules = nn.ModuleList([])
        for idx, (dim_in, dim_out) in enumerate(in_out_channels):
            last = idx >= (len(in_out_channels)-1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock(
                    dim_in, dim_out, conditional_dim=conditional_dim,
                    kernel_size=kernel_size, n_groups=n_groups
                ) ,
                ConditionalResidualBlock(
                    dim_out, dim_out, conditional_dim=conditional_dim,
                    kernel_size=kernel_size, n_groups=n_groups
                ) ,
                
                DownSample(dim_out) if not last else nn.Identity(),
            ]))
            
        up_modules = nn.ModuleList([])
        for idx, (dim_in, dim_out) in enumerate(reversed(in_out_channels[1:])):
            last = idx >= (len(in_out_channels)-1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock(
                    dim_out*2, dim_in, conditional_dim=conditional_dim,
                    kernel_size=kernel_size, n_groups=n_groups
                ),
                
                ConditionalResidualBlock(
                    dim_in, dim_in, conditional_dim=conditional_dim,
                    kernel_size=kernel_size, n_groups=n_groups
                ),
                
                UpSample(dim_in) if not last else nn.Identity()
            ]))
        
        final_conv = nn.Sequential(
            ConvolutionBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1)
        )
        
        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv
    
    
    def forward(self, sample: torch.Tensor, timesteps: Union[torch.Tensor, float, int], global_conditional=None):
        '''
        sample = (batch_size, timestep, input_dim)
        timesteps = (batch_size, ) or diffusion step
        global_condition = (batch_size, condition_dim)
        
        return:
        output = (batch_size, timestep, input_dim)
        '''
        
        # (batch_size, input_dim, timestep)        
        sample = sample.moveaxis(-1, -2)
        
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
            
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        
        timesteps = timesteps.expand(sample.shape[0])
        
        # diff step encoder generates features 
        global_feature = self.diffusion_step_encoder(timesteps)
            
        # only runs if doing conditional generation, which we should always be doing
        if global_conditional is not None:
            global_feature = torch.cat([global_feature, global_conditional], axis=-1) 
        
        
        
        '''
        Run the UNet
        '''
        h = []
        for _, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            sample = resnet(sample, global_feature)
            sample = resnet2(sample, global_feature)
            h.append(sample)
            sample = downsample(sample)
        
        for middle_module in self.mid_modules:
            sample =  middle_module(sample, global_feature)
            
        for _, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            sample = torch.cat(sample, h.pop(), dim=1)
            sample = resnet(sample, global_feature)
            sample = resnet2(sample, global_feature)
            sample = upsample(sample)
            
        
        sample = self.final_conv(sample)
        
        # return sample back to original shape of (batch_size, timestep, input_dim)
        sample = sample.moveaxis(-1, -2)
        
        return sample
        
            
        
        
    
    