import numpy as np
import minari

from diffusion_policy import DiffusionPolicy

def main():
    ds = minari.load_dataset('mujoco/reacher/medium-v0')
    env = ds.recover_environment()
    
    obs_dim = ds.observation_space.shape()
    action_dim = ds.action_space.shape()
        
    diff_policy = DiffusionPolicy()
    
    for episode in ds.iterate_episodes():
        diff_policy.loss(episode.observations, episode.actions)
        
if __name__ == "__main__":
    main()
