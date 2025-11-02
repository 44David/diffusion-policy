import minari
from torch.utils.data import DataLoader, Dataset
from diffusion_policy import DiffusionPolicy


class TrajectoryDataset(Dataset):
    def __init__(self, data, horizon=16):
        self.horizon = horizon
        self.trajectories = []
        
        for episode in data.iterate_episodes():
            
            for i in range(len(episode.actions) - horizon+1):
                self.trajectories.append({
                    'observations': episode.observations[i],
                    'actions': episode.actions[i:i+horizon],
                })
        
        
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        return self.trajectories[idx]

def main():
    """
    action space: (-1, 1, (2, ), float32)
    action_space[0]: torque to apply on first hinge (joint0)
    action_space[1]: torque for second hinge (joint1)
    """
    
    ds = minari.load_dataset('mujoco/reacher/medium-v0')
    
    dataset = DataLoader(
        dataset=TrajectoryDataset(ds),
        batch_size=32,
        shuffle=True
    )    

    obs_dim = ds.observation_space.shape[0]
    diff_policy = DiffusionPolicy(obs_dim)
    
    for batch in dataset:
        actions = batch["actions"]
        observations = batch["observations"]
        diff_policy.loss(observations, actions)    

    

if __name__ == "__main__":
    main()
