import torch.nn as nn


class MLPEncoder(nn.Module):
    '''
    Simple MLP based encoder, used when dealing with state based observations
    '''
    def __init__(self, state_dim, observation_embed_dim ):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, observation_embed_dim)
        self.relu = nn.ReLU
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        
        return x
        