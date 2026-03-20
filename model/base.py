from torch import nn


class TrajectoryModel(nn.Module):
    def __init__(self, ac_dim, max_len=None):
        super().__init__()
        
        self.ac_dim = ac_dim
        self.max_len = max_len
    
    def forward(self, observations, actions, rewards, returns_to_go, timesteps, mask=None):
        raise NotImplementedError
    
    # def get_action(self, observations, actions, rewards, returns_to_go, timesteps):
    #     return torch.zeros_like(actions[-1])  # latest action
