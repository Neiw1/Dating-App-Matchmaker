import torch.nn as nn
import torch.nn.functional as F
#from copy import deepcopy
class user_AE(nn.Module):
    def __init__(self, input_dim=75, latent_dim=8):
        super(user_AE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.Sigmoid()  # Use Tanh or Sigmoid based on your data
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
class MF(nn.Module):
    def __init__(self, name, AE):
        super(MF, self).__init__()
        self.name = name
        self.user_emb = AE.encoder
        self.item_emb = AE.encoder

    def forward(self, u, v):
        u = F.normalize(self.user_emb(u), p=2, dim=1)  # unit vector (L2 norm = 1)
        v = F.normalize(self.item_emb(v), p=2, dim=1)
        cos_sim = (u * v).sum(1)
        return (cos_sim + 1) / 2  # maps from [-1, 1] to [0, 1]