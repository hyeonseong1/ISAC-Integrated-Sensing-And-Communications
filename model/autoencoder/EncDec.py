import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, K, N):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(K, 2*N),
            nn.Tanh(),
            nn.Linear(2*N, 2*N),
            nn.Tanh(),
            nn.Linear(2*N, 2*N),
            nn.Tanh(),
            nn.Linear(2*N, 2*N),
            nn.BatchNorm1d(2*N)
        )
    def forward(self, m):
        return self.net(m)

class Decoder(nn.Module):
    def __init__(self, K, N):
        super().__init__()
        self.K   = K
        self.net = nn.Sequential(
            nn.Linear(2*N, 2*N),
            nn.Tanh(),
            nn.Linear(2*N, 2*N),
            nn.Tanh(),
            nn.Linear(2*N, 2*N),
            nn.Tanh(),
            nn.Linear(2*N, 2*K)
        )

    def forward(self, c_noisy):
        B = c_noisy.size(0)
        x = self.net(c_noisy)   # raw logits → [B, 2*K]
        x = x.view(B, self.K, 2)    # reshape → [B, K, 2]
        return F.softmax(x, dim=-1)