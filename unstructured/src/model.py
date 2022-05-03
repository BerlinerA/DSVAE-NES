import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Gumbel


class VAE(torch.nn.Module):
    def __init__(self, input_size, hidden_dim, latent_dim, nes=False, log_prob_bound=100):
        super().__init__()

        self.latent_dim = latent_dim
        self.log_prob_bound = log_prob_bound

        # whether to optimize via NES
        self.nes = nes

        # initialize encoder layers
        self.enc_fc1 = nn.Linear(input_size, hidden_dim)
        self.enc_fc2 = nn.Linear(hidden_dim, latent_dim)
        # initialize decoder layers
        self.dec_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.dec_fc2 = nn.Linear(hidden_dim, input_size)

    def _encode(self, x):
        x = self.enc_fc1(x)
        x = F.relu(x)
        return self.enc_fc2(x)

    def _decode(self, z):
        x = self.dec_fc1(z)
        x = F.relu(x)
        x = self.dec_fc2(x)
        x = torch.sigmoid(x)
        low_prob_bound = math.exp(-self.log_prob_bound) if self.training else 0.
        return torch.clamp(x, min=low_prob_bound, max=1. - low_prob_bound)

    def forward(self, x, tau=1.):
        x = torch.flatten(x, start_dim=1)
        logits = self._encode(x).view(x.size(0), -1, self.latent_dim)
        gumbels = Gumbel(loc=0, scale=1).sample(logits.shape).to(logits.device)
        if not self.training or self.nes:
            top1_indices = torch.argmax(logits if not self.training else logits + gumbels, dim=-1, keepdim=True)
            z = torch.zeros_like(logits).scatter(-1, top1_indices, 1.0)
        else:
            z = torch.nn.functional.softmax((logits + gumbels) / tau, dim=-1)

        z = z.view(x.size(0), -1)

        return [self._decode(z), logits]
