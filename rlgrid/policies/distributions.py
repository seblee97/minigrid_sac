from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F

@dataclass
class CategoricalDist:
    logits: torch.Tensor

    def sample(self) -> torch.Tensor:
        return torch.distributions.Categorical(logits=self.logits).sample()

    def mode(self) -> torch.Tensor:
        return torch.argmax(self.logits, dim=-1)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return torch.distributions.Categorical(logits=self.logits).log_prob(actions)

    def entropy(self) -> torch.Tensor:
        return torch.distributions.Categorical(logits=self.logits).entropy()

    def kl_divergence(self, other: 'CategoricalDist') -> torch.Tensor:
        """Compute KL divergence between this distribution and another."""
        return torch.distributions.kl.kl_divergence(
            torch.distributions.Categorical(logits=self.logits),
            torch.distributions.Categorical(logits=other.logits)
        )

def huber_quantile_loss(td_errors: torch.Tensor, taus: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
    '''
    Quantile regression Huber loss for QRDQN.
    td_errors: (B, N, N') where N = n_quantiles (current), N' = n_quantiles (target)
    taus: (N,) quantile midpoints in (0,1)
    '''
    # Huber
    abs_err = torch.abs(td_errors)
    huber = torch.where(abs_err <= kappa, 0.5 * td_errors**2, kappa * (abs_err - 0.5 * kappa))
    # Quantile weights
    # indicator: td_error < 0
    indicator = (td_errors.detach() < 0).float()
    quantile_weight = torch.abs(taus.view(1, -1, 1) - indicator)
    loss = (quantile_weight * huber).mean()
    return loss
