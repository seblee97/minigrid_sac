from __future__ import annotations
from typing import Any, Dict
import numpy as np
import torch

def to_tensor(x: Any, device: torch.device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device)
    return torch.tensor(x, device=device)

def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    # 1 - Var[y - yhat] / Var[y]
    var_y = torch.var(y_true)
    if var_y.item() < 1e-12:
        return float("nan")
    return float(1.0 - torch.var(y_true - y_pred) / var_y)
