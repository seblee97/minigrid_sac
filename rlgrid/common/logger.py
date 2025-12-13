from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List
import time

@dataclass
class Logger:
    start_time: float = field(default_factory=time.time)
    scalars: Dict[str, List[float]] = field(default_factory=dict)

    def log(self, key: str, value: float) -> None:
        self.scalars.setdefault(key, []).append(float(value))

    def summary(self) -> Dict[str, float]:
        out = {}
        for k, vs in self.scalars.items():
            out[k] = sum(vs) / max(1, len(vs))
        return out

    def reset(self) -> None:
        self.scalars.clear()
