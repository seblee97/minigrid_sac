from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class Logger:
    scalars: Dict[str, List[float]] = field(default_factory=dict)

    def log(self, key: str, value: float) -> None:
        self.scalars.setdefault(key, []).append(float(value))

    def summary(self) -> Dict[str, float]:
        return {k: (sum(vs) / max(1, len(vs))) for k, vs in self.scalars.items()}

    def reset(self) -> None:
        self.scalars.clear()
