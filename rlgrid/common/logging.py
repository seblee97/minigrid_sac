from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional
from torch.utils.tensorboard import SummaryWriter

class _CSVAppender:
    def __init__(self, path: str):
        self.path = path
        self._file = open(path, "a", newline="", encoding="utf-8")
        self._writer: Optional[csv.DictWriter] = None
        self._fieldnames: Optional[list[str]] = None

    def write_row(self, row: Dict[str, Any]) -> None:
        if self._writer is None:
            self._fieldnames = list(row.keys())
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
            if self._file.tell() == 0:
                self._writer.writeheader()
        else:
            if any(k not in self._fieldnames for k in row.keys()):  # type: ignore[arg-type]
                self._upgrade_header(row)
                return
        self._writer.writerow(row)  # type: ignore[union-attr]
        self._file.flush()

    def _upgrade_header(self, row: Dict[str, Any]) -> None:
        # Minimal approach: keep existing header and drop new keys.
        # (Avoids rewriting large CSVs; extend if you want full schema evolution.)
        self._file.flush()
        self._file.close()
        with open(self.path, "r", encoding="utf-8") as rf:
            header = rf.readline().strip().split(",")
        self._file = open(self.path, "a", newline="", encoding="utf-8")
        self._fieldnames = header
        self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
        filtered = {k: row.get(k, None) for k in self._fieldnames}
        self._writer.writerow(filtered)
        self._file.flush()

    def close(self) -> None:
        try:
            self._file.flush()
            self._file.close()
        except Exception:
            pass

@dataclass
class LogConfig:
    log_dir: str = "runs"
    run_name: str = "run"
    tensorboard: bool = True

class LogWriter:
    """Simple, comprehensive logging:
    - progress.csv: scalar metrics keyed by training step
    - monitor.csv: episode return/length/time (SB3-like)
    - config.json: run config payload
    - tb/: TensorBoard scalars (optional)
    """
    def __init__(self, cfg: LogConfig, config_payload: Optional[Dict[str, Any]] = None):
        self.cfg = cfg
        self.run_path = os.path.join(cfg.log_dir, cfg.run_name)
        os.makedirs(self.run_path, exist_ok=True)

        self._progress = _CSVAppender(os.path.join(self.run_path, "progress.csv"))

        self._monitor_fp = open(os.path.join(self.run_path, "monitor.csv"), "a", newline="", encoding="utf-8")
        self._monitor_writer = csv.DictWriter(self._monitor_fp, fieldnames=["r", "l", "t", "step"])
        if self._monitor_fp.tell() == 0:
            self._monitor_fp.write("# RLGrid Monitor v0\n")
            self._monitor_writer.writeheader()

        self._tb = None
        if cfg.tensorboard:
            self._tb = SummaryWriter(log_dir=os.path.join(self.run_path, "tb"))

        self._t0 = time.time()
        if config_payload is not None:
            with open(os.path.join(self.run_path, "config.json"), "w", encoding="utf-8") as f:
                json.dump(config_payload, f, indent=2)

    def dump(self, step: int, metrics: Dict[str, Any]) -> None:
        row: Dict[str, Any] = {"step": int(step), "time_s": round(time.time() - self._t0, 3)}
        for k, v in metrics.items():
            if v is None:
                continue
            if isinstance(v, (int, float)):
                row[k] = float(v)
            else:
                row[k] = str(v)
        self._progress.write_row(row)

        if self._tb is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    try:
                        self._tb.add_scalar(k, float(v), global_step=step)
                    except Exception:
                        pass

    def log_episode(self, step: int, ep_return: float, ep_len: int) -> None:
        t = time.time() - self._t0
        self._monitor_writer.writerow({"r": float(ep_return), "l": int(ep_len), "t": round(t, 6), "step": int(step)})
        self._monitor_fp.flush()
        if self._tb is not None:
            try:
                self._tb.add_scalar("rollout/ep_rew", float(ep_return), global_step=step)
                self._tb.add_scalar("rollout/ep_len", float(ep_len), global_step=step)
            except Exception:
                pass

    def checkpoint_dir(self) -> str:
        p = os.path.join(self.run_path, "checkpoints")
        os.makedirs(p, exist_ok=True)
        return p

    def checkpoint_path(self, step: int, prefix: str = "ckpt") -> str:
        return os.path.join(self.checkpoint_dir(), f"{prefix}_step_{int(step)}.pt")


    def close(self) -> None:
        try:
            self._progress.close()
        except Exception:
            pass
        try:
            self._monitor_fp.flush()
            self._monitor_fp.close()
        except Exception:
            pass
        try:
            if self._tb is not None:
                self._tb.flush()
                self._tb.close()
        except Exception:
            pass
