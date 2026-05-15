from __future__ import annotations

from datetime import datetime
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def make_run_dir(root: str | Path, stamp: datetime | None = None) -> Path:
    stamp = stamp or datetime.now()
    name = stamp.strftime("run-%Y%m%d-%H%M%S")
    run_dir = Path(root) / name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "plots").mkdir()
    (run_dir / "logs").mkdir()
    return run_dir
