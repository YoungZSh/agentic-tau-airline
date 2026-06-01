"""Centralized path resolution — eliminates hard-coded machine paths.

All machine-specific locations come from environment variables (loaded from
`.env` via python-dotenv). Change them in one place per machine; verl/Hydra
configs reference the same variables through `${oc.env:...}`.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# paths.py = src/tau2_airline_verl/utils/paths.py -> parents[3] is the repo root.
REPO_ROOT = Path(__file__).resolve().parents[3]


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return (Path(raw) if raw else default).expanduser()


def qwen3_path() -> Path:
    """Local Qwen3-8B checkpoint dir (policy model, stage 1+)."""
    return _env_path("QWEN3_8B_PATH", REPO_ROOT / "ckpts" / "Qwen3-8B")


def tau2_dir() -> Path:
    """tau2-bench source (git submodule under third_party/)."""
    return _env_path("TAU2_DIR", REPO_ROOT / "third_party" / "tau2-bench")


def verl_dir() -> Path:
    """verl source (git submodule under third_party/)."""
    return _env_path("VERL_DIR", REPO_ROOT / "third_party" / "verl")


def output_dir() -> Path:
    """Experiment outputs (checkpoints / logs / eval)."""
    return _env_path("OUTPUT_DIR", REPO_ROOT / "outputs")


def trajectories_dir() -> Path:
    """Rollout trajectory JSONL (plan §10)."""
    return _env_path("TRAJ_DIR", REPO_ROOT / "data" / "trajectories")


__all__ = [
    "REPO_ROOT",
    "qwen3_path",
    "tau2_dir",
    "verl_dir",
    "output_dir",
    "trajectories_dir",
]
