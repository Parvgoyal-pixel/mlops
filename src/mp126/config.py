"""
config.py — load params.yaml with yaml.safe_load.

Every other module imports get_params() from here so paths and
thresholds always come from one place and are never hard-coded.
"""

import pathlib
import yaml

# Project root is two levels above this file (src/mp126/config.py → root)
_ROOT = pathlib.Path(__file__).resolve().parents[2]


def get_params() -> dict:
    """Return the full params.yaml dictionary.

    Uses yaml.safe_load so only standard YAML types are loaded —
    no arbitrary Python objects, which keeps the file safe to share.
    """
    params_path = _ROOT / "params.yaml"
    with open(params_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def project_root() -> pathlib.Path:
    """Return the absolute path to the project root directory."""
    return _ROOT
