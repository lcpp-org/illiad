import json
from pathlib import Path

import numpy as np


def load_inputs_json(path, label="Inputs"):
    """Load a JSON object from path."""
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as stream:
            input_params = json.load(stream)
    except OSError as exc:
        raise SystemExit(f"Could not read {label} file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Could not parse {label} file {path}: {exc}") from exc

    if not isinstance(input_params, dict):
        raise SystemExit(f"{label} JSON must contain an object.")
    return input_params


def merge_input_params(defaults, overrides=None):
    """Return defaults updated by optional JSON-style overrides."""
    input_params = dict(defaults)
    if overrides:
        input_params.update(overrides)
    return input_params


def normalize_phi_gens(input_params):
    """Derive PHI_GENs from NPHI when not explicitly provided."""
    if "NPHI" not in input_params:
        return input_params

    nphi = int(input_params["NPHI"])
    if "PHI_GENs" not in input_params or input_params["PHI_GENs"] is None:
        input_params["PHI_GENs"] = np.linspace(360 // nphi, 360, nphi)
    else:
        input_params["PHI_GENs"] = np.asarray(input_params["PHI_GENs"])
    return input_params
