import yaml
import os

PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "params.yaml")

def load_params(params_path=PARAMS_PATH):
    """Loads parameters from params.yaml."""
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

PARAMS = load_params()
