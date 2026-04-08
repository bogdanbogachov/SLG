"""Project-wide RNG seeding (Python, NumPy, PyTorch)."""
from transformers import set_seed as hf_set_seed


def set_project_seed(seed: int) -> None:
    """Set seeds for `random`, NumPy, and PyTorch (including CUDA when available)."""
    hf_set_seed(seed)
