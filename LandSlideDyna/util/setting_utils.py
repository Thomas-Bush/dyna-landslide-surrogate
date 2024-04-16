import numpy as np
import torch
import random

def set_seed(seed_value):
    """
    Set seed for reproducibility across multiple libraries.

    This function sets the random seed for numpy, torch, and the built-in random module
    to ensure that experiments are reproducible. It also configures torch to behave
    more deterministically when running on a GPU.

    Args:
        seed_value (int): The seed value to use for random number generators.
    """
    np.random.seed(seed_value)  # Set seed for NumPy
    torch.manual_seed(seed_value)  # Set seed for PyTorch
    random.seed(seed_value)  # Set seed for Python's built-in random module
    
    # Configure PyTorch for more deterministic behavior on GPU (if used)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)