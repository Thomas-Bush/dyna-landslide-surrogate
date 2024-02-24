import sys
import os

def add_parent_dir_to_path():
    """Adds the parent directory to the system path."""
    parent_dir = os.path.dirname(os.getcwd())
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)