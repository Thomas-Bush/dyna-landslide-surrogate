import os
import numpy as np
import torch
from torch.utils.data import Dataset

class DebrisFlowDataset(Dataset):
    """A PyTorch Dataset for loading debris flow data."""

    def __init__(self, main_dir, sequence_length):
        """
        Initialize the dataset with the main data directory and the sequence length.

        Args:
            main_dir (str): The main directory where the data is stored.
            sequence_length (int): The length of the sequences to be generated.
        """
        self.main_dir = main_dir
        self.sequence_length = sequence_length
        self.data_info = self._gather_data_info()

    def _gather_data_info(self):
        """
        Gather information about the data files from the main directory.

        Returns:
            list: A list of tuples with (model_id, elevation_file, thickness_files, velocity_files).
        """
        data_info = []
        for model_id in sorted(os.listdir(self.main_dir)):
            model_dir = os.path.join(self.main_dir, model_id, '04_FinalProcessedData')
            elevation_file = os.path.join(model_dir, 'elevation', f'{model_id}_elevation.npy')
            thickness_dir = os.path.join(model_dir, 'thickness')
            velocity_dir = os.path.join(model_dir, 'velocity')

            thickness_files = sorted([os.path.join(thickness_dir, f) for f in os.listdir(thickness_dir)])
            velocity_files = sorted([os.path.join(velocity_dir, f) for f in os.listdir(velocity_dir)])

            data_info.append((model_id, elevation_file, thickness_files, velocity_files))

        return data_info

    def __len__(self):
        """
        Denotes the total number of samples.
        """
        return len(self.data_info)

    def __getitem__(self, idx):
        model_id, elevation_file, thickness_files, velocity_files = self.data_info[idx]
        elevation = np.load(elevation_file)

        # Load thickness and velocity sequences
        thickness_sequence = [np.load(tf) for tf in thickness_files[:self.sequence_length]]
        velocity_sequence = [np.load(vf) for vf in velocity_files[:self.sequence_length]]

        padding_value = 0  # Define a padding value
        sequence_padding = self.sequence_length - len(thickness_sequence)

        # Pad sequences if they are shorter than the desired sequence length
        if sequence_padding > 0:
            thickness_padding = [(0, 0)] * elevation.ndim + [(0, sequence_padding)]
            velocity_padding = [(0, 0)] * elevation.ndim + [(0, sequence_padding)]
            
            padded_thickness_sequence = np.pad(np.array(thickness_sequence),
                                               thickness_padding,
                                               mode='constant',
                                               constant_values=padding_value)
            padded_velocity_sequence = np.pad(np.array(velocity_sequence),
                                              velocity_padding,
                                              mode='constant',
                                              constant_values=padding_value)
        else:
            # If sequences are already the desired length or longer, we can use them as is or trim them
            padded_thickness_sequence = np.array(thickness_sequence)
            padded_velocity_sequence = np.array(velocity_sequence)

        # Repeat the elevation for each time step
        sequence_elevation = np.repeat(elevation[np.newaxis, ...], self.sequence_length, axis=0)

        # Stack the arrays along the first axis (after batch dimension) to create a sequence of 3-channel images
        sequence_data = np.stack((sequence_elevation, padded_thickness_sequence, padded_velocity_sequence), axis=1)

        # The target is the state at the current index `i`
        target_thickness = padded_thickness_sequence[-1] if thickness_sequence else np.zeros_like(elevation)
        target_velocity = padded_velocity_sequence[-1] if velocity_sequence else np.zeros_like(elevation)
        target_data = np.stack((elevation, target_thickness, target_velocity), axis=-1)

        # Convert the lists to numpy arrays and then to torch tensors
        sequence_data_tensor = torch.from_numpy(sequence_data).float()
        target_data_tensor = torch.from_numpy(target_data).float()

        return sequence_data_tensor, target_data_tensor
