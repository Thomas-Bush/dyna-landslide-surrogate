import os
import numpy as np
import re
import random
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

        self.scaling_params = {
            'elevation': {'min': None, 'max': None},
            'thickness': {'min': None, 'max': None},
            'velocity': {'min': None, 'max': None}
        }

    def set_scaling_params(self, scaling_params):
        """
        Set the scaling parameters for each channel.
        Args:
            scaling_params (dict): A dictionary containing scaling parameters for each channel.
        """
        self.scaling_params = scaling_params

    def _scale_data(self, data, channel_name):
        """
        Scale the data for a given channel using min-max scaling.
        Args:
            data (np.ndarray): The data to scale.
            channel_name (str): The name of the channel to which the data belongs.
        Returns:
            np.ndarray: Scaled data.
        """
        min_val = self.scaling_params[channel_name]['min']
        max_val = self.scaling_params[channel_name]['max']
        if min_val is not None and max_val is not None and max_val != min_val:
            # Apply min-max scaling
            data = (data - min_val) / (max_val - min_val)
        
        return data

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

            def numerical_sort(file):
                """Helper function to sort files based on numerical value in filename."""
                numbers = re.findall(r'\d+', file)
                return int(numbers[-1]) if numbers else file

            thickness_files = sorted(
                [os.path.join(thickness_dir, f) for f in os.listdir(thickness_dir)],
                key=numerical_sort
            )
            velocity_files = sorted(
                [os.path.join(velocity_dir, f) for f in os.listdir(velocity_dir)],
                key=numerical_sort
            )

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

        # Define padding value and calculate padding needed
        padding_value = 0
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
            padded_thickness_sequence = np.array(thickness_sequence)
            padded_velocity_sequence = np.array(velocity_sequence)

        # Repeat the elevation for each time step
        sequence_elevation = np.repeat(elevation[np.newaxis, ...], self.sequence_length, axis=0)

        # Apply scaling to each channel sequence
        scaled_sequence_elevation = self._scale_data(sequence_elevation, 'elevation')
        scaled_padded_thickness_sequence = self._scale_data(padded_thickness_sequence, 'thickness')
        scaled_padded_velocity_sequence = self._scale_data(padded_velocity_sequence, 'velocity')

        # Stack the arrays along the first axis (after batch dimension) to create a sequence of 3-channel images
        sequence_data = np.stack((scaled_sequence_elevation, scaled_padded_thickness_sequence, scaled_padded_velocity_sequence), axis=1)

        # Load the target thickness and velocity for the next state, if available
        if self.sequence_length < len(thickness_files):
            target_thickness = np.load(thickness_files[self.sequence_length])
            target_velocity = np.load(velocity_files[self.sequence_length])
        else:
            # If there is no next state, use zeros
            target_thickness = np.zeros_like(elevation)
            target_velocity = np.zeros_like(elevation)

        # Apply scaling to each channel of the target data
        scaled_target_thickness = self._scale_data(target_thickness, 'thickness')
        scaled_target_velocity = self._scale_data(target_velocity, 'velocity')

        # Stack the elevation, target thickness, and target velocity to create the target 3-channel image
        target_data = np.stack((scaled_target_thickness, scaled_target_velocity), axis=-1)

        # Convert the arrays to numpy arrays and then to torch tensors
        sequence_data_tensor = torch.from_numpy(sequence_data).float()
        target_data_tensor = torch.from_numpy(target_data).float()

        return sequence_data_tensor, target_data_tensor

    
def compute_channel_scaling_params(dataloader):
    """Compute the min and max values for each channel in the dataset.

    Args:
        dataloader (DataLoader): The dataloader for the dataset.

    Returns:
        tuple: A tuple containing the dictionaries of minimum and maximum values for each channel.
    """
    # Initialize min and max values for each channel
    min_vals = {'elevation': float('inf'), 'thickness': float('inf'), 'velocity': float('inf')}
    max_vals = {'elevation': float('-inf'), 'thickness': float('-inf'), 'velocity': float('-inf')}

    for inputs, _ in dataloader:
        elevation, thickness, velocity = inputs[:, 0], inputs[:, 1], inputs[:, 2]
        min_vals['elevation'] = min(min_vals['elevation'], elevation.min().item())
        max_vals['elevation'] = max(max_vals['elevation'], elevation.max().item())
        min_vals['thickness'] = min(min_vals['thickness'], thickness.min().item())
        max_vals['thickness'] = max(max_vals['thickness'], thickness.max().item())
        min_vals['velocity'] = min(min_vals['velocity'], velocity.min().item())
        max_vals['velocity'] = max(max_vals['velocity'], velocity.max().item())

    return min_vals, max_vals

def set_channel_scaling_params_to_dataset(dataset, min_vals, max_vals):
    """Apply the computed min and max values for each channel to the dataset.

    Args:
        dataset (DebrisFlowDataset): The dataset to apply the scaling parameters to.
        min_vals (dict): A dictionary containing the minimum values for each channel.
        max_vals (dict): A dictionary containing the maximum values for each channel.
    """
    scaling_params = {
        'elevation': {'min': min_vals['elevation'], 'max': max_vals['elevation']},
        'thickness': {'min': min_vals['thickness'], 'max': max_vals['thickness']},
        'velocity': {'min': min_vals['velocity'], 'max': max_vals['velocity']}
    }
    dataset.set_scaling_params(scaling_params)



class Augmentation(object):
    def __init__(self, p_flip=0.5, p_rotate=0.5):
        """
        Initialize the transformation with probabilities for flipping and rotation.

        Args:
            p_flip (float): Probability of applying a horizontal or vertical flip.
            p_rotate (float): Probability of applying rotation.
        """
        self.p_flip = p_flip
        self.p_rotate = p_rotate

    def _flip(self, images, flip_type):
        """
        Flip the images horizontally or vertically.

        Args:
            images (torch.Tensor): A tensor of images (B, T, C, H, W).
            flip_type (str): Type of flip, 'horizontal' or 'vertical'.

        Returns:
            torch.Tensor: The flipped images.
        """
        if flip_type == 'horizontal':
            return images.flip(dims=[-1])
        elif flip_type == 'vertical':
            return images.flip(dims=[-2])
        return images

    def _rotate(self, images):
        """
        Rotate the images by 90 degrees.

        Args:
            images (torch.Tensor): A tensor of images (B, T, C, H, W).

        Returns:
            torch.Tensor: The rotated images.
        """
        return images.rot90(1, dims=[-2, -1])

    def __call__(self, inputs, targets):
        """
        Apply the transformation to inputs and targets.

        Args:
            inputs (torch.Tensor): The input tensor (B, T, C, H, W).
            targets (torch.Tensor): The target tensor (B, H, W, C').

        Returns:
            tuple: A tuple containing transformed inputs and targets.
        """
        # Convert targets to match input shape (B, C', H, W)
        targets = targets.permute(0, 3, 1, 2)

        # Apply horizontal or vertical flips with a certain probability
        if random.random() < self.p_flip:
            flip_type = 'horizontal' if random.random() < 0.5 else 'vertical'
            inputs = self._flip(inputs, flip_type)
            targets = self._flip(targets, flip_type)

        # Apply rotation with a certain probability
        if random.random() < self.p_rotate:
            inputs = self._rotate(inputs)
            targets = self._rotate(targets)

        # Convert targets back to original shape (B, H, W, C')
        targets = targets.permute(0, 2, 3, 1)

        return inputs, targets