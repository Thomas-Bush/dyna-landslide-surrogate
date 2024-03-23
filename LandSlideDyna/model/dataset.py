import os
import numpy as np
import re
import random
import torch
from torch.utils.data import Dataset

import time

# Using normlaisation of non-zeroes

class DebrisFlowDataset(Dataset):
    """A PyTorch Dataset for loading debris flow data."""

    def __init__(self, main_dir, input_array_size, sequence_length):
        """
        Initialize the dataset with the main data directory and the sequence length.

        Args:
            main_dir (str): The main directory where the data is stored.
            sequence_length (int): The length of the sequences to be generated.
        """
        self.main_dir = main_dir
        self.sequence_length = sequence_length
        self.input_array_size = input_array_size
        self.data_info = self._gather_data_info()

        self.scaling_params = {
            'elevation': {'median': None, 'mad': None},
            'thickness': {'median': None, 'mad': None},
            'velocity': {'median': None, 'mad': None}
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
        Scale the data for a given channel using non-zero median and MAD scaling.
        Args:
            data (np.ndarray): The data to scale.
            channel_name (str): The name of the channel to which the data belongs.
        Returns:
            np.ndarray: Scaled data.
        """
        median_val = self.scaling_params[channel_name]['median']
        mad_val = self.scaling_params[channel_name]['mad']
        if median_val is not None and mad_val != 0:
            # Apply scaling only to non-zero entries
            non_zero_mask = data != 0
            data[non_zero_mask] = (data[non_zero_mask] - median_val) / mad_val
        
        return data

    def _gather_data_info(self):
        data_info = []
        for model_id in sorted(os.listdir(self.main_dir)):
            model_dir = os.path.join(self.main_dir, model_id, f'04_FinalProcessedData_{self.input_array_size}')
            elevation_file = os.path.join(model_dir, 'elevation', f'{model_id}_elevation.npy')

            def numerical_sort(file):
                numbers = re.findall(r'\d+', file)
                return int(numbers[-1]) if numbers else file

            thickness_dir = os.path.join(model_dir, 'thickness')
            thickness_files = sorted(os.listdir(thickness_dir), key=numerical_sort)
            velocity_dir = os.path.join(model_dir, 'velocity')
            velocity_files = sorted(os.listdir(velocity_dir), key=numerical_sort)

            # Create all possible sequences and obtain target information
            for start_idx in range(len(thickness_files) - self.sequence_length):
                end_idx = start_idx + self.sequence_length
                next_idx = end_idx + 1
                sequence_thickness_files = thickness_files[start_idx:end_idx]
                sequence_velocity_files = velocity_files[start_idx:end_idx]

                # Determine the target file if it exists
                target_thickness_file = thickness_files[end_idx] if next_idx < len(thickness_files) else None
                target_velocity_file = velocity_files[end_idx] if next_idx < len(velocity_files) else None

                # Construct the full path to target files if they are not None
                target_info = {
                    'thickness_file': os.path.join(thickness_dir, target_thickness_file) if target_thickness_file else None,
                    'velocity_file': os.path.join(velocity_dir, target_velocity_file) if target_velocity_file else None
                }

                sequence_info = {
                    'elevation_file': elevation_file,
                    'thickness_files': [os.path.join(thickness_dir, tf) for tf in sequence_thickness_files],
                    'velocity_files': [os.path.join(velocity_dir, vf) for vf in sequence_velocity_files],
                    'target_info': target_info
                }
                data_info.append((model_id, sequence_info))

        return data_info

    def __len__(self):
        """
        Denotes the total number of samples.
        """
        return len(self.data_info)

    
    
    def __getitem__(self, idx):
        model_id, sequence_info = self.data_info[idx]
        elevation = np.load(sequence_info['elevation_file'])

        thickness_files = sequence_info['thickness_files']
        velocity_files = sequence_info['velocity_files']
        
        # Load thickness and velocity sequences
        thickness_sequence = [np.load(tf) for tf in thickness_files[:self.sequence_length]]
        velocity_sequence = [np.load(vf) for vf in velocity_files[:self.sequence_length]]

        # Pad sequences if necessary (this code assumes sequences always have the desired length)
        padded_thickness_sequence = np.array(thickness_sequence)
        padded_velocity_sequence = np.array(velocity_sequence)

        # Repeat the elevation for each time step
        sequence_elevation = np.repeat(elevation[np.newaxis, ...], self.sequence_length, axis=0)

        # Apply scaling to each channel sequence (assuming _scale_data is a method of your class)
        scaled_sequence_elevation = self._scale_data(sequence_elevation, 'elevation')
        scaled_padded_thickness_sequence = self._scale_data(padded_thickness_sequence, 'thickness')
        scaled_padded_velocity_sequence = self._scale_data(padded_velocity_sequence, 'velocity')

        # Stack the arrays along the first axis (after batch dimension) to create a sequence of 3-channel images
        sequence_data = np.stack((scaled_sequence_elevation, scaled_padded_thickness_sequence, scaled_padded_velocity_sequence), axis=1)

        # Load the target thickness and velocity for the next state, or use zeros if not available
        target_thickness_file = sequence_info['target_info']['thickness_file']
        if target_thickness_file is not None:
            target_thickness = np.load(target_thickness_file)
        else:
            target_thickness = np.zeros_like(elevation)

        target_velocity_file = sequence_info['target_info']['velocity_file']
        if target_velocity_file is not None:
            target_velocity = np.load(target_velocity_file)
        else:
            target_velocity = np.zeros_like(elevation)

        # Apply scaling to each channel of the target data
        scaled_target_thickness = self._scale_data(target_thickness, 'thickness')
        scaled_target_velocity = self._scale_data(target_velocity, 'velocity')

        # Stack the target thickness and target velocity to create the target 2-channel image
        target_data = np.stack((scaled_target_thickness, scaled_target_velocity), axis=-1)

        # Convert the arrays to numpy arrays and then to torch tensors
        sequence_data_tensor = torch.from_numpy(sequence_data).float()
        target_data_tensor = torch.from_numpy(target_data).float()

        return sequence_data_tensor, target_data_tensor


def compute_channel_scaling_params(elevation_files, thickness_files, velocity_files):
    """Compute the non-zero median and MAD for each channel using unique filenames.

    Args:
        elevation_files (set): A set of unique elevation filenames.
        thickness_files (set): A set of unique thickness filenames.
        velocity_files (set): A set of unique velocity filenames.

    Returns:
        tuple: A tuple containing the dictionaries of median and MAD values for each channel.
    """
    print(f"Processing {len(elevation_files)} elevation files, {len(thickness_files)} thickness files, and {len(velocity_files)} velocity files.")
    
    
    def compute_median_and_mad(data):
        # Filter out zero values and flatten the array
        non_zero_data = data[data != 0].flatten()
        # Calculate non-zero median and MAD
        median_val = np.median(non_zero_data)
        mad_val = np.median(np.abs(non_zero_data - median_val))
        return median_val, mad_val

    # Initialize dictionaries to store the median and MAD values
    median_vals = {}
    mad_vals = {}

    # Process elevation files

    start_time = time.time()

    elevation_data = []
    for file in elevation_files:
        print(f"Loading elevation file: {file}")  # Print the file path
        elevation_data.append(np.load(file))
    elevation_data = np.concatenate(elevation_data)
    print(elevation_data.shape)
    # elevation_data = np.concatenate([np.load(file) for file in elevation_files])
    median_vals['elevation'], mad_vals['elevation'] = compute_median_and_mad(elevation_data)
    del elevation_data  # Free up memory

    print(f"Processed elevation files in {time.time() - start_time:.2f} seconds.")

    # Process thickness files
    thickness_data = np.concatenate([np.load(file) for file in thickness_files])
    median_vals['thickness'], mad_vals['thickness'] = compute_median_and_mad(thickness_data)
    del thickness_data  # Free up memory

    # Process velocity files
    velocity_data = np.concatenate([np.load(file) for file in velocity_files])
    median_vals['velocity'], mad_vals['velocity'] = compute_median_and_mad(velocity_data)
    del velocity_data  # Free up memory

    return median_vals, mad_vals

# def compute_channel_scaling_params(dataloader):
#     """Compute the non-zero median and MAD for each channel in the dataset.

#     Args:
#         dataloader (DataLoader): The dataloader for the dataset.

#     Returns:
#         tuple: A tuple containing the dictionaries of median and MAD values for each channel.
#     """
#     # Initialize lists to collect non-zero data for each channel
#     elevation_data = []
#     thickness_data = []
#     velocity_data = []

#     for inputs, _ in dataloader:
#         elevation_data.extend(inputs[:, 0].numpy().flatten())
#         thickness_data.extend(inputs[:, 1].numpy().flatten())
#         velocity_data.extend(inputs[:, 2].numpy().flatten())

#     # Filter out zero values
#     elevation_data = list(filter(lambda x: x != 0, elevation_data))
#     thickness_data = list(filter(lambda x: x != 0, thickness_data))
#     velocity_data = list(filter(lambda x: x != 0, velocity_data))

#     # Calculate non-zero medians and MADs
#     median_vals = {
#         'elevation': np.median(elevation_data),
#         'thickness': np.median(thickness_data),
#         'velocity': np.median(velocity_data)
#     }
#     mad_vals = {
#         'elevation': np.median(np.abs(elevation_data - median_vals['elevation'])),
#         'thickness': np.median(np.abs(thickness_data - median_vals['thickness'])),
#         'velocity': np.median(np.abs(velocity_data - median_vals['velocity']))
#     }

#     return median_vals, mad_vals

def set_channel_scaling_params_to_dataset(dataset, median_vals, mad_vals):
    """Apply the computed non-zero median and MAD values for each channel to the dataset.

    Args:
        dataset (DebrisFlowDataset): The dataset to apply the scaling parameters to.
        median_vals (dict): A dictionary containing the non-zero median values for each channel.
        mad_vals (dict): A dictionary containing the MAD values for each channel.
    """
    scaling_params = {
        'elevation': {'median': median_vals['elevation'], 'mad': mad_vals['elevation']},
        'thickness': {'median': median_vals['thickness'], 'mad': mad_vals['thickness']},
        'velocity': {'median': median_vals['velocity'], 'mad': mad_vals['velocity']}
    }
    dataset.set_scaling_params(scaling_params)





















# # Using standardisation / z-score normalisaton
# # due to sparse data - use a median and interquartile range (IQR) for scaling to reduce the influence of outliers.

# class DebrisFlowDataset(Dataset):
#     """A PyTorch Dataset for loading debris flow data."""

#     def __init__(self, main_dir, input_array_size, sequence_length):
#         """
#         Initialize the dataset with the main data directory and the sequence length.

#         Args:
#             main_dir (str): The main directory where the data is stored.
#             sequence_length (int): The length of the sequences to be generated.
#         """
#         self.main_dir = main_dir
#         self.sequence_length = sequence_length
#         self.input_array_size = input_array_size
#         self.data_info = self._gather_data_info()

#         self.scaling_params = {
#             'elevation': {'median': None, 'iqr': None},
#             'thickness': {'median': None, 'iqr': None},
#             'velocity': {'median': None, 'iqr': None}
#         }       

#     def set_scaling_params(self, scaling_params):
#         """
#         Set the scaling parameters for each channel.
#         Args:
#             scaling_params (dict): A dictionary containing scaling parameters for each channel.
#         """
#         self.scaling_params = scaling_params

#     def _scale_data(self, data, channel_name):
#         """
#         Scale the data for a given channel using median and interquartile range scaling.
#         Args:
#             data (np.ndarray): The data to scale.
#             channel_name (str): The name of the channel to which the data belongs.
#         Returns:
#             np.ndarray: Scaled data.
#         """
#         median_val = self.scaling_params[channel_name]['median']
#         iqr_val = self.scaling_params[channel_name]['iqr']
#         if median_val is not None and iqr_val != 0:
#             # Apply median and IQR scaling
#             data = (data - median_val) / iqr_val
        
#         return data

#     def _gather_data_info(self):
#         data_info = []
#         for model_id in sorted(os.listdir(self.main_dir)):
#             model_dir = os.path.join(self.main_dir, model_id, f'04_FinalProcessedData_{self.input_array_size}')
#             elevation_file = os.path.join(model_dir, 'elevation', f'{model_id}_elevation.npy')

#             def numerical_sort(file):
#                 numbers = re.findall(r'\d+', file)
#                 return int(numbers[-1]) if numbers else file

#             thickness_dir = os.path.join(model_dir, 'thickness')
#             thickness_files = sorted(os.listdir(thickness_dir), key=numerical_sort)
#             velocity_dir = os.path.join(model_dir, 'velocity')
#             velocity_files = sorted(os.listdir(velocity_dir), key=numerical_sort)

#             # Create all possible sequences and obtain target information
#             for start_idx in range(len(thickness_files) - self.sequence_length):
#                 end_idx = start_idx + self.sequence_length
#                 next_idx = end_idx + 1
#                 sequence_thickness_files = thickness_files[start_idx:end_idx]
#                 sequence_velocity_files = velocity_files[start_idx:end_idx]

#                 # Determine the target file if it exists
#                 target_thickness_file = thickness_files[end_idx] if next_idx < len(thickness_files) else None
#                 target_velocity_file = velocity_files[end_idx] if next_idx < len(velocity_files) else None

#                 # Construct the full path to target files if they are not None
#                 target_info = {
#                     'thickness_file': os.path.join(thickness_dir, target_thickness_file) if target_thickness_file else None,
#                     'velocity_file': os.path.join(velocity_dir, target_velocity_file) if target_velocity_file else None
#                 }

#                 sequence_info = {
#                     'elevation_file': elevation_file,
#                     'thickness_files': [os.path.join(thickness_dir, tf) for tf in sequence_thickness_files],
#                     'velocity_files': [os.path.join(velocity_dir, vf) for vf in sequence_velocity_files],
#                     'target_info': target_info
#                 }
#                 data_info.append((model_id, sequence_info))

#         return data_info

#     def __len__(self):
#         """
#         Denotes the total number of samples.
#         """
#         return len(self.data_info)

    
    
#     def __getitem__(self, idx):
#         model_id, sequence_info = self.data_info[idx]
#         elevation = np.load(sequence_info['elevation_file'])

#         thickness_files = sequence_info['thickness_files']
#         velocity_files = sequence_info['velocity_files']
        
#         # Load thickness and velocity sequences
#         thickness_sequence = [np.load(tf) for tf in thickness_files[:self.sequence_length]]
#         velocity_sequence = [np.load(vf) for vf in velocity_files[:self.sequence_length]]

#         # Pad sequences if necessary (this code assumes sequences always have the desired length)
#         padded_thickness_sequence = np.array(thickness_sequence)
#         padded_velocity_sequence = np.array(velocity_sequence)

#         # Repeat the elevation for each time step
#         sequence_elevation = np.repeat(elevation[np.newaxis, ...], self.sequence_length, axis=0)

#         # Apply scaling to each channel sequence (assuming _scale_data is a method of your class)
#         scaled_sequence_elevation = self._scale_data(sequence_elevation, 'elevation')
#         scaled_padded_thickness_sequence = self._scale_data(padded_thickness_sequence, 'thickness')
#         scaled_padded_velocity_sequence = self._scale_data(padded_velocity_sequence, 'velocity')

#         # Stack the arrays along the first axis (after batch dimension) to create a sequence of 3-channel images
#         sequence_data = np.stack((scaled_sequence_elevation, scaled_padded_thickness_sequence, scaled_padded_velocity_sequence), axis=1)

#         # Load the target thickness and velocity for the next state, or use zeros if not available
#         target_thickness_file = sequence_info['target_info']['thickness_file']
#         if target_thickness_file is not None:
#             target_thickness = np.load(target_thickness_file)
#         else:
#             target_thickness = np.zeros_like(elevation)

#         target_velocity_file = sequence_info['target_info']['velocity_file']
#         if target_velocity_file is not None:
#             target_velocity = np.load(target_velocity_file)
#         else:
#             target_velocity = np.zeros_like(elevation)

#         # Apply scaling to each channel of the target data
#         scaled_target_thickness = self._scale_data(target_thickness, 'thickness')
#         scaled_target_velocity = self._scale_data(target_velocity, 'velocity')

#         # Stack the target thickness and target velocity to create the target 2-channel image
#         target_data = np.stack((scaled_target_thickness, scaled_target_velocity), axis=-1)

#         # Convert the arrays to numpy arrays and then to torch tensors
#         sequence_data_tensor = torch.from_numpy(sequence_data).float()
#         target_data_tensor = torch.from_numpy(target_data).float()

#         return sequence_data_tensor, target_data_tensor

  
# def compute_channel_scaling_params(dataloader):
#     """Compute the median and IQR for each channel in the dataset.

#     Args:
#         dataloader (DataLoader): The dataloader for the dataset.

#     Returns:
#         tuple: A tuple containing the dictionaries of median and IQR values for each channel.
#     """
#     # Initialize lists to collect data for each channel
#     elevation_data = []
#     thickness_data = []
#     velocity_data = []

#     for inputs, _ in dataloader:
#         elevation_data.append(inputs[:, 0].numpy())
#         thickness_data.append(inputs[:, 1].numpy())
#         velocity_data.append(inputs[:, 2].numpy())

#     # Flatten the lists and convert to numpy arrays
#     elevation_data = np.concatenate(elevation_data)
#     thickness_data = np.concatenate(thickness_data)
#     velocity_data = np.concatenate(velocity_data)

#     # Calculate medians and IQRs
#     median_vals = {
#         'elevation': np.median(elevation_data),
#         'thickness': np.median(thickness_data),
#         'velocity': np.median(velocity_data)
#     }
#     iqr_vals = {
#         'elevation': np.subtract(*np.percentile(elevation_data, [75, 25])),
#         'thickness': np.subtract(*np.percentile(thickness_data, [75, 25])),
#         'velocity': np.subtract(*np.percentile(velocity_data, [75, 25]))
#     }

#     return median_vals, iqr_vals

# def set_channel_scaling_params_to_dataset(dataset, median_vals, iqr_vals):
#     """Apply the computed median and IQR values for each channel to the dataset.

#     Args:
#         dataset (DebrisFlowDataset): The dataset to apply the scaling parameters to.
#         median_vals (dict): A dictionary containing the median values for each channel.
#         iqr_vals (dict): A dictionary containing the IQR values for each channel.
#     """
#     scaling_params = {
#         'elevation': {'median': median_vals['elevation'], 'iqr': iqr_vals['elevation']},
#         'thickness': {'median': median_vals['thickness'], 'iqr': iqr_vals['thickness']},
#         'velocity': {'median': median_vals['velocity'], 'iqr': iqr_vals['velocity']}
#     }
#     dataset.set_scaling_params(scaling_params)













# # USING MIN-MAX SCALING

# class DebrisFlowDataset(Dataset):
#     """A PyTorch Dataset for loading debris flow data."""

#     def __init__(self, main_dir, input_array_size, sequence_length):
#         """
#         Initialize the dataset with the main data directory and the sequence length.

#         Args:
#             main_dir (str): The main directory where the data is stored.
#             sequence_length (int): The length of the sequences to be generated.
#         """
#         self.main_dir = main_dir
#         self.sequence_length = sequence_length
#         self.input_array_size = input_array_size
#         self.data_info = self._gather_data_info()

#         self.scaling_params = {
#             'elevation': {'min': None, 'max': None},
#             'thickness': {'min': None, 'max': None},
#             'velocity': {'min': None, 'max': None}
#         }

#     def set_scaling_params(self, scaling_params):
#         """
#         Set the scaling parameters for each channel.
#         Args:
#             scaling_params (dict): A dictionary containing scaling parameters for each channel.
#         """
#         self.scaling_params = scaling_params

#     def _scale_data(self, data, channel_name):
#         """
#         Scale the data for a given channel using min-max scaling.
#         Args:
#             data (np.ndarray): The data to scale.
#             channel_name (str): The name of the channel to which the data belongs.
#         Returns:
#             np.ndarray: Scaled data.
#         """
#         min_val = self.scaling_params[channel_name]['min']
#         max_val = self.scaling_params[channel_name]['max']
#         if min_val is not None and max_val is not None and max_val != min_val:
#             # Apply min-max scaling
#             data = (data - min_val) / (max_val - min_val)
        
#         return data

#     def _gather_data_info(self):
#         data_info = []
#         for model_id in sorted(os.listdir(self.main_dir)):
#             model_dir = os.path.join(self.main_dir, model_id, f'04_FinalProcessedData_{self.input_array_size}')
#             elevation_file = os.path.join(model_dir, 'elevation', f'{model_id}_elevation.npy')

#             def numerical_sort(file):
#                 numbers = re.findall(r'\d+', file)
#                 return int(numbers[-1]) if numbers else file

#             thickness_dir = os.path.join(model_dir, 'thickness')
#             thickness_files = sorted(os.listdir(thickness_dir), key=numerical_sort)
#             velocity_dir = os.path.join(model_dir, 'velocity')
#             velocity_files = sorted(os.listdir(velocity_dir), key=numerical_sort)

#             # Create all possible sequences and obtain target information
#             for start_idx in range(len(thickness_files) - self.sequence_length):
#                 end_idx = start_idx + self.sequence_length
#                 next_idx = end_idx + 1
#                 sequence_thickness_files = thickness_files[start_idx:end_idx]
#                 sequence_velocity_files = velocity_files[start_idx:end_idx]

#                 # Determine the target file if it exists
#                 target_thickness_file = thickness_files[end_idx] if next_idx < len(thickness_files) else None
#                 target_velocity_file = velocity_files[end_idx] if next_idx < len(velocity_files) else None

#                 # Construct the full path to target files if they are not None
#                 target_info = {
#                     'thickness_file': os.path.join(thickness_dir, target_thickness_file) if target_thickness_file else None,
#                     'velocity_file': os.path.join(velocity_dir, target_velocity_file) if target_velocity_file else None
#                 }

#                 sequence_info = {
#                     'elevation_file': elevation_file,
#                     'thickness_files': [os.path.join(thickness_dir, tf) for tf in sequence_thickness_files],
#                     'velocity_files': [os.path.join(velocity_dir, vf) for vf in sequence_velocity_files],
#                     'target_info': target_info
#                 }
#                 data_info.append((model_id, sequence_info))

#         return data_info

#     def __len__(self):
#         """
#         Denotes the total number of samples.
#         """
#         return len(self.data_info)

    
    
#     def __getitem__(self, idx):
#         model_id, sequence_info = self.data_info[idx]
#         elevation = np.load(sequence_info['elevation_file'])

#         thickness_files = sequence_info['thickness_files']
#         velocity_files = sequence_info['velocity_files']
        
#         # Load thickness and velocity sequences
#         thickness_sequence = [np.load(tf) for tf in thickness_files[:self.sequence_length]]
#         velocity_sequence = [np.load(vf) for vf in velocity_files[:self.sequence_length]]

#         # Pad sequences if necessary (this code assumes sequences always have the desired length)
#         padded_thickness_sequence = np.array(thickness_sequence)
#         padded_velocity_sequence = np.array(velocity_sequence)

#         # Repeat the elevation for each time step
#         sequence_elevation = np.repeat(elevation[np.newaxis, ...], self.sequence_length, axis=0)

#         # Apply scaling to each channel sequence (assuming _scale_data is a method of your class)
#         scaled_sequence_elevation = self._scale_data(sequence_elevation, 'elevation')
#         scaled_padded_thickness_sequence = self._scale_data(padded_thickness_sequence, 'thickness')
#         scaled_padded_velocity_sequence = self._scale_data(padded_velocity_sequence, 'velocity')

#         # Stack the arrays along the first axis (after batch dimension) to create a sequence of 3-channel images
#         sequence_data = np.stack((scaled_sequence_elevation, scaled_padded_thickness_sequence, scaled_padded_velocity_sequence), axis=1)

#         # Load the target thickness and velocity for the next state, or use zeros if not available
#         target_thickness_file = sequence_info['target_info']['thickness_file']
#         if target_thickness_file is not None:
#             target_thickness = np.load(target_thickness_file)
#         else:
#             target_thickness = np.zeros_like(elevation)

#         target_velocity_file = sequence_info['target_info']['velocity_file']
#         if target_velocity_file is not None:
#             target_velocity = np.load(target_velocity_file)
#         else:
#             target_velocity = np.zeros_like(elevation)

#         # Apply scaling to each channel of the target data
#         scaled_target_thickness = self._scale_data(target_thickness, 'thickness')
#         scaled_target_velocity = self._scale_data(target_velocity, 'velocity')

#         # Stack the target thickness and target velocity to create the target 2-channel image
#         target_data = np.stack((scaled_target_thickness, scaled_target_velocity), axis=-1)

#         # Convert the arrays to numpy arrays and then to torch tensors
#         sequence_data_tensor = torch.from_numpy(sequence_data).float()
#         target_data_tensor = torch.from_numpy(target_data).float()

#         return sequence_data_tensor, target_data_tensor

  
# def compute_channel_scaling_params(dataloader):
#     """Compute the min and max values for each channel in the dataset.

#     Args:
#         dataloader (DataLoader): The dataloader for the dataset.

#     Returns:
#         tuple: A tuple containing the dictionaries of minimum and maximum values for each channel.
#     """
#     # Initialize min and max values for each channel
#     min_vals = {'elevation': float('inf'), 'thickness': float('inf'), 'velocity': float('inf')}
#     max_vals = {'elevation': float('-inf'), 'thickness': float('-inf'), 'velocity': float('-inf')}

#     for inputs, _ in dataloader:
#         elevation, thickness, velocity = inputs[:, 0], inputs[:, 1], inputs[:, 2]
#         min_vals['elevation'] = min(min_vals['elevation'], elevation.min().item())
#         max_vals['elevation'] = max(max_vals['elevation'], elevation.max().item())
#         min_vals['thickness'] = min(min_vals['thickness'], thickness.min().item())
#         max_vals['thickness'] = max(max_vals['thickness'], thickness.max().item())
#         min_vals['velocity'] = min(min_vals['velocity'], velocity.min().item())
#         max_vals['velocity'] = max(max_vals['velocity'], velocity.max().item())

#     return min_vals, max_vals

# def set_channel_scaling_params_to_dataset(dataset, min_vals, max_vals):
#     """Apply the computed min and max values for each channel to the dataset.

#     Args:
#         dataset (DebrisFlowDataset): The dataset to apply the scaling parameters to.
#         min_vals (dict): A dictionary containing the minimum values for each channel.
#         max_vals (dict): A dictionary containing the maximum values for each channel.
#     """
#     scaling_params = {
#         'elevation': {'min': min_vals['elevation'], 'max': max_vals['elevation']},
#         'thickness': {'min': min_vals['thickness'], 'max': max_vals['thickness']},
#         'velocity': {'min': min_vals['velocity'], 'max': max_vals['velocity']}
#     }
#     dataset.set_scaling_params(scaling_params)



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


    
 