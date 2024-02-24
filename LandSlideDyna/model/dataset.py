import torch
from torch.utils.data import Dataset, random_split
import numpy as np
import os

class DebrisFlowDataset(Dataset):
    def __init__(self, data_directory):
        """
        Custom dataset to handle variable-length sequences for each model.
        
        Args:
            data_directory (str): The path to the directory containing the models' data.
        """
        self.data_directory = data_directory
        self.model_directories = [os.path.join(data_directory, name)
                                  for name in sorted(os.listdir(data_directory))
                                  if os.path.isdir(os.path.join(data_directory, name))]
    
    def __len__(self):
        return len(self.model_directories)
    
# TODO - check the file logic below


    def __getitem__(self, idx):
        # Load the data for a single model based on the index
        model_dir = self.model_directories[idx]
        elevation = np.load(os.path.join(model_dir, 'elevation.npy'))  # Same for all states
        
        # Lists to hold the velocity and thickness arrays for each state
        velocity_list = []
        thickness_list = []
        
        ######################
        # NEED TO CHECK THIS #
        ######################

        state_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.npy') and 'velocity' in f])
        
        for state_file in state_files:
            state_idx = state_file.split('_')[2]  # Extracting the state index from the file name
            velocity = np.load(os.path.join(model_dir, state_file))
            thickness = np.load(os.path.join(model_dir, f"model_{state_idx}_thickness.npy"))
            
            velocity_list.append(velocity)
            thickness_list.append(thickness)
        
        # Convert the lists to numpy arrays and stack along a new dimension
        velocity_array = np.stack(velocity_list, axis=0)
        thickness_array = np.stack(thickness_list, axis=0)
        
        # Stack the channels to create a multi-channel image for each time step
        # The resulting shape will be (time_steps, channels, height, width)
        elevation_array = np.repeat(elevation[np.newaxis, :, :], velocity_array.shape[0], axis=0)
        model_states = np.stack((elevation_array, velocity_array, thickness_array), axis=1)
        
        # Convert the numpy arrays to PyTorch tensors
        model_states_tensor = torch.from_numpy(model_states).float()
        
        return model_states_tensor


def collate_fn(batch):
    """
    Custom collate function to handle batches of variable-length sequences.
    
    Args:
        batch (list): A list of tensors from the dataset, each representing a model's states.
        
    Returns:
        torch.Tensor: A batch of sequences padded to the length of the longest sequence.
    """
    # Find the longest sequence in the batch
    max_length = max(seq.size(0) for seq in batch)
    num_channels = batch[0].size(1)
    height = batch[0].size(2)
    width = batch[0].size(3)
    
    # Pad all sequences to the length of the longest sequence
    padded_batch = torch.zeros(len(batch), max_length, num_channels, height, width)
    for i, seq in enumerate(batch):
        padded_batch[i, :seq.size(0), :, :, :] = seq
    
    return padded_batch

def split_dataset(dataset, train_frac=0.7, val_frac=0.15, test_frac=0.15):
    """
    Splits the dataset into training, validation, and test sets.
    
    Args:
        dataset (Dataset): The original dataset to be split.
        train_frac (float): The fraction of the dataset to be used as the training set.
        val_frac (float): The fraction of the dataset to be used as the validation set.
        test_frac (float): The fraction of the dataset to be used as the test set.
    
    Returns:
        Subset: The training subset.
        Subset: The validation subset.
        Subset: The test subset.
    """
    # Ensure the fractions add up to 1
    assert train_frac + val_frac + test_frac == 1, "The fractions must add up to 1."
    
    # Calculate the lengths of each split
    total_size = len(dataset)
    train_size = int(total_size * train_frac)
    val_size = int(total_size * val_frac)
    test_size = total_size - train_size - val_size
    
    # Randomly split the dataset into training, validation, and test subsets
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    return train_dataset, val_dataset, test_dataset

# Example usage:
# dataset = VariableLengthDataset(data_directory='path/to/model_data')
# data_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)