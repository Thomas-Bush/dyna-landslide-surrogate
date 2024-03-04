import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class DebrisFlowDataset(Dataset):
    def __init__(self, main_dir, transform=None):
        """
        Initializes the dataset by scanning the main directory for subdirectories
        that contain the model data.

        Args:
            main_dir (str): Path to the main directory containing the model subdirectories.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.main_dir = main_dir
        self.transform = transform
        self.model_dirs = sorted([d for d in os.listdir(main_dir)
                                  if os.path.isdir(os.path.join(main_dir, d))])

        # Generate a list of tuples (model_id, state_index)
        self.data_info = []
        for model_id in self.model_dirs:
            state_files = sorted(os.listdir(os.path.join(main_dir, model_id, 'thickness')))
            self.data_info.extend([(model_id, i) for i in range(len(state_files))])

    def __len__(self):
        """
        Returns the total number of data samples in the dataset.
        """
        return len(self.data_info)

    def __getitem__(self, idx):
        """
        Retrieves the 3-channel image at the given index from the dataset.

        Args:
            idx (int): Index of the data item.

        Returns:
            sample (dict): Dictionary containing the 'image' stack, 'model_id',
                           'state_idx', and 'sequence_length'.
        """
        model_id, state_idx = self.data_info[idx]

        # Construct the paths to elevation, thickness, and velocity arrays
        elevation_path = os.path.join(self.main_dir, model_id, 'elevation', f'elevation.npy')
        thickness_path = os.path.join(self.main_dir, model_id, 'thickness', f'thickness_{state_idx}.npy')
        velocity_path = os.path.join(self.main_dir, model_id, 'velocity', f'velocity_{state_idx}.npy')

        # Load the arrays
        elevation = np.load(elevation_path)
        thickness = np.load(thickness_path)
        velocity = np.load(velocity_path)

        # Stack the individual arrays to form a 3-channel image
        image = np.stack((elevation, thickness, velocity), axis=-1)

        # Apply transforms if any
        if self.transform:
            image = self.transform(image)

        # Get the sequence length (number of states for the model)
        sequence_length = len(os.listdir(os.path.join(self.main_dir, model_id, 'thickness')))

        sample = {
            'image': image,
            'model_id': model_id,
            'state_idx': state_idx,
            'sequence_length': sequence_length
        }

        return sample

def debris_collate_fn(batch):
    """
    Collate function to pad the sequences to the same length.

    Args:
        batch (list): List of samples from the dataset.

    Returns:
        dict: Dictionary containing padded images, model_ids, state_indices, and sequence_lengths.
    """
    # Sort the batch in the descending order of sequence lengths
    batch.sort(key=lambda x: x['sequence_length'], reverse=True)
    
    # Separate the batch into its components
    images = [item['image'] for item in batch]
    model_ids = [item['model_id'] for item in batch]
    state_indices = [item['state_idx'] for item in batch]
    sequence_lengths = torch.tensor([item['sequence_length'] for item in batch])
    
    # Pad the image sequences
    images_padded = pad_sequence([torch.tensor(img) for img in images], batch_first=True)
    
    # Return the collated batch
    return {
        'images': images_padded,
        'model_ids': model_ids,
        'state_indices': state_indices,
        'sequence_lengths': sequence_lengths
    }