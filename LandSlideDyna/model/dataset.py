
import numpy as np
import glob
import os
import torch

from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

class DebrisStatePairsDataset(Dataset):
    def __init__(self, root_dir, array_size=128, apply_scaling=False, timestep_interval=1):
        self.array_size = array_size
        self.data_files = []
        self.model_ids = []  # List to store the model IDs
        self.terrain_files = {}
        self.apply_scaling = apply_scaling
        self.timestep_interval = timestep_interval
        self.scaling_factors = None

        def get_state_number(file_path):
            return int(file_path.split('_')[-1].split('.')[0])

        model_dirs = glob.glob(os.path.join(root_dir, '*'))
        
        for model_dir in model_dirs:
            if os.path.isdir(model_dir):
                # Extract the model ID from the model_dir path
                model_id = os.path.basename(model_dir)

                file_patterns = {
                    'velocity': os.path.join(model_dir, f'04_FinalProcessedData_{str(self.array_size)}', 'velocity', '*_velocity_*.npy'),
                    'thickness': os.path.join(model_dir, f'04_FinalProcessedData_{str(self.array_size)}', 'thickness', '*_thickness_*.npy')
                }
                file_collections = {key: sorted(glob.glob(pattern), key=get_state_number) for key, pattern in file_patterns.items()}

                terrain_pattern = os.path.join(model_dir, f'04_FinalProcessedData_{str(self.array_size)}', 'elevation', '*_elevation.npy')
                terrain_file = glob.glob(terrain_pattern)

                if terrain_file:
                    self.terrain_files[model_dir] = terrain_file[0]

                num_states = len(file_collections['velocity'])
                for i in range(num_states - self.timestep_interval):
                    current_velocity_path = file_collections['velocity'][i]
                    next_velocity_path = file_collections['velocity'][i + self.timestep_interval]
                    current_thickness_path = file_collections['thickness'][i]
                    next_thickness_path = file_collections['thickness'][i + self.timestep_interval]

                    self.data_files.append((current_velocity_path, next_velocity_path,
                                            current_thickness_path, next_thickness_path))
                    self.model_ids.append(model_id)

    def compute_scaling_factors(self, subset):
        # Ensure that scaling is intended before computing factors
        if not self.apply_scaling:
            raise RuntimeError("Scaling factors called to be computed when scaling is not applied.")

        # Initialize min and max values with infinities
        min_elevation = np.inf
        max_elevation = -np.inf
        min_velocity = np.inf
        max_velocity = -np.inf
        min_thickness = np.inf
        max_thickness = -np.inf

        # Compute min and max values over the subset
        for idx in subset.indices:
            current_velocity_path, next_velocity_path, current_thickness_path, next_thickness_path = self.data_files[idx]

            # Get the model directory from the current velocity path
            model_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_velocity_path)))

            # Load the corresponding terrain file
            terrain_path = self.terrain_files[model_dir]
            terrain = np.load(terrain_path)
            min_elevation = min(min_elevation, terrain.min())
            max_elevation = max(max_elevation, terrain.max())

            current_velocity = np.load(current_velocity_path)
            next_velocity = np.load(next_velocity_path)
            current_thickness = np.load(current_thickness_path)
            next_thickness = np.load(next_thickness_path)

            min_velocity = min(min_velocity, current_velocity.min(), next_velocity.min())
            max_velocity = max(max_velocity, current_velocity.max(), next_velocity.max())
            min_thickness = min(min_thickness, current_thickness.min(), next_thickness.min())
            max_thickness = max(max_thickness, current_thickness.max(), next_thickness.max())

        return min_elevation, max_elevation, min_velocity, max_velocity, min_thickness, max_thickness

    def scale_data(self, terrain, velocity, thickness):
        if self.scaling_factors is None:
            raise RuntimeError("Scaling factors not computed.")

        min_elevation, max_elevation, min_velocity, max_velocity, min_thickness, max_thickness = self.scaling_factors

        terrain_scaled = (terrain - min_elevation) / (max_elevation - min_elevation)
        velocity_scaled = (velocity - min_velocity) / (max_velocity - min_velocity)
        thickness_scaled = (thickness - min_thickness) / (max_thickness - min_thickness)

        return terrain_scaled, velocity_scaled, thickness_scaled

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        current_velocity_path, next_velocity_path, current_thickness_path, next_thickness_path = self.data_files[idx]
        
        # Load current and next states for velocity and thickness
        current_velocity = np.load(current_velocity_path)
        next_velocity = np.load(next_velocity_path)
        current_thickness = np.load(current_thickness_path)
        next_thickness = np.load(next_thickness_path)

        # Get the model directory from the current velocity path
        model_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_velocity_path)))

        # Load the corresponding terrain file
        terrain_path = self.terrain_files[model_dir]
        terrain = np.load(terrain_path)

        if self.apply_scaling:
            terrain_scaled, current_velocity_scaled, current_thickness_scaled = self.scale_data(
                terrain, current_velocity, current_thickness
            )
            _, next_velocity_scaled, next_thickness_scaled = self.scale_data(
                terrain, next_velocity, next_thickness
            )
        else:
            terrain_scaled = terrain
            current_velocity_scaled = current_velocity
            current_thickness_scaled = current_thickness
            next_velocity_scaled = next_velocity
            next_thickness_scaled = next_thickness

        # Stack arrays as channels for CNN input and output
        cnn_input = np.stack((terrain_scaled, current_velocity_scaled, current_thickness_scaled), axis=0)
        cnn_output = np.stack((next_velocity_scaled, next_thickness_scaled), axis=0)

        return torch.from_numpy(cnn_input).float(), torch.from_numpy(cnn_output).float()

    def create_dataloaders(self, split_proportions, batch_size, random_state=42):
        # Unpack the proportions for clarity
        train_proportion, val_proportion, test_proportion = split_proportions
        
        # Assert that the proportions sum to 1
        assert np.isclose(sum(split_proportions), 1.0), "Proportions must sum up to 1."
        
        # Create a list of unique model IDs
        unique_model_ids = np.unique(self.model_ids)
        
        # Split model IDs into train, val, and test sets
        train_model_ids, temp_model_ids = train_test_split(
            unique_model_ids, test_size=(val_proportion + test_proportion), random_state=random_state
        )
        val_model_ids, test_model_ids = train_test_split(
            temp_model_ids, test_size=test_proportion / (val_proportion + test_proportion), random_state=random_state
        )
        
        # Now, filter the dataset's data points based on the model IDs
        train_indices = [i for i, model_id in enumerate(self.model_ids) if model_id in train_model_ids]
        val_indices = [i for i, model_id in enumerate(self.model_ids) if model_id in val_model_ids]
        test_indices = [i for i, model_id in enumerate(self.model_ids) if model_id in test_model_ids]
        
        # Compute scaling factors based on the train subset, but only if scaling is applied
        if self.apply_scaling:
            train_subset = Subset(self, train_indices)
            self.scaling_factors = self.compute_scaling_factors(train_subset)

        # Create subsets
        train_dataset = Subset(self, train_indices)
        val_dataset = Subset(self, val_indices)
        test_dataset = Subset(self, test_indices)
        
        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Return the DataLoaders
        return train_loader, val_loader, test_loader

    def get_pair_filenames(self):
        pair_filenames = []
        for current_velocity_path, next_velocity_path, current_thickness_path, next_thickness_path in self.data_files:
            current_velocity_filename = os.path.basename(current_velocity_path)
            next_velocity_filename = os.path.basename(next_velocity_path)
            current_thickness_filename = os.path.basename(current_thickness_path)
            next_thickness_filename = os.path.basename(next_thickness_path)
            
            model_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_velocity_path)))
            terrain_filename = os.path.basename(self.terrain_files[model_dir]) if model_dir in self.terrain_files else None

            pair_filenames.append({
                'current_velocity': current_velocity_filename,
                'next_velocity': next_velocity_filename,
                'current_thickness': current_thickness_filename,
                'next_thickness': next_thickness_filename,
                'terrain': terrain_filename
            })
        return pair_filenames




    



class DebrisStateSeriesDataset(Dataset):

    def __init__(self, root_dir, array_size, sequence_length, apply_scaling=False, timestep_interval=1):
        self.array_size = array_size
        self.data_files = []
        self.model_ids = []  # List to store the model IDs
        self.terrain_files = {}
        self.apply_scaling = apply_scaling
        self.sequence_length = sequence_length
        self.timestep_interval = timestep_interval
        
        self.scaling_factors = None

        def get_state_number(file_path):
            return int(file_path.split('_')[-1].split('.')[0])

        model_dirs = glob.glob(os.path.join(root_dir, '*'))
        
        for model_dir in model_dirs:
            if os.path.isdir(model_dir):
                # Extract the model ID from the model_dir path
                model_id = os.path.basename(model_dir)

                file_patterns = {
                    'velocity': os.path.join(model_dir, f'04_FinalProcessedData_{str(self.array_size)}', 'velocity', '*_velocity_*.npy'),
                    'thickness': os.path.join(model_dir, f'04_FinalProcessedData_{str(self.array_size)}', 'thickness', '*_thickness_*.npy')
                }
                file_collections = {key: sorted(glob.glob(pattern), key=get_state_number) for key, pattern in file_patterns.items()}

                terrain_pattern = os.path.join(model_dir, f'04_FinalProcessedData_{str(self.array_size)}', 'elevation', '*_elevation.npy')
                terrain_file = glob.glob(terrain_pattern)

                if terrain_file:
                    self.terrain_files[model_dir] = terrain_file[0]

                num_states = len(file_collections['velocity'])
                for i in range(num_states - (self.sequence_length - 1) * self.timestep_interval):
                    sequence_velocity = []
                    sequence_thickness = []
                    for j in range(self.sequence_length):
                        velocity_path = file_collections['velocity'][i + j * self.timestep_interval]
                        thickness_path = file_collections['thickness'][i + j * self.timestep_interval]
                        sequence_velocity.append(velocity_path)
                        sequence_thickness.append(thickness_path)

                    if i + self.sequence_length * self.timestep_interval < num_states:
                        next_velocity = file_collections['velocity'][i + self.sequence_length * self.timestep_interval]
                        next_thickness = file_collections['thickness'][i + self.sequence_length * self.timestep_interval]
                        self.data_files.append((sequence_velocity, sequence_thickness, next_velocity, next_thickness))
                        self.model_ids.append(model_id)

    def get_sequence_filenames(self):
        sequence_filenames = []
        for velocity_sequence, thickness_sequence, next_velocity, next_thickness in self.data_files:
            velocity_filenames = [os.path.basename(v) for v in velocity_sequence]
            thickness_filenames = [os.path.basename(t) for t in thickness_sequence]
            next_velocity_filename = os.path.basename(next_velocity)
            next_thickness_filename = os.path.basename(next_thickness)
            sequence_filenames.append({
                'velocity_sequence': velocity_filenames,
                'thickness_sequence': thickness_filenames,
                'next_velocity': next_velocity_filename,
                'next_thickness': next_thickness_filename
            })
        return sequence_filenames

    def compute_scaling_factors(self, subset):
        # Ensure that scaling is intended before computing factors
        if not self.apply_scaling:
            raise RuntimeError("Scaling factors called to be computed when scaling is not applied.")

        # Initialize min and max values with infinities
        min_elevation = np.inf
        max_elevation = -np.inf
        min_velocity = np.inf
        max_velocity = -np.inf
        min_thickness = np.inf
        max_thickness = -np.inf

        # Compute min and max values over the subset
        for idx in subset.indices:
            velocity_sequence, thickness_sequence, next_velocity_path, next_thickness_path = self.data_files[idx]

            # Get the model directory from the next velocity path
            model_dir = os.path.dirname(os.path.dirname(os.path.dirname(next_velocity_path)))

            # Load the corresponding terrain file
            terrain_path = self.terrain_files[model_dir]
            terrain = np.load(terrain_path)
            min_elevation = min(min_elevation, terrain.min())
            max_elevation = max(max_elevation, terrain.max())

            for velocity_path, thickness_path in zip(velocity_sequence, thickness_sequence):
                if velocity_path is None or thickness_path is None:
                    continue

                velocity = np.load(velocity_path)
                thickness = np.load(thickness_path)
                min_velocity = min(min_velocity, velocity.min())
                max_velocity = max(max_velocity, velocity.max())
                min_thickness = min(min_thickness, thickness.min())
                max_thickness = max(max_thickness, thickness.max())

        return min_elevation, max_elevation, min_velocity, max_velocity, min_thickness, max_thickness

    def scale_data(self, terrain, velocity, thickness):
        if self.scaling_factors is None:
            raise RuntimeError("Scaling factors not computed.")

        min_elevation, max_elevation, min_velocity, max_velocity, min_thickness, max_thickness = self.scaling_factors

        terrain_scaled = (terrain - min_elevation) / (max_elevation - min_elevation)

        if velocity is None:
            velocity_scaled = np.zeros_like(terrain)
        elif max_velocity == min_velocity:
            velocity_scaled = np.zeros_like(velocity)
        else:
            velocity_scaled = (velocity - min_velocity) / (max_velocity - min_velocity)

        if thickness is None:
            thickness_scaled = np.zeros_like(terrain)
        elif max_thickness == min_thickness:
            thickness_scaled = np.zeros_like(thickness)
        else:
            thickness_scaled = (thickness - min_thickness) / (max_thickness - min_thickness)

        return terrain_scaled, velocity_scaled, thickness_scaled

    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        velocity_sequence, thickness_sequence, next_velocity_path, next_thickness_path = self.data_files[idx]

        # Get the model directory from the next velocity path
        model_dir = os.path.dirname(os.path.dirname(os.path.dirname(next_velocity_path)))

        # Load the corresponding terrain file
        terrain_path = self.terrain_files[model_dir]
        terrain = np.load(terrain_path)

        # Initialize empty lists to store the sequence data
        input_sequence_data = []

        # Load and process each state in the sequence
        for i in range(self.sequence_length):
            if i < len(velocity_sequence) and velocity_sequence[i] is not None and thickness_sequence[i] is not None:
                velocity = np.load(velocity_sequence[i])
                thickness = np.load(thickness_sequence[i])

                if self.apply_scaling:
                    _, velocity_scaled, thickness_scaled = self.scale_data(terrain, velocity, thickness)
                    input_sequence_data.append(np.stack((terrain, thickness_scaled, velocity_scaled), axis=0))
                else:
                    input_sequence_data.append(np.stack((terrain, thickness, velocity), axis=0))
            else:
                # If the file path is None or the sequence is shorter than the desired length, pad with zeros
                input_sequence_data.append(np.zeros((3, terrain.shape[0], terrain.shape[1])))

        # Stack the sequence data
        input_sequence_data = np.stack(input_sequence_data, axis=0)

        if next_velocity_path is not None and next_thickness_path is not None:
            # Load next states for velocity and thickness
            next_velocity = np.load(next_velocity_path)
            next_thickness = np.load(next_thickness_path)

            if self.apply_scaling:
                _, next_velocity_scaled, next_thickness_scaled = self.scale_data(terrain, next_velocity, next_thickness)
                cnn_output = np.stack((next_thickness_scaled, next_velocity_scaled), axis=0)
            else:
                cnn_output = np.stack((next_thickness, next_velocity), axis=0)
        else:
            # If the next velocity or thickness path is None, return arrays of zeros
            cnn_output = np.zeros((2, terrain.shape[0], terrain.shape[1]))

        return torch.from_numpy(input_sequence_data).float(), torch.from_numpy(cnn_output).float()

    def create_dataloaders(self, split_proportions, batch_size, random_state=42):
        # Unpack the proportions for clarity
        train_proportion, val_proportion, test_proportion = split_proportions
        
        # Assert that the proportions sum to 1
        assert np.isclose(sum(split_proportions), 1.0), "Proportions must sum up to 1."
        
        # Create a list of unique model IDs
        unique_model_ids = np.unique(self.model_ids)
        
        # Split model IDs into train, val, and test sets
        train_model_ids, temp_model_ids = train_test_split(
            unique_model_ids, test_size=(val_proportion + test_proportion), random_state=random_state
        )
        val_model_ids, test_model_ids = train_test_split(
            temp_model_ids, test_size=test_proportion / (val_proportion + test_proportion), random_state=random_state
        )
        
        # Now, filter the dataset's data points based on the model IDs
        train_indices = [i for i, model_id in enumerate(self.model_ids) if model_id in train_model_ids]
        val_indices = [i for i, model_id in enumerate(self.model_ids) if model_id in val_model_ids]
        test_indices = [i for i, model_id in enumerate(self.model_ids) if model_id in test_model_ids]
        
        # Compute scaling factors based on the train subset, but only if scaling is applied
        if self.apply_scaling:
            train_subset = Subset(self, train_indices)
            self.scaling_factors = self.compute_scaling_factors(train_subset)

        # Create subsets
        train_dataset = Subset(self, train_indices)
        val_dataset = Subset(self, val_indices)
        test_dataset = Subset(self, test_indices)
        
        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Return the DataLoaders
        return train_loader, val_loader, test_loader
    
    def get_sequence_filenames(self):
        """Retrieves filenames from the data_files attribute and includes corresponding terrain file."""
        sequence_info = []
        for velocity_sequence, thickness_sequence, next_velocity, next_thickness in self.data_files:
            # Extract filenames from paths, handling None values
            velocity_filenames = [os.path.basename(v) if v else None for v in velocity_sequence]
            thickness_filenames = [os.path.basename(t) if t else None for t in thickness_sequence]
            next_velocity_filename = os.path.basename(next_velocity)
            next_thickness_filename = os.path.basename(next_thickness)
            
            # Get the model directory from the next velocity path
            # Assuming the directory structure remains consistent for all files
            model_dir = os.path.dirname(os.path.dirname(os.path.dirname(next_velocity)))
            
            # Retrieve the terrain filename
            terrain_filename = os.path.basename(self.terrain_files[model_dir]) if model_dir in self.terrain_files else None

            # Store the filenames of the full sequence including the terrain
            sequence_info.append({
                'velocity_sequence': velocity_filenames,
                'thickness_sequence': thickness_filenames,
                'next_velocity': next_velocity_filename,
                'next_thickness': next_thickness_filename,
                'terrain': terrain_filename
            })
        return sequence_info