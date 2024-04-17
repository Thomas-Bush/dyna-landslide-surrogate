
import numpy as np
import glob
import os
import torch

from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split


class DebrisStatePairsDataset(Dataset):
    def __init__(self, root_dir, array_size=128, apply_scaling=False):
        self.array_size = array_size
        self.data_files = []
        self.model_ids = []  # List to store the model IDs
        self.terrain_files = {}
        self.apply_scaling = apply_scaling

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

                for i in range(len(file_collections['velocity']) - 1):
                    current_state_velocity = get_state_number(file_collections['velocity'][i])
                    next_state_velocity = get_state_number(file_collections['velocity'][i + 1])
                    current_state_thickness = get_state_number(file_collections['thickness'][i])
                    next_state_thickness = get_state_number(file_collections['thickness'][i + 1])

                    if (next_state_velocity == current_state_velocity + 1) and (next_state_thickness == current_state_thickness + 1):
                        self.data_files.append((file_collections['velocity'][i], file_collections['velocity'][i + 1],
                                                file_collections['thickness'][i], file_collections['thickness'][i + 1]))
                        # Append the model_id for each valid data point
                        self.model_ids.append(model_id)


    def compute_scaling_factors(self):
        # Ensure that scaling is intended before computing factors
        if not self.apply_scaling:
            raise RuntimeError("Scaling factors called to be computed when scaling is not applied.")

        # Initialize min and max values with infinities
        self.min_elevation = np.inf
        self.max_elevation = -np.inf
        self.min_velocity = np.inf
        self.max_velocity = -np.inf
        self.min_thickness = np.inf
        self.max_thickness = -np.inf

        # Compute min and max values over the training set
        for model_dir, terrain_path in self.terrain_files.items():
            terrain = np.load(terrain_path)
            self.min_elevation = min(self.min_elevation, terrain.min())
            self.max_elevation = max(self.max_elevation, terrain.max())

            # Assuming that velocity and thickness files are matched in the data_files
            for velocity_path, _, thickness_path, _ in self.data_files:
                if model_dir in velocity_path:  # Checking if the file belongs to the current model directory
                    velocity = np.load(velocity_path)
                    thickness = np.load(thickness_path)
                    self.min_velocity = min(self.min_velocity, velocity.min())
                    self.max_velocity = max(self.max_velocity, velocity.max())
                    self.min_thickness = min(self.min_thickness, thickness.min())
                    self.max_thickness = max(self.max_thickness, thickness.max())

    def scale_data(self, terrain, velocity, thickness):
        # Apply Min-Max scaling to each feature
        terrain_scaled = (terrain - self.min_elevation) / (self.max_elevation - self.min_elevation) * 10
        velocity_scaled = (velocity - self.min_velocity) / (self.max_velocity - self.min_velocity) * 10 
        thickness_scaled = (thickness - self.min_thickness) / (self.max_thickness - self.min_thickness) * 10

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
            # Apply scaling to the loaded data
            terrain_scaled, current_velocity_scaled, current_thickness_scaled = self.scale_data(
                terrain, current_velocity, current_thickness
            )
            _, next_velocity_scaled, next_thickness_scaled = self.scale_data(
                terrain, next_velocity, next_thickness
            )
        else:
            # No scaling applied, use the original values
            terrain_scaled, current_velocity_scaled, current_thickness_scaled = terrain, current_velocity, current_thickness
            next_velocity_scaled, next_thickness_scaled = next_velocity, next_thickness

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
    



class DebrisStateSeriesDataset(Dataset):
    def __init__(self, root_dir, array_size, apply_scaling=False, sequence_length=3):
        self.array_size = array_size
        self.data_files = []
        self.model_ids = []  # List to store the model IDs
        self.terrain_files = {}
        self.apply_scaling = apply_scaling
        self.sequence_length = sequence_length

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
                for i in range(num_states):
                    start_index = max(0, i - self.sequence_length + 1)
                    end_index = i + 1
                    sequence_velocity = file_collections['velocity'][start_index:end_index]
                    sequence_thickness = file_collections['thickness'][start_index:end_index]

                    # Pad the sequences with None if they are shorter than the desired length
                    sequence_velocity = [None] * (self.sequence_length - len(sequence_velocity)) + sequence_velocity
                    sequence_thickness = [None] * (self.sequence_length - len(sequence_thickness)) + sequence_thickness

                    if i + 1 < num_states:
                        next_velocity = file_collections['velocity'][i+1]
                        next_thickness = file_collections['thickness'][i+1]
                        self.data_files.append((sequence_velocity, sequence_thickness, next_velocity, next_thickness))
                        self.model_ids.append(model_id)

    def compute_scaling_factors(self):
        # Ensure that scaling is intended before computing factors
        if not self.apply_scaling:
            raise RuntimeError("Scaling factors called to be computed when scaling is not applied.")

        # Initialize min and max values with infinities
        self.min_elevation = np.inf
        self.max_elevation = -np.inf
        self.min_velocity = np.inf
        self.max_velocity = -np.inf
        self.min_thickness = np.inf
        self.max_thickness = -np.inf

        # Compute min and max values over the training set
        for model_dir, terrain_path in self.terrain_files.items():
            terrain = np.load(terrain_path)
            self.min_elevation = min(self.min_elevation, terrain.min())
            self.max_elevation = max(self.max_elevation, terrain.max())

            # Assuming that velocity and thickness files are matched in the data_files
            for velocity_paths, thickness_paths, _, _ in self.data_files:
                for velocity_path, thickness_path in zip(velocity_paths, thickness_paths):
                    if velocity_path is None or thickness_path is None:
                        continue

                    if model_dir in velocity_path:  # Checking if the file belongs to the current model directory
                        velocity = np.load(velocity_path)
                        thickness = np.load(thickness_path)
                        self.min_velocity = min(self.min_velocity, velocity.min())
                        self.max_velocity = max(self.max_velocity, velocity.max())
                        self.min_thickness = min(self.min_thickness, thickness.min())
                        self.max_thickness = max(self.max_thickness, thickness.max())


    def scale_data(self, terrain, velocity, thickness):
        terrain_scaled = (terrain - self.min_elevation) / (self.max_elevation - self.min_elevation) * 10

        if velocity is None:
            velocity_scaled = np.zeros_like(terrain)
        elif self.max_velocity == self.min_velocity:
            velocity_scaled = np.zeros_like(velocity)
        else:
            velocity_scaled = (velocity - self.min_velocity) / (self.max_velocity - self.min_velocity) * 10

        if thickness is None:
            thickness_scaled = np.zeros_like(terrain)
        elif self.max_thickness == self.min_thickness:
            thickness_scaled = np.zeros_like(thickness)
        else:
            thickness_scaled = (thickness - self.min_thickness) / (self.max_thickness - self.min_thickness) * 10

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