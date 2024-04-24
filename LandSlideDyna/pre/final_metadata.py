import os
import json
import numpy as np
import pandas as pd

from collections import OrderedDict


class FinallMetadata:
    """A class to handle metadata about a debris flow model."""

    def __init__(self, root_directory, model_id):
        """Initializes the ModelMetadata instance.
        
        Args:
            root_directory (str): The root directory where model data is stored.
            model_id (int): The unique identifier for the model.
        """
        self.model_id = model_id
        self.root_directory = root_directory
        
        self.model_dir = os.path.join(self.root_directory, f'{self.model_id}')
        self.elevation_dir = os.path.join(self.model_dir, '04_FinalProcessedData', 'elevation')
        self.thickness_dir = os.path.join(self.model_dir, '04_FinalProcessedData', 'thickness')
        self.velocity_dir = os.path.join(self.model_dir, '04_FinalProcessedData', 'velocity')


        # Data structure to hold the information about elevation, thickness, velocity, and states
        self.data = {
            'elevation_array_dict': {},
            'thickness_state_array_dict': OrderedDict(),
            'velocity_state_array_dict': OrderedDict(),
        }
        
        self.metadata = {}
        
        self.load_data()
        self.calculate_metadata()


    def load_data(self):
        """Loads the elevation, thickness, and velocity data from the respective directories."""

        # Load elevation data

        self.data['elevation_array_dict']['z_values'] = np.load(os.path.join(self.elevation_dir, f'{self.model_id}_elevation.npy'))

        # Load thickness data
        self._load_state_and_summary_data(self.thickness_dir, 'thickness')

        # Load velocity data
        self._load_state_and_summary_data(self.velocity_dir, 'velocity')


    def _load_state_and_summary_data(self, directory, data_type):
        """Loads state-specific data and summary data from the given directory.
        
        Args:
            directory (str): The directory containing the state and summary data files.
            data_type (str): A string representing the type of data ('thickness' or 'velocity').
        """
        state_data_dict = {}
        summary_data_dict = {}
        
        files = [f for f in os.listdir(directory) if f.startswith(f'{self.model_id}_{data_type}_') and f.endswith('.npy')]
        for filename in files:
            split_name = filename.split(f'{self.model_id}_{data_type}_')
            key = split_name[-1].replace('.npy', '')
            if key.isdigit():
                # State data files have numeric keys
                state_data_dict[int(key)] = np.load(os.path.join(directory, filename))
            else:
                # Summary data files have descriptive keys
                summary_data_dict[key] = np.load(os.path.join(directory, filename))

        # Sort the state data dictionary by key (converted to integer) and then convert back to string
        sorted_state_data_dict = OrderedDict(sorted(state_data_dict.items()))

        # Update the respective dictionaries in self.data with sorted state data and summary data
        self.data[f'{data_type}_state_array_dict'] = {str(k): v for k, v in sorted_state_data_dict.items()}
        self.data[f'{data_type}_summary_data_dict'] = summary_data_dict


    def calculate_metadata(self):
        self.calculate_bounding_boxes()
    

    def calculate_bounding_boxes(self):
        """Calculates and stores the bounding box and its dimensions of the debris for every state,
        as well as the overall bounding box for all states."""
        # Initialize the bounding boxes metadata
        self.metadata['bounding_boxes'] = {}

        # Variables to track the extremities for the overall bounding box
        global_min_x = float('inf')
        global_max_x = -float('inf')
        global_min_y = float('inf')
        global_max_y = -float('inf')

        # Iterate over states using keys from thickness_state_array_dict
        for state_number, thickness_array in self.data['thickness_state_array_dict'].items():
            
            # Only consider the indices where there are non-zero thickness values
            nonzero_indices = np.nonzero(thickness_array)
            
            if nonzero_indices[0].size > 0:
                # Calculate bounding box
                min_x = np.min(nonzero_indices[1])
                max_x = np.max(nonzero_indices[1])
                min_y = np.min(nonzero_indices[0])
                max_y = np.max(nonzero_indices[0])

                # Update global bounding box values
                global_min_x = min(min_x, global_min_x)
                global_max_x = max(max_x, global_max_x)
                global_min_y = min(min_y, global_min_y)
                global_max_y = max(max_y, global_max_y)

                # Calculate dimensions of the bounding box
                bbox_width = max_x - min_x + 1  # +1 to include the starting index
                bbox_height = max_y - min_y + 1  # +1 to include the starting index

                # Store in metadata
                self.metadata['bounding_boxes'][state_number] = {
                    'min_x': min_x, 'max_x': max_x,
                    'min_y': min_y, 'max_y': max_y,
                    'width': bbox_width, 'height': bbox_height,
                    'area': bbox_width * bbox_height
                }
            else:
                # If no debris is present, store None or an appropriate representation
                self.metadata['bounding_boxes'][state_number] = None

        # Store the overall bounding box
        self.metadata['overall_bounding_box'] = {
            'min_x': global_min_x, 'max_x': global_max_x,
            'min_y': global_min_y, 'max_y': global_max_y,
            'width': global_max_x - global_min_x + 1,
            'height': global_max_y - global_min_y + 1
        }

        # Calculate and store the center point for the overall bounding box
        self.metadata['overall_bounding_box']['center_x'] = (global_min_x + global_max_x) / 2
        self.metadata['overall_bounding_box']['center_y'] = (global_min_y + global_max_y) / 2

    
    def save_metadata(self):
        """Saves the metadata to a JSON file named with the model_id in the model_dir directory."""
        # Convert NumPy types to Python types for JSON serialization
        metadata_serializable = self._convert_to_serializable(self.metadata)

        # Create the filename using the model_id
        filename = f"{self.model_id}_final_metadata.json"
        # Define the full filepath including the model directory
        filepath = os.path.join(self.model_dir, filename)

        # Ensure the model directory exists, create if it does not
        os.makedirs(self.model_dir, exist_ok=True)

        # Write the metadata to the file
        with open(filepath, 'w') as f:
            json.dump(metadata_serializable, f, indent=4)  # use indent for pretty-printing


    def _convert_to_serializable(self, data):
        """Recursively convert NumPy types to Python types for JSON serialization."""
        if isinstance(data, (np.ndarray, np.generic)):
            return data.tolist()  # Convert NumPy arrays or scalar to Python list
        elif isinstance(data, dict):
            return {key: self._convert_to_serializable(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._convert_to_serializable(item) for item in data]
        elif isinstance(data, (int, float, str, bool)):
            return data
        else:
            return str(data)  # Fallback conversion for any other types

