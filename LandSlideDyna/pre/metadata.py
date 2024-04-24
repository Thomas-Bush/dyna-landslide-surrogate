import os
import json
import numpy as np
import pandas as pd

from collections import OrderedDict


class ModelMetadata:
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
        self.elevation_dir = os.path.join(self.model_dir, '03_initial_process', 'elevation')
        self.thickness_dir = os.path.join(self.model_dir, '03_initial_process', 'thickness')
        self.velocity_dir = os.path.join(self.model_dir, '03_initial_process', 'velocity')
        self.states_dir = os.path.join(self.model_dir, "02_extract")

        # Data structure to hold the information about elevation, thickness, velocity, and states
        self.data = {
            'elevation_array_dict': {},
            'thickness_state_array_dict': OrderedDict(),
            'velocity_state_array_dict': OrderedDict(),
            'thickness_summary_data_dict': {},
            'velocity_summary_data_dict': {},
            'state_timesteps': None 
        }
        
        self.metadata = {}
        
        self.load_data()
        self.calculate_metadata()


    def load_data(self):
        """Loads the elevation, thickness, and velocity data from the respective directories."""

        # Load elevation data
        self.data['elevation_array_dict']['x_values'] = np.load(os.path.join(self.elevation_dir, f'{self.model_id}_elevation_x_values.npy'))
        self.data['elevation_array_dict']['y_values'] = np.load(os.path.join(self.elevation_dir, f'{self.model_id}_elevation_y_values.npy'))
        self.data['elevation_array_dict']['z_values'] = np.load(os.path.join(self.elevation_dir, f'{self.model_id}_elevation_z_values.npy'))

        # Load thickness data
        self._load_state_and_summary_data(self.thickness_dir, 'thickness')

        # Load velocity data
        self._load_state_and_summary_data(self.velocity_dir, 'velocity')

        # Load states
        self._load_state_timesteps()


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


    def _load_state_timesteps(self):
        """Loads the state timesteps from a CSV file and adds them to the data dictionary."""
        csv_file_path = os.path.join(self.states_dir, f"{self.model_id}_states.csv")

        # Check if the file exists before attempting to read
        if not os.path.isfile(csv_file_path):
            raise FileNotFoundError(f"The file {csv_file_path} does not exist.")

        # Load state timesteps from CSV
        state_timesteps_df = pd.read_csv(csv_file_path)

        # Convert to dictionary {state_label: state_timestamp}, ensuring state_label is an integer
        state_timesteps_dict = {str(int(row['State_Label'])): row['State_Timestamp']
                                for index, row in state_timesteps_df.iterrows()}

        # Add to the data dictionary under the 'state_timesteps' key
        self.data['state_timesteps'] = state_timesteps_dict

    
    def calculate_metadata(self):
        """Calculate and update the metadata based on the provided data."""
        # Assuming that elevation_array_dict contains 'x_values', 'y_values', 'z_values'
        elevation_dict = self.data['elevation_array_dict']
        self.metadata['min_x_value'] = np.min(elevation_dict['x_values'])
        self.metadata['max_x_value'] = np.max(elevation_dict['x_values'])
        self.metadata['min_y_value'] = np.min(elevation_dict['y_values'])
        self.metadata['max_y_value'] = np.max(elevation_dict['y_values'])
        self.metadata['min_z_value'] = np.min(elevation_dict['z_values'])
        self.metadata['max_z_value'] = np.max(elevation_dict['z_values'])

        # Dimensions based on min and max values
        self.metadata['x_dimension'] = self.metadata['max_x_value'] - self.metadata['min_x_value']
        self.metadata['y_dimension'] = self.metadata['max_y_value'] - self.metadata['min_y_value']
        self.metadata['z_dimension'] = self.metadata['max_z_value'] - self.metadata['min_z_value']

        # Grid resolution calculation
        self.metadata['grid_resolution_x'] = self.calculate_grid_resolution(elevation_dict['x_values'])
        self.metadata['grid_resolution_y'] = self.calculate_grid_resolution(elevation_dict['y_values'])

        # Total number of states and average timestep calculation
        state_timesteps = self.data['state_timesteps']
        self.metadata['total_number_of_states'] = len(state_timesteps)
        self.metadata['average_timestep'] = self.calculate_average_timestep()
        self.metadata['timesteps'] = state_timesteps

        # Calculate more metadata
        self.check_empty_states()

        self.calculate_bounding_boxes()

        self.calculate_state_statistics()

        self.calculate_debris_metrics()


    def calculate_grid_resolution(self, values):
        """Calculate the grid resolution based on the axis values.

        Args:
            values (np.ndarray): The values along an axis.

        Returns:
            float: The calculated grid resolution.
        """
        return np.diff(values).mean() if len(values) > 1 else None



    def calculate_average_timestep(self):
        """Calculate the average timestep from a sequence of timestamps stored in self.data.

        Returns:
            float or None: The average timestep calculated as the mean of the difference between sequential timestamps,
                        or None if insufficient timestamps.
        """
        timestamps = pd.Series(self.data['state_timesteps'].values())
        sorted_timestamps = timestamps.sort_values().values
        if len(sorted_timestamps) > 1:
            timestep_differences = np.diff(sorted_timestamps)
            return np.mean(timestep_differences)
        return None


    def check_empty_states(self):
        """Checks each state and records which ones are empty (all zeroes).
        
        The result is stored in the class attribute `metadata` with a key 'empty_states', mapping each state number
        to a boolean indicating whether it is empty or not.
        """
        # Initialize empty states metadata
        self.metadata['empty_states'] = {}

        # Get all state numbers from the thickness and velocity dicts (assuming they have the same keys)
        state_numbers = self.data['thickness_state_array_dict'].keys()

        # Loop over all state numbers and check for empty states
        for state_number in state_numbers:
            thickness_values = self.data['thickness_state_array_dict'][state_number]
            velocity_values = self.data['velocity_state_array_dict'][state_number]

            # Check if all values for a state are zero for both thickness and velocity
            is_empty = np.all(thickness_values == 0) and np.all(velocity_values == 0)
            self.metadata['empty_states'][state_number] = is_empty


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


    def calculate_state_statistics(self):
        """Calculates and stores the max and average values of thickness and velocity for each state."""
        # Initialize the statistics metadata
        self.metadata['state_statistics'] = {}

        # Use the keys from thickness_state_array_dict as they should match the velocity_state_array_dict keys
        state_numbers = self.data['thickness_state_array_dict'].keys()

        # Iterate over states
        for state_number in state_numbers:
            thickness_values = self.data['thickness_state_array_dict'][state_number]
            velocity_values = self.data['velocity_state_array_dict'][state_number]

            # Calculate max values
            max_thickness = np.max(thickness_values)
            max_velocity = np.max(velocity_values)

            # Calculate averages (considering zeroes)
            avg_thickness_incl_zeroes = np.mean(thickness_values)
            avg_velocity_incl_zeroes = np.mean(velocity_values)

            # Calculate averages (ignoring zeroes)
            avg_thickness_excl_zeroes = np.mean(thickness_values[thickness_values > 0]) if np.any(thickness_values > 0) else 0
            avg_velocity_excl_zeroes = np.mean(velocity_values[velocity_values > 0]) if np.any(velocity_values > 0) else 0

            # Store in metadata
            self.metadata['state_statistics'][state_number] = {
                'max_thickness': max_thickness,
                'max_velocity': max_velocity,
                'avg_thickness_incl_zeroes': avg_thickness_incl_zeroes,
                'avg_velocity_incl_zeroes': avg_velocity_incl_zeroes,
                'avg_thickness_excl_zeroes': avg_thickness_excl_zeroes,
                'avg_velocity_excl_zeroes': avg_velocity_excl_zeroes
        }


    def calculate_debris_metrics(self):
        """Calculates and stores various metrics for each state that could help identify a stopping condition."""
        # Initialize the metrics metadata
        self.metadata['debris_metrics'] = {}

        previous_occupied_area = None

        # Use the keys from thickness_state_array_dict as they should match the velocity_state_array_dict keys
        state_numbers = self.data['thickness_state_array_dict'].keys()

        # Iterate over states
        for state_number in state_numbers:
            thickness_values = self.data['thickness_state_array_dict'][state_number]
            velocity_values = self.data['velocity_state_array_dict'][state_number]

            # Calculate Occupied Area Change
            occupied_area = np.count_nonzero(thickness_values)
            occupied_area_change = self._calculate_occupied_area_change(previous_occupied_area, occupied_area)
            previous_occupied_area = occupied_area  # Update for next iteration

            # Calculate Standard Deviation of Velocity
            std_dev_velocity = np.std(velocity_values)

            # Calculate Standard Deviation of Thickness
            std_dev_thickness = np.std(thickness_values)

            # Calculate Temporal Stability
            # Assuming temporal stability is represented by the variability in the occupied area across frames.
            # This could be refined based on a more sophisticated definition of temporal stability.
            temporal_stability = 0 if state_number == min(state_numbers) else self.metadata['debris_metrics'][prev_state_number]['occupied_area_change']

            # Store in metadata
            self.metadata['debris_metrics'][state_number] = {
                'occupied_area_change': occupied_area_change,
                'std_dev_velocity': std_dev_velocity,
                'std_dev_thickness': std_dev_thickness,
                'temporal_stability': temporal_stability
            }
            prev_state_number = state_number  # Store the current state number for the next iteration


    def _calculate_occupied_area_change(self, previous_area, current_area):
        """Calculate the change in occupied area from the previous to the current state."""
        if previous_area is None:  # Handle the case where this is the first state
            return 0
        return current_area - previous_area

    
    def save_metadata(self):
        """Saves the metadata to a JSON file named with the model_id in the model_dir directory."""
        # Convert NumPy types to Python types for JSON serialization
        metadata_serializable = self._convert_to_serializable(self.metadata)

        # Create the filename using the model_id
        filename = f"{self.model_id}_metadata.json"
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

