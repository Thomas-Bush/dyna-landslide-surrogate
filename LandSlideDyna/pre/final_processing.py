import os
import json
import numpy as np
import pandas as pd

from collections import OrderedDict
from scipy.ndimage import zoom


class FinalProcessor:
    def __init__(self, root_directory, model_id, target_size, target_resolution, interpolation_order):
        """Initializes a Data instance to read in the intially processed data.
        
        Args:
            root_directory (str): The root directory where model data is stored.
            model_id (int): The unique identifier for the model.
        """
        self.model_id = model_id
        self.root_directory = root_directory
        self.target_res = target_resolution
        self.target_size = target_size
        self.interp_order = interpolation_order
        
        self.model_dir = os.path.join(self.root_directory, f'{self.model_id}')
        self.elevation_dir = os.path.join(self.model_dir, '03_initial_process', 'elevation')
        self.thickness_dir = os.path.join(self.model_dir, '03_initial_process', 'thickness')
        self.velocity_dir = os.path.join(self.model_dir, '03_initial_process', 'velocity')

        self.metadata_path = os.path.join(self.model_dir, f'{model_id}_metadata.json')

        # Data structure to hold the information about elevation, thickness, velocity, and states
        self.data = {
            'elevation_array_dict': {},
            'thickness_state_array_dict': OrderedDict(),
            'velocity_state_array_dict': OrderedDict(),
            'thickness_summary_data_dict': {},
            'velocity_summary_data_dict': {},
            'metadata_dict' : {}
        }
  
        # Final Processing Pipeline

        self.load_data()
        self.remove_blank_states()
        self.limit_velocity()
        self.match_resolution()
        self.crop_or_pad_arrays(self.target_size)


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
        self._load_metadata()

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

    def _load_metadata(self):
        """Load metadata from JSON files into a dict of dicts."""

        metadata_path = self.metadata_path

        if not os.path.isfile(metadata_path):
            raise FileNotFoundError(f"The file {metadata_path} does not exist.")

        with open(metadata_path, 'r') as file:
            metadata_dict = json.load(file)

        self.data['metadata_dict'] = metadata_dict

    
    
    def remove_blank_states(self):
        """Removes any state result arrays where both thickness and velocity are blank."""
        thickness_dict = self.data['thickness_state_array_dict']
        velocity_dict = self.data['velocity_state_array_dict']

        states_to_remove = []

        for state in thickness_dict:
            thickness_array = thickness_dict[state]
            velocity_array = velocity_dict[state]

            # Check if both arrays are all zeros
            if not thickness_array.any() and not velocity_array.any():
                states_to_remove.append(state)

        # Remove the identified states
        for state in states_to_remove:
            del thickness_dict[state]
            del velocity_dict[state]

        # Update the data dictionary with the cleaned states
        self.data['thickness_state_array_dict'] = thickness_dict
        self.data['velocity_state_array_dict'] = velocity_dict

    def limit_velocity(self, max_velocity=30):
        """Limit the maximum velocity value in each state's velocity array to a specified value.

        Args:
            max_velocity (float, optional): The maximum velocity value to enforce. Defaults to 30.
        """
        for state_key, velocity_array in self.data['velocity_state_array_dict'].items():
            # Clip the values at max_velocity
            self.data['velocity_state_array_dict'][state_key] = np.clip(velocity_array, None, max_velocity)

    def get_current_resolution(self):
        """Retrieve the current grid resolution from the metadata dictionary.
        
        Returns:
            float: The current grid resolution.
        """
        current_resolution_x = self.data['metadata_dict']['grid_resolution_x']
        current_resolution_y = self.data['metadata_dict']['grid_resolution_y']
        
        if current_resolution_x != current_resolution_y:
            print("Grid resolutions in X and Y are not the same.")
            return None
        
        return current_resolution_x

    def match_resolution(self):
        """Match the resolution of the underlying data to the target resolution."""
        current_resolution = self.get_current_resolution()
        target_resolution = self.target_res

        # If current resolution is the same as target, no need to resample
        if current_resolution == target_resolution:
            return

        # Resample the elevation array since it's not dependent on state
        elevation_array = self.data['elevation_array_dict']['z_values']
        self.data['elevation_array_dict']['z_values'] = self.resample_grid(
            elevation_array, current_resolution, target_resolution)

        # Now loop through all thickness and velocity arrays and resample
        for state_key in self.data['thickness_state_array_dict']:
            thickness_array = self.data['thickness_state_array_dict'][state_key]
            velocity_array = self.data['velocity_state_array_dict'][state_key]

            self.data['thickness_state_array_dict'][state_key] = self.resample_grid(
                thickness_array, current_resolution, target_resolution)
            self.data['velocity_state_array_dict'][state_key] = self.resample_grid(
                velocity_array, current_resolution, target_resolution)

    def resample_grid(self, grid_data, current_resolution, target_resolution):
        """Resample a grid to the target resolution.

        Args:
            grid_data (numpy.ndarray): The grid data to be resampled.
            current_resolution (float): The current resolution of the grid_data.
            target_resolution (float): The target resolution to resample the grid_data to.

        Returns:
            numpy.ndarray: The resampled grid data.
        """
        zoom_factor = current_resolution / target_resolution
        # Use zoom factor to upsample or downsample the grid
        resampled_grid = zoom(grid_data, zoom_factor, order=self.interp_order)

        return resampled_grid

    # def calculate_bounding_boxes(self):
    #     """Recalculates and stores the bounding box of the debris for every state after interpolation."""
    #     # Initialize the bounding boxes metadata
    #     self.data['metadata_dict']['bounding_boxes'] = {}

    #     # Variables to track the extremities for the overall bounding box
    #     global_min_x = float('inf')
    #     global_max_x = -float('inf')
    #     global_min_y = float('inf')
    #     global_max_y = -float('inf')

    #     # Iterate over states using keys from thickness_state_array_dict and velocity_state_array_dict
    #     for state_number in self.data['thickness_state_array_dict']:
    #         thickness_array = self.data['thickness_state_array_dict'][state_number]
    #         velocity_array = self.data['velocity_state_array_dict'][state_number]

    #         # Find the indices where either thickness or velocity are non-zero
    #         nonzero_thickness_indices = np.nonzero(thickness_array)
    #         nonzero_velocity_indices = np.nonzero(velocity_array)
    #         nonzero_indices = np.unique(np.hstack((nonzero_thickness_indices, nonzero_velocity_indices)), axis=1)

    #         if nonzero_indices.size > 0:
    #             # Calculate bounding box
    #             min_x = np.min(nonzero_indices[1])
    #             max_x = np.max(nonzero_indices[1])
    #             min_y = np.min(nonzero_indices[0])
    #             max_y = np.max(nonzero_indices[0])

    #             # Update global bounding box values
    #             global_min_x = min(min_x, global_min_x)
    #             global_max_x = max(max_x, global_max_x)
    #             global_min_y = min(min_y, global_min_y)
    #             global_max_y = max(max_y, global_max_y)

    #             # Calculate dimensions of the bounding box
    #             bbox_width = max_x - min_x + 1  # +1 to include the starting index
    #             bbox_height = max_y - min_y + 1  # +1 to include the starting index

    #             # Store in metadata
    #             self.metadata['bounding_boxes'][state_number] = {
    #                 'min_x': min_x, 'max_x': max_x,
    #                 'min_y': min_y, 'max_y': max_y,
    #                 'width': bbox_width, 'height': bbox_height,
    #                 'area': bbox_width * bbox_height
    #             }
    #         else:
    #             # If no debris is present, store None or an appropriate representation
    #             self.metadata['bounding_boxes'][state_number] = None

    #     # Store the overall bounding box
    #     self.metadata['overall_bounding_box'] = {
    #         'min_x': global_min_x, 'max_x': global_max_x,
    #         'min_y': global_min_y, 'max_y': global_max_y,
    #         'width': global_max_x - global_min_x + 1,
    #         'height': global_max_y - global_min_y + 1
    #     }

    #     # Calculate and store the center point for the overall bounding box
    #     self.metadata['overall_bounding_box']['center_x'] = (global_min_x + global_max_x) / 2
    #     self.metadata['overall_bounding_box']['center_y'] = (global_min_y + global_max_y) / 2

    def calculate_bounding_boxes(self):
        """Recalculates and stores the bounding box of the debris for every state after interpolation."""
        # Initialize the bounding boxes metadata
        self.data['metadata_dict']['bounding_boxes'] = {}

        # Variables to track the extremities for the overall bounding box
        global_min_x = float('inf')
        global_max_x = -float('inf')
        global_min_y = float('inf')
        global_max_y = -float('inf')

        # Iterate over states using keys from thickness_state_array_dict and velocity_state_array_dict
        for state_number in self.data['thickness_state_array_dict']:
            thickness_array = self.data['thickness_state_array_dict'][state_number]
            velocity_array = self.data['velocity_state_array_dict'][state_number]

            # Find the indices where either thickness or velocity are non-zero
            nonzero_thickness_indices = np.nonzero(thickness_array)
            nonzero_velocity_indices = np.nonzero(velocity_array)
            nonzero_indices = np.unique(np.hstack((nonzero_thickness_indices, nonzero_velocity_indices)), axis=1)

            if nonzero_indices.size > 0:
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
                self.data['metadata_dict']['bounding_boxes'][state_number] = {
                    'min_x': min_x, 'max_x': max_x,
                    'min_y': min_y, 'max_y': max_y,
                    'width': bbox_width, 'height': bbox_height,
                    'area': bbox_width * bbox_height
                }
            else:
                # If no debris is present, store None or an appropriate representation
                self.data['metadata_dict']['bounding_boxes'][state_number] = None

        # Store the overall bounding box
        self.data['metadata_dict']['overall_bounding_box'] = {
            'min_x': global_min_x, 'max_x': global_max_x,
            'min_y': global_min_y, 'max_y': global_max_y,
            'width': global_max_x - global_min_x + 1,
            'height': global_max_y - global_min_y + 1
        }

        # Calculate and store the center point for the overall bounding box
        self.data['metadata_dict']['overall_bounding_box']['center_x'] = (global_min_x + global_max_x) / 2
        self.data['metadata_dict']['overall_bounding_box']['center_y'] = (global_min_y + global_max_y) / 2

    def crop_or_pad_array_to_fixed_size(self, array, target_size=512, overall_bbox_center=None):
        """Crop or pad the input array to a target size centered around the overall bounding box center.

        Args:
            array (np.ndarray): The input 2D array to be cropped or padded.
            target_size (int): The desired size for both dimensions of the output array.
            overall_bbox_center (tuple): The center of the overall bounding box (x, y).

        Returns:
            np.ndarray: The cropped or padded array.
        """
        if overall_bbox_center is None:
            raise ValueError("Overall bounding box center must be provided")

        center_x, center_y = overall_bbox_center
        current_height, current_width = array.shape
        new_array = np.zeros((target_size, target_size), dtype=array.dtype)

        # Calculate the cropping box centered around the overall bounding box center
        crop_x1 = max(0, int(center_x - target_size // 2))
        crop_x2 = min(current_width, int(center_x + target_size // 2))
        crop_y1 = max(0, int(center_y - target_size // 2))
        crop_y2 = min(current_height, int(center_y + target_size // 2))

        # Calculate the position where the array will be placed in the new_array
        paste_x1 = max(0, target_size // 2 - int(center_x))
        paste_x2 = min(target_size, target_size // 2 + current_width - int(center_x))
        paste_y1 = max(0, target_size // 2 - int(center_y))
        paste_y2 = min(target_size, target_size // 2 + current_height - int(center_y))

        # Paste the cropped array into the new array
        new_array[paste_y1:paste_y2, paste_x1:paste_x2] = array[crop_y1:crop_y2, crop_x1:crop_x2]

        return new_array

    # def crop_or_pad_arrays(self, target_size=512):
    #     """Crops or pads all arrays to a target size based on the overall bounding box center."""
    #     self.calculate_bounding_boxes()
    #     overall_bbox_center = (
    #         self.data['metadata_dict']['overall_bounding_box']['center_x'],
    #         self.data['metadata_dict']['overall_bounding_box']['center_y']
    #     )

    #     # Crop or pad elevation array
    #     for key, elevation_array in self.data['elevation_array_dict'].items():
    #         self.data['elevation_array_dict'][key] = self.crop_or_pad_array_to_fixed_size(
    #             elevation_array, target_size, overall_bbox_center)

    #     # Crop or pad thickness and velocity arrays for each state
    #     for key, thickness_array in self.data['thickness_state_array_dict'].items():
    #         self.data['thickness_state_array_dict'][key] = self.crop_or_pad_array_to_fixed_size(
    #             thickness_array, target_size, overall_bbox_center)

    #     for key, velocity_array in self.data['velocity_state_array_dict'].items():
    #         self.data['velocity_state_array_dict'][key] = self.crop_or_pad_array_to_fixed_size(
    #             velocity_array, target_size, overall_bbox_center)

    #     # The cropped or padded arrays will now be stored within self.data


    def crop_or_pad_arrays(self, target_size=512):
        """Crops or pads all arrays to a target size based on the overall bounding box center."""
        self.calculate_bounding_boxes()
        overall_bbox_center = (
            self.data['metadata_dict']['overall_bounding_box']['center_x'],
            self.data['metadata_dict']['overall_bounding_box']['center_y']
        )

        # Crop or pad elevation array
        elevation_array = self.data['elevation_array_dict']['z_values']
        self.data['elevation_array_dict']['z_values'] = self.crop_or_pad_array_to_fixed_size(
            elevation_array, target_size, overall_bbox_center)

        # Crop or pad thickness and velocity arrays for each state
        for key, thickness_array in self.data['thickness_state_array_dict'].items():
            self.data['thickness_state_array_dict'][key] = self.crop_or_pad_array_to_fixed_size(
                thickness_array, target_size, overall_bbox_center)

        for key, velocity_array in self.data['velocity_state_array_dict'].items():
            self.data['velocity_state_array_dict'][key] = self.crop_or_pad_array_to_fixed_size(
                velocity_array, target_size, overall_bbox_center)

    # The cropped or padded arrays will now be stored within self.data

    def export_data(self, base_path):
        """Export elevation, velocity, and thickness arrays as .npy files for the model,
        rounding all values to 2 decimal places.

        Args:
            base_path (str): The base directory where the model's "04_FinalProcessedData"
                             subfolder will be created.
        """
        # Create the "04_FinalProcessedData" subdirectory within the model's directory
        final_data_path = os.path.join(base_path, str(self.model_id), f"04_FinalProcessedData_{self.target_size}")
        elevation_path = os.path.join(final_data_path, 'elevation')
        velocity_path = os.path.join(final_data_path, 'velocity')
        thickness_path = os.path.join(final_data_path, 'thickness')

        for path in [elevation_path, velocity_path, thickness_path]:
            os.makedirs(path, exist_ok=True)

        # Round and save elevation array with model_id in the filename
        elevation_array = np.around(self.data['elevation_array_dict']['z_values'], decimals=2)
        elevation_file_path = os.path.join(elevation_path, f'{self.model_id}_elevation.npy')
        np.save(elevation_file_path, elevation_array)

        # Round and save velocity and thickness arrays with model_id in the filenames
        for state_key in self.data['velocity_state_array_dict']:
            velocity_array = np.around(self.data['velocity_state_array_dict'][state_key], decimals=2)
            thickness_array = np.around(self.data['thickness_state_array_dict'][state_key], decimals=2)

            velocity_file_path = os.path.join(velocity_path, f'{self.model_id}_velocity_{state_key}.npy')
            thickness_file_path = os.path.join(thickness_path, f'{self.model_id}_thickness_{state_key}.npy')

            np.save(velocity_file_path, velocity_array)
            np.save(thickness_file_path, thickness_array)