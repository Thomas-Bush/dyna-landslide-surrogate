import os
import json
import pandas as pd
import numpy as np
from functools import partial

class Data:
    def __init__(self, raw_data_path):
        """Initialize Data with the directory path.
        
        Args:
            raw_data_path (str): The directory path where the raw data CSV files are stored.
        """
        self.raw_data_path = raw_data_path
        self.filepaths = self._build_file_paths()
        self.converters = self._define_converters()
        self.data = self.read_data()

    def _build_file_paths(self):
        """Constructs the file paths for the raw data CSVs within the subfolder."""
        parent_dir = os.path.dirname(os.path.normpath(self.raw_data_path))
        folder_name = os.path.basename(parent_dir)
        
        file_paths = {
            'nodes': os.path.join(self.raw_data_path, f"{folder_name}_nodes.csv"),
            'shells': os.path.join(self.raw_data_path, f"{folder_name}_shells.csv"),
            'solids': os.path.join(self.raw_data_path, f"{folder_name}_solids.csv"),
            'nodal_velocities': os.path.join(self.raw_data_path, f"{folder_name}_nodal_velocities.csv"),
            'solid_thicknesses': os.path.join(self.raw_data_path, f"{folder_name}_solid_thicknesses.csv"),
            'states': os.path.join(self.raw_data_path, f"{folder_name}_states.csv")
        }
        return file_paths

    def _define_converters(self):
        """Defines converters for data preprocessing on import."""
        # Rounding function
        def round_to_n_decimals(x, n=2):
            return round(float(x), n)
        
        # XYZ rounding converter (applied to nodes)
        xyz_round_converter = {col_name: partial(round_to_n_decimals, n=2) for col_name in ['Node_X', 'Node_Y', 'Node_Z']}
        
        # Nodal velocities and solid thickness converters (applied to nodal velocities and solid thicknesses)
        def create_rounding_converter(csv_path, exclude_cols):
            with open(csv_path, 'r') as file:
                headers = file.readline().strip().split(',')
            return {col_name: partial(round_to_n_decimals, n=2) for col_name in headers if col_name not in exclude_cols}
        
        converters = {
            'nodes': xyz_round_converter,
            'nodal_velocities': create_rounding_converter(self.filepaths['nodal_velocities'], ['Node_Label']),
            'solid_thicknesses': create_rounding_converter(self.filepaths['solid_thicknesses'], ['Solid_Label'])
        }
        return converters

    def read_data(self):
        """Reads the CSV files using the appropriate converters and returns the DataFrames."""
        data = {}
        
        # Load the CSV files into pandas DataFrames using converters and required columns
        data['nodes'] = pd.read_csv(self.filepaths['nodes'], converters=self.converters['nodes'])
        data['shells'] = pd.read_csv(self.filepaths['shells'])
        data['solids'] = pd.read_csv(self.filepaths['solids'])
        data['nodal_velocities'] = pd.read_csv(self.filepaths['nodal_velocities'], converters=self.converters['nodal_velocities'])
        data['solid_thicknesses'] = pd.read_csv(self.filepaths['solid_thicknesses'], converters=self.converters['solid_thicknesses'])
        data['states'] = pd.read_csv(self.filepaths['states'])
        
        # Set Node_Label as the index for easier lookup
        data['nodes'].set_index('Node_Label', inplace=True)
        data['nodal_velocities'].set_index('Node_Label', inplace=True)
        
        return data

    def set_processed_data(self, key, value):
        """Set the processed data with a specific key.

        Args:
            key (str): The key corresponding to the processed data.
            value: The processed data to be stored.
        """
        self.data[key] = value

    def get(self, key):
        """Get the processed data with a specific key.

        Args:
            key (str): The key corresponding to the processed data.

        Returns:
            The processed data associated with the given key.
        """
        return self.data.get(key, None)


class NodalStatisticsProcessor:
    """Processor class for calculating nodal statistics for finite element models."""

    @staticmethod
    def calculate_nodal_statistics(element_df, node_df, node_columns, stat_columns, stat_func):
        """Calculate statistics for nodes associated with finite elements.

        Args:
            element_df (pd.DataFrame): DataFrame containing finite element data.
            node_df (pd.DataFrame): DataFrame containing nodal data.
            node_columns (list): List of columns in element_df that reference nodes.
            stat_columns (list): List of columns in node_df for which to calculate statistics.
            stat_func (function): Statistical function to apply to nodal data.

        Returns:
            np.ndarray: Array of calculated statistics.
        """
        extracted_node_values = pd.DataFrame(index=element_df.index)

        for node_col in node_columns:
            temp_df = element_df[node_col].map(node_df[stat_columns].to_dict(orient='index'))
            extracted_node_values = pd.concat([extracted_node_values, pd.DataFrame(list(temp_df.values), index=temp_df.index)], axis=1)

        num_nodes = len(node_columns)
        num_statistics = len(stat_columns)
        reshaped_values = extracted_node_values.values.reshape(-1, num_nodes, num_statistics)
        statistics = stat_func(reshaped_values, axis=1)

        return statistics

class TopographyProcessor:
    """Processor class for handling topography data."""

    def __init__(self, data):
        """Initialize the TopographyProcessor with a Data instance.

        Args:
            data (Data): An instance of the Data class containing dataframes.
        """
        self.data = data
        self.process_topography()

    def process_topography(self):
        """Process topography data to calculate the center points of shell elements.

        Returns:
            pd.DataFrame: Processed DataFrame with topographical data.
        """
        # Access the shells and nodes dataframes from the data
        shells_df = self.data.data['shells']
        nodes_df = self.data.data['nodes']

        # Calculate the mean X, Y, and Z (center point) for each shell element
        shell_centre_points = NodalStatisticsProcessor.calculate_nodal_statistics(
            element_df=shells_df,
            node_df=nodes_df,
            node_columns=['Shell_N1', 'Shell_N2', 'Shell_N3', 'Shell_N4'],
            stat_columns=['Node_X', 'Node_Y', 'Node_Z'],
            stat_func=np.mean
        )

        # Split the mean_positions array into separate columns and add to shells_df
        shells_df[['Shell_X', 'Shell_Y', 'Shell_Z']] = shell_centre_points

        # Filter shells_df to include only parts with 'Shell_Part' equal to 1
        filtered_shells_df = shells_df[shells_df['Shell_Part'] == 1]

        # Select relevant columns and copy to create a new dataframe
        topo_shells_df = filtered_shells_df[['Shell_Label', 'Shell_X', 'Shell_Y', 'Shell_Z']].copy()

        # Create a Group_ID by concatenating the 'Shell_X' and 'Shell_Y' as strings
        topo_shells_df['Group_ID'] = topo_shells_df['Shell_X'].astype(str) + '_' + topo_shells_df['Shell_Y'].astype(str)

        # Store the processed topography data back into the data instance
        self.data.set_processed_data('topo_shells', topo_shells_df)


class DebrisProcessor:
    """Processor class for handling debris data."""

    def __init__(self, data):
        """Initialize the DebrisProcessor with a Data instance.

        Args:
            data (Data): An instance of the Data class containing dataframes.
        """
        self.data = data
        self.calculate_solid_centre_points()
        self.calculate_max_nodal_velocity_per_solid()
        self.group_solid_data_by_xy('velocity', 'max')
        self.group_solid_data_by_xy('thickness', 'sum')

    def calculate_solid_centre_points(self):
        """Calculate the center points of the solids and update the solids DataFrame with this information."""
        solids_df = self.data.data['solids']
        nodes_df = self.data.data['nodes']

        solid_centre_points = NodalStatisticsProcessor.calculate_nodal_statistics(
            element_df=solids_df,
            node_df=nodes_df,
            node_columns=['Solid_N1', 'Solid_N2', 'Solid_N3', 'Solid_N4', 'Solid_N5', 'Solid_N6', 'Solid_N7', 'Solid_N8'],
            stat_columns=['Node_X', 'Node_Y', 'Node_Z'],
            stat_func=np.mean
        )

        solids_df[['Solid_X', 'Solid_Y', 'Solid_Z']] = solid_centre_points

        self.data.set_processed_data('solids', solids_df)

    def calculate_max_nodal_velocity_per_solid(self):
        """Calculate the maximum nodal velocity for each solid and update the solids DataFrame with this information."""
        solids_df = self.data.data['solids']
        nodal_vels_df = self.data.data['nodal_velocities']

        timestep_columns = [col for col in nodal_vels_df.columns if col != 'Node_Label']

        max_nodal_velocities = NodalStatisticsProcessor.calculate_nodal_statistics(
            element_df=solids_df,
            node_df=nodal_vels_df,
            node_columns=['Solid_N1', 'Solid_N2', 'Solid_N3', 'Solid_N4', 'Solid_N5', 'Solid_N6', 'Solid_N7', 'Solid_N8'],
            stat_columns=timestep_columns,
            stat_func=np.max
        )

        max_velocity_dict = {col: max_nodal_velocities[:, i] for i, col in enumerate(timestep_columns)}

        max_velocity_df = pd.DataFrame(max_velocity_dict)
        solids_vels_df = pd.concat([solids_df[['Solid_Label']], max_velocity_df], axis=1)

        self.data.set_processed_data('solids_velocity', solids_vels_df)

    def group_solid_data_by_xy(self, data_type, agg_func):
        """Group solids with the same XY and aggregate their data using the specified function.

        Args:
            data_type (str): Type of the data to group by ('velocity' or 'thickness').
            agg_func (str): The aggregation function to use as a string ('max' or 'sum').
        """
        if data_type == 'velocity':
            df = self.data.data['solids_velocity']
            value_column = 'Velocity'
        elif data_type == 'thickness':
            df = self.data.data['solid_thicknesses']
            value_column = 'Thickness'
        else:
            raise ValueError(f"Invalid data_type: {data_type}")

        df = df.merge(
            self.data.data['solids'][['Solid_Label', 'Solid_X', 'Solid_Y']],
            on='Solid_Label',
            how='left'
        )

        df['Group_ID'] = df['Solid_X'].astype(str) + '_' + df['Solid_Y'].astype(str)

        df = df.drop('Solid_Label', axis=1)

        agg_funcs = {col: agg_func for col in df.columns if col not in ['Solid_X', 'Solid_Y', 'Group_ID']}
        agg_funcs['Solid_X'] = 'first'
        agg_funcs['Solid_Y'] = 'first'

        df = df.groupby('Group_ID').agg(agg_funcs).reset_index()

        cols = ['Group_ID', 'Solid_X', 'Solid_Y'] + [col for col in df.columns if col not in ['Group_ID', 'Solid_X', 'Solid_Y']]
        df = df[cols]

        if data_type == 'velocity':
            self.data.set_processed_data('grouped_velocity', df)
        elif data_type == 'thickness':
            self.data.set_processed_data('grouped_thickness', df)


class DataPadder:
    """Class for padding debris results dataframes to match topography extents."""

    def __init__(self, data, group_id_col='Group_ID', fill_value=0):
        """Initialize the DataPadder with a Data instance and topography dataframe.

        Args:
            data (Data): An instance of the Data class containing dataframes.
            group_id_col (str): Column name for the group ID used to merge the DataFrames.
            fill_value (int, float): The value to fill in for missing state data.
        """
        self.data = data

        self.group_id_col = group_id_col
        self.fill_value = fill_value

        # Pad and update the data for all relevant dataframes
        self.pad_and_update('grouped_velocity')
        self.pad_and_update('grouped_thickness')

    def pad_and_update(self, dataframe_key):
        """Pad a results dataframe and update it in the data instance.

        Args:
            dataframe_key (str): The key of the dataframe to be padded.
        """
        topo_df = self.data.data['topo_shells']
        results_df = self.data.data[dataframe_key]

        if results_df is not None:
            padded_df = self.pad_results_to_match_topo(topo_df, results_df)
            self.data.set_processed_data(f'padded_{dataframe_key}', padded_df)

    def pad_results_to_match_topo(self, topo_df, results_df):
        """Pad the results DataFrame to match the topography DataFrame extents.

        Args:
            topo_df (pd.DataFrame): DataFrame containing topography data.
            results_df (pd.DataFrame): DataFrame containing results data.

        Returns:
            pd.DataFrame: Padded results DataFrame.
        """
        # Merge the topography and results DataFrame on the Group_ID column with an outer join
        merged_df = pd.merge(topo_df, results_df, how='outer', on=self.group_id_col)

        # Fill NaN values for 'Solid_X' and 'Solid_Y' in the results data
        # with the corresponding 'Shell_X' and 'Shell_Y' values from the topo data
        merged_df['Solid_X'] = merged_df['Solid_X'].fillna(merged_df['Shell_X'])
        merged_df['Solid_Y'] = merged_df['Solid_Y'].fillna(merged_df['Shell_Y'])

        # Fill NaN values in the state columns with the specified fill value
        state_cols = [col for col in merged_df.columns if col.startswith('State_')]
        merged_df[state_cols] = merged_df[state_cols].fillna(self.fill_value)

        # Drop the shell-related columns from the merged DataFrame
        shell_cols = ['Shell_Label', 'Shell_X', 'Shell_Y', 'Shell_Z']
        merged_df.drop(columns=shell_cols, inplace=True)

        return merged_df


class ArrayConverter:
    """Class for converting dataframes to arrays and updating the main Data class."""

    def __init__(self, data):
        """Initialize the ArrayConverter with a Data instance.

        Args:
            data (Data): An instance of the Data class containing dataframes and arrays.
        """
        self.data = data
        self.convert_and_update_data()

    def convert_and_update_data(self):
        """Perform all data conversions and updates the Data instance."""
        self.update_elevation_array()
        self.update_and_flatten_state_arrays('padded_grouped_velocity', 'velocity')
        self.update_and_flatten_state_arrays('padded_grouped_thickness', 'thickness')

    def update_elevation_array(self, elevation_array_key='elevation_array_dict'):
        """Convert topo dataframe to elevation array and update Data instance."""
        topo_df = self.data.data['topo_shells']
        elevation_dict = self.dataframe_to_elevation_dict(topo_df)
        self.data.set_processed_data(f"{elevation_array_key}", elevation_dict)

    def update_and_flatten_state_arrays(self, results_df_key, state_type):
        """Convert results dataframe to state arrays, flatten to max values and update Data instance.

        Args:
            results_df_key (str): The key of the results dataframe in the Data instance.
            state_type (str): The type of state being processed ('velocity' or 'thickness').
        """
        results_df = self.data.data[results_df_key]
        state_dict = self.dataframe_to_state_dict(results_df)
        # Flatten the state arrays to their maximum values
        flattened_array = self.flatten_to_max_value(state_dict['result_arrays'])
        # Update the Data instance with the new flattened array
        self.data.set_processed_data(f'{state_type}_max_value_array', flattened_array)
        # Also keep the original state arrays dictionary
        self.data.set_processed_data(f'{state_type}_state_arrays_dict', state_dict)


    def update_state_arrays(self, results_df_key, state_type):
        """Convert results dataframe to state arrays for thickness or velocity and update Data instance.

        Args:
            results_df_key (str): The key of the results dataframe in the Data instance.
            state_type (str): The type of state being processed ('velocity' or 'thickness').
        """
        results_df = self.data.data[results_df_key]
        state_dict = self.dataframe_to_state_dict(results_df)
        self.data.set_processed_data(f'{state_type}_state_arrays_dict', state_dict)

    @staticmethod
    def dataframe_to_elevation_dict(df, x_col='Shell_X', y_col='Shell_Y', z_col='Shell_Z'):
        """Convert a DataFrame with positions and elevation values into a dictionary.

        Args:
            df (pd.DataFrame): Input DataFrame with positions and elevation values.
            x_col (str): Name of the column representing the X position.
            y_col (str): Name of the column representing the Y position.
            z_col (str): Name of the column representing the elevation value.

        Returns:
            dict: A dictionary containing the z_values, x_values, and y_values as numpy arrays.
        """
        elevation_array, unique_x, unique_y = ArrayConverter.dataframe_to_elevation_array(df, x_col, y_col, z_col)
        elevation_dict = {
            'z_values': elevation_array,
            'x_values': unique_x,
            'y_values': unique_y,
        }
        return elevation_dict

    @staticmethod
    def dataframe_to_state_dict(df, x_col='Solid_X', y_col='Solid_Y', state_cols_prefix='State_'):
        """Convert a DataFrame with positions and state values into a dictionary.

        Args:
            df (pd.DataFrame): Input DataFrame with positions and state values.
            x_col (str): Name of the column representing the X position.
            y_col (str): Name of the column representing the Y position.
            state_cols_prefix (str): Prefix of the columns representing state values.

        Returns:
            dict: A dictionary containing result_arrays, state_labels, x_values, and y_values.
        """
        state_arrays, state_labels, unique_x, unique_y = ArrayConverter.dataframe_to_state_arrays(df, x_col, y_col, state_cols_prefix)
        state_dict = {
            'result_arrays': state_arrays,
            'state_labels': state_labels,
            'x_values': unique_x,
            'y_values': unique_y,
        }
        return state_dict

    @staticmethod
    def dataframe_to_elevation_array(df, x_col='Shell_X', y_col='Shell_Y', z_col='Shell_Z'):
        """
        Convert a DataFrame with positions and elevation values into a 2D NumPy array.

        Args:
        df (pd.DataFrame): Input DataFrame with positions and elevation values.
        x_col (str): Name of the column representing the X position.
        y_col (str): Name of the column representing the Y position.
        z_col (str): Name of the column representing the elevation value.

        Returns:
        tuple: A tuple containing:
            - np.ndarray: 2D NumPy array of elevation values.
            - np.ndarray: Array of unique X coordinates.
            - np.ndarray: Array of unique Y coordinates.
        """
        # Get unique X and Y positions
        unique_x = np.sort(df[x_col].unique())
        unique_y = np.sort(df[y_col].unique())

        # Create a mapping from position value to index
        x_positions = {x: i for i, x in enumerate(unique_x)}
        y_positions = {y: i for i, y in enumerate(unique_y)}

        # Initialize the elevation array
        elevation_array = np.zeros((len(unique_y), len(unique_x)))

        # Fill the elevation array with the corresponding values from the DataFrame
        for _, row in df.iterrows():
            x_idx = x_positions[row[x_col]]
            y_idx = y_positions[row[y_col]]
            elevation_array[y_idx, x_idx] = row[z_col]

        return elevation_array, unique_x, unique_y

    @staticmethod
    def dataframe_to_state_arrays(df, x_col='Solid_X', y_col='Solid_Y', state_cols_prefix='State_'):
        """
        Convert a DataFrame with positions and state values into a series of 2D NumPy arrays,
        return the unique X and Y coordinates, and the state labels as well.

        Args:
            df (pd.DataFrame): Input DataFrame with positions and state values.
            x_col (str): Name of the column representing the X position.
            y_col (str): Name of the column representing the Y position.
            state_cols_prefix (str): Prefix of the columns representing state values.

        Returns:
            tuple: A tuple containing:
                - List[np.ndarray]: List of 2D NumPy arrays for each state.
                - List[str]: List of labels for each state array.
                - np.ndarray: Array of unique X coordinates.
                - np.ndarray: Array of unique Y coordinates.
        """
        # Get unique X and Y positions
        unique_x = np.sort(df[x_col].unique())
        unique_y = np.sort(df[y_col].unique())

        # Create a mapping from position value to index
        x_positions = {x: i for i, x in enumerate(unique_x)}
        y_positions = {y: i for i, y in enumerate(unique_y)}

        # Get the state columns and their labels
        state_cols = [col for col in df.columns if col.startswith(state_cols_prefix)]
        state_labels = [col.replace(state_cols_prefix, '') for col in state_cols]

        # Initialize a list to hold the state arrays
        state_arrays = [np.zeros((len(unique_y), len(unique_x))) for _ in state_cols]

        # Fill the state arrays with the corresponding values from the DataFrame
        for _, row in df.iterrows():
            x_idx = x_positions[row[x_col]]
            y_idx = y_positions[row[y_col]]
            for i, state_col in enumerate(state_cols):
                state_arrays[i][y_idx, x_idx] = row[state_col]

        return state_arrays, state_labels, unique_x, unique_y
    
    @staticmethod
    def flatten_to_max_value(state_arrays):
        """Flatten a list of 2D state arrays into a single array, keeping the maximum value.

        Args:
            state_arrays (List[np.ndarray]): List of 2D NumPy arrays for different states.

        Returns:
            np.ndarray: A 2D NumPy array with the maximum value at each location.
        """
        # Stack the arrays along a new dimension and then take the max along this new dimension
        max_value_array = np.max(np.stack(state_arrays), axis=0)
        return max_value_array


class ModelMetadata:
    """Class to handle metadata about the model."""

    def __init__(self, model_id, data):
        self.metadata = {
            'model_id': model_id,
            'total_number_of_states': None,
            'min_x_value': None,
            'max_x_value': None,
            'min_y_value': None,
            'max_y_value': None,
            'min_z_value': None,
            'max_z_value': None,
            'grid_resolution_x': None,
            'grid_resolution_y': None,
            'timesteps': {},
            'average_timestep': None,
        }
        self.data = data
        self.calculate_metadata()

    def calculate_metadata(self):
        """Calculate and update the metadata based on the provided data."""
       
        self.metadata['min_x_value'] = np.min(self.data.get('elevation_array_dict')['x_values'])
        self.metadata['max_x_value'] = np.max(self.data.get('elevation_array_dict')['x_values'])
        self.metadata['min_y_value'] = np.min(self.data.get('elevation_array_dict')['y_values'])
        self.metadata['max_y_value'] = np.max(self.data.get('elevation_array_dict')['y_values'])
        self.metadata['min_z_value'] = np.min(self.data.get('elevation_array_dict')['z_values'])
        self.metadata['max_z_value'] = np.max(self.data.get('elevation_array_dict')['z_values'])
        self.metadata['grid_resolution_x'] = self.calculate_grid_resolution(self.data.get('elevation_array_dict')['x_values'])
        self.metadata['grid_resolution_y'] = self.calculate_grid_resolution(self.data.get('elevation_array_dict')['y_values'])     
        states_df = self.data.get('states')
        self.metadata['total_number_of_states'] = states_df.shape[0]        
        self.metadata['average_timestep'] = self.calculate_average_timestep(states_df['State_Timestamp'])
        self.metadata['timesteps'] = dict(zip(states_df['State_Label'], states_df['State_Timestamp']))
        

    def calculate_grid_resolution(self, values):
        """Calculate the grid resolution based on the axis values.

        Args:
            values (list or np.array): The values along an axis.

        Returns:
            The calculated grid resolution.
        """
        # Assuming that 'values' is sorted and uniformly spaced
        return np.diff(values).mean() if len(values) > 1 else None

    def calculate_average_timestep(self, timestamps):
        """Calculate the average timestep from a sequence of timestamps.

        Args:
            timestamps (pd.Series): Timestamps for each state.

        Returns:
            float: The average timestep calculated as the mean of the difference between sequential timestamps.
        """
        sorted_timestamps = timestamps.sort_values().values
        if len(sorted_timestamps) > 1:
            timestep_differences = np.diff(sorted_timestamps)
            return np.mean(timestep_differences)
        return None
    
    
    
    def save_to_json(self, file_path):
        """Save the metadata to a JSON file.

        Args:
            file_path (str): The path to the JSON file where metadata will be saved.
        """
        with open(file_path, 'w') as outfile:
            json.dump(self.metadata, outfile, indent=4)



class InitialProcessPipeline:
    def __init__(self, model_id, raw_data_path, processed_data_path):
        self.model_id = model_id
        self.processed_data_path = processed_data_path

        self.data = Data(raw_data_path)
        self.topo_processor = TopographyProcessor(self.data)  # Process topography data
        self.debris_processor = DebrisProcessor(self.data)  # Process debris data
        self.data_padder = DataPadder(self.data)  # Pad results data to match topography data
        self.array_converter = ArrayConverter(self.data)  # Convert topography & result dfs to arrays
        
    def run(self):
        """The main method to run the entire processing pipeline."""
        if self.processed_data_path:
            self.export_processed_data()
            self.export_metadata()

    def get_data(self, key):
        """Retrieve data using the key from the Data instance.

        Args:
            key (str): The key corresponding to the processed data.

        Returns:
            The processed data associated with the given key.
        """
        return self.data.get(key)

    def _export_array(self, array, subdir, filename):
        """Helper function to export a NumPy array to a file.

        Args:
            array (np.array): The array to be exported.
            subdir (str): Subdirectory to place the file in.
            filename (str): The name of the file to save the array to.
        """
        dir_path = os.path.join(self.processed_data_path, subdir)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        file_path = os.path.join(dir_path, filename)
        np.save(file_path, array)

    def export_processed_data(self):
        """Export processed data to the specified directory structure."""
        # Export elevation data
        elevation_data = self.get_data('elevation_array_dict')
        for key in ['z_values', 'x_values', 'y_values']:
            filename = f"{self.model_id}_elevation_{key}.npy"
            self._export_array(elevation_data[key], 'elevation', filename)

        # Export velocity data
        velocity_data = self.get_data('velocity_state_arrays_dict')
        for label, array in zip(velocity_data['state_labels'], velocity_data['result_arrays']):
            filename = f"{self.model_id}_velocity_{label}.npy"
            self._export_array(array, 'velocity', filename)

        for key in ['x_values', 'y_values']:
            filename = f"{self.model_id}_velocity_{key}.npy"
            self._export_array(velocity_data[key], 'velocity', filename)

        # Export max velocity value
        max_velocity_array = self.get_data('velocity_max_value_array')
        filename = f"{self.model_id}_velocity_max_value.npy"
        self._export_array(max_velocity_array, 'velocity', filename)

        # Export thickness data using the same logic as velocity
        thickness_data = self.get_data('thickness_state_arrays_dict')
        for label, array in zip(thickness_data['state_labels'], thickness_data['result_arrays']):
            filename = f"{self.model_id}_thickness_{label}.npy"
            self._export_array(array, 'thickness', filename)

        for key in ['x_values', 'y_values']:
            filename = f"{self.model_id}_thickness_{key}.npy"
            self._export_array(thickness_data[key], 'thickness', filename)

        # Export max thickness value
        max_thickness_array = self.get_data('thickness_max_value_array')
        filename = f"{self.model_id}_thickness_max_value.npy"
        self._export_array(max_thickness_array, 'thickness', filename)

    def export_metadata(self):
        """Export metadata as a JSON file."""
        metadata = ModelMetadata(self.model_id, self.data)
        metadata_file = f"{self.model_id}_metadata.json"
        metadata_path = os.path.join(self.processed_data_path, metadata_file)
        metadata.save_to_json(metadata_path)