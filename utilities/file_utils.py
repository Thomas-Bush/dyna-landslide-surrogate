import os
import shutil
import json
from pathlib import Path

class FileUtils:
    """Utility class for common file operations."""

    @staticmethod
    def rename_files_in_folder(folder_path, base_file_name):
        """
        Renames all files in the specified folder to have the same base file name while
        keeping their original extensions.

        Args:
            folder_path (str): The path to the folder containing the files to be renamed.
            base_file_name (str): The new base name to apply to all files.
        """
        for filename in os.listdir(folder_path):
            old_file_path = os.path.join(folder_path, filename)
            if os.path.isfile(old_file_path):
                file_extension = os.path.splitext(filename)[1]
                new_file_name = f"{base_file_name}{file_extension}"
                new_file_path = os.path.join(folder_path, new_file_name)
                os.rename(old_file_path, new_file_path)
                print(f"Renamed '{filename}' to '{new_file_name}'")

    @staticmethod
    def get_dir_from_filepath(file_path):
        """
        Extract the directory path from the given file path.

        Args:
            file_path (str): A string representing the file path.

        Returns:
            str: A string representing the directory path where the file is located.
        """
        return os.path.dirname(file_path)

    @staticmethod
    def add_subfolder_to_dir_path(dir_path, subfolder_name):
        """
        Append a subfolder to the given directory path.

        Args:
            dir_path (str): A string representing the directory path.
            subfolder_name (str): A string representing the name of the subfolder to add.

        Returns:
            str: A new directory path string with the subfolder appended.
        """
        return os.path.join(dir_path, subfolder_name)


    @staticmethod
    def build_file_mapping(root_directory, target_directory, file_extension):
        """
        Creates a mapping of folders containing specified file extension to a new directory structure.

        Args:
            root_directory (str): The root directory to search for files with the specified extension.
            target_directory (str): The directory where the folders will be moved to.
            file_extension (str): The file extension to search for within the root directory.

        Returns:
            dict: A dictionary mapping old folder paths to new folder paths.
        """
        catalogue = {}
        relevant_folders = []
        model_id = 1

        for root, dirs, files in os.walk(root_directory):
            for file in files:
                if file.endswith(file_extension) or file_extension in file:
                    relevant_folders.append(root)
                    break  # Found a file with the specified extension, no need to check other files in this folder

        # Ensure the target directory exists
        Path(target_directory).mkdir(parents=True, exist_ok=True)

        for folder in sorted(set(relevant_folders)):
            old_path = Path(folder)
            new_folder_name = f"{model_id:05d}"
            new_path = Path(target_directory) / new_folder_name
            catalogue[str(old_path)] = str(new_path)
            model_id += 1

        return catalogue
    
    @staticmethod
    def copy_and_rename_files(mapping, file_extension):
        """
        Copies files with the specified extension from source to target directories and renames them.

        Args:
            mapping (dict): A dictionary mapping source directory paths to target directory paths.
            file_extension (str): The file extension to search for and copy.
        """
        for old_path, new_path in mapping.items():
            new_base_name = os.path.basename(new_path)
            Path(new_path).mkdir(parents=True, exist_ok=True)  # Ensure the target directory exists

            # Iterate over all files in the old directory
            for file_name in os.listdir(old_path):
                if file_name.endswith(file_extension) or file_extension in file_name:
                    old_file_full_path = os.path.join(old_path, file_name)
                    new_file_name = f"{new_base_name}{os.path.splitext(file_name)[1]}"
                    new_file_full_path = os.path.join(new_path, new_file_name)
                    shutil.copy2(old_file_full_path, new_file_full_path)

    @staticmethod
    def json_file_to_dict(file_path):
        """
        Reads a JSON file and converts it to a dictionary.

        Args:
            file_path (str): The path to the JSON file.

        Returns:
            dict: The dictionary representation of the JSON file, or None if an error occurs.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as json_file:
                return json.load(json_file)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"An error occurred: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        return None
    
    @staticmethod
    def find_files_at_folder_depth(base_path, extension, level=1):
        """
        Find files with a specified extension by walking through the directory tree up to a specified depth.

        Args:
            base_path (str): The path to the base directory.
            extension (str): The file extension to search for.
            level (int): The depth of directories to walk through.

        Returns:
            List[str]: A list of paths to files with the specified extension at the specified level.
        """
        base_level = base_path.rstrip(os.path.sep).count(os.path.sep)
        target_files = []

        for root, dirs, files in os.walk(base_path):
            current_level = root.count(os.path.sep)
            if current_level - base_level < level:
                # Continue walking
                pass
            elif current_level - base_level == level:
                # At the right level, look for files with the specified extension and add to the list
                target_files.extend([os.path.join(root, f) for f in files if f.lower().endswith(extension)])
            else:
                # Beyond the desired level, stop walking this branch
                dirs[:] = []

        return target_files
    
    @staticmethod
    def get_subfolder_names(directory):
        """Get all subfolder names within a specified directory.

        Args:
            directory (str): The path to the directory from which to retrieve subfolder names.

        Returns:
            list: A list of subfolder names within the specified directory.
        """
        # List to hold the names of subfolders
        subfolders = []

        # Iterate over the entries in the given directory
        for entry in os.listdir(directory):
            # Construct full path
            full_path = os.path.join(directory, entry)
            # Check if this entry is a directory
            if os.path.isdir(full_path):
                subfolders.append(entry)

        return subfolders