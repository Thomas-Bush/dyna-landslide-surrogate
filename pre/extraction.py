import os
import subprocess

class RawDataExtractor:
    def __init__(self, base_directory, d3plot_executable, js_path, walk_levels=None):
        """
        Initialize the RawDataExtractor with the required paths and settings.

        Args:
            base_directory (str): The path to the base directory where .ptf files are located.
            d3plot_executable (str): The path to the d3plot executable.
            js_path (str): The path to the JavaScript file to run with d3plot.
            walk_levels (int, optional): The depth of directories to walk through to find .ptf files.
                                         If None, all levels will be walked.
        """
        self.base_directory = base_directory
        self.d3plot_executable = d3plot_executable
        self.js_path = js_path
        self.walk_levels = walk_levels

    def find_ptf_files(self):
        """
        Find .ptf files by walking through the directory tree up to the specified depth.

        Returns:
            A list of paths to .ptf files at the specified level or all levels if walk_levels is None.
        """
        base_level = self.base_directory.rstrip(os.path.sep).count(os.path.sep)
        ptf_files = []

        for root, dirs, files in os.walk(self.base_directory):
            current_level = root.count(os.path.sep)
            level_difference = current_level - base_level
            if self.walk_levels is not None and level_difference > self.walk_levels:
                dirs[:] = []  # Do not walk any further if we've reached the walk_levels depth
            else:
                ptf_files.extend([os.path.join(root, f) for f in files if f.lower().endswith('.ptf')])

        return ptf_files

    def run_js_in_ptf(self, ptf_path, raw_data_dir=None):
        """
        Run a JS script inside a PTF file: open a .ptf file and execute the specified JS script file.

        Args:
            ptf_path (str): The path to the .ptf file to process.
            raw_data_dir (str, optional): Path to a directory to place the RAW_DATA folder.
                                          If not specified, defaults to the same directory as the ptf file.
        """
        if raw_data_dir is None:
            raw_data_dir = os.path.dirname(ptf_path)

        # Run the d3plot executable with the JavaScript script
        subprocess.run([self.d3plot_executable, ptf_path, "-js=" + self.js_path, "-exit"],
                       check=True, cwd=raw_data_dir)

    def extract_all(self):
        """
        Find and process all .ptf files up to the specified depth or all levels if walk_levels is None.
        """
        ptf_files = self.find_ptf_files()

        for ptf_path in ptf_files:
            self.run_js_in_ptf(ptf_path)

