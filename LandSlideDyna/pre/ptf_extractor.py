import os
import subprocess

class PTFDataExtractor:
    def __init__(self, d3plot_executable):
        """
        Initialize the PTFDataExtractor with the required path to d3plot executable.
        Esssentially a batch script to execute the extractor in multiple files.

        Args:
            d3plot_executable (str): The path to the d3plot executable.
        """
        self.d3plot_executable = d3plot_executable
        self.js_path = "d3plot_data_extractor.js"


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

        abs_js_path = os.path.abspath(self.js_path)

        subprocess.run([self.d3plot_executable, ptf_path, "-js=" + abs_js_path, "-exit"],
                       check=True, cwd=raw_data_dir)


    def run(self, ptf_path):
        """
        Process the provided .ptf file.

        Args:
            ptf_path (str): The path to the .ptf file to process.
        """
        self.run_js_in_ptf(ptf_path)




