import os
import subprocess

class D3PlotUtils:
    """Class to handle interactions with D3Plot."""

    @staticmethod
    def run_js_in_ptf(d3plot_executable, ptf_path, js_path, raw_data_dir=None):
        """
        Run a JS script inside a PTF file: open a .ptf file and execute the specified JS script file.

        Args:
            d3plot_executable (str): The path to the d3plot executable.
            ptf_path (str): The path to the .ptf file to process.
            js_path (str): The path to the JavaScript file to run with d3plot.
            raw_data_dir (str, optional): Path to a directory to place the RAW_DATA folder.
                                          If not specified, defaults to the same directory as the ptf file.
        """
        if raw_data_dir is None:
            raw_data_dir = os.path.dirname(ptf_path)

        # Run the d3plot executable with the JavaScript script
        subprocess.run([d3plot_executable, ptf_path, "-js=" + js_path, "-exit"], check=True, cwd=raw_data_dir)