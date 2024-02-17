import os
import subprocess




def find_ptf_files_at_folder_depth(base_path, level=1):
    """
    Find .ptf files by walking through the directory tree up to a specified depth.

    Parameters:
    - base_path: The path to the base directory.
    - level: The depth of directories to walk through.

    Returns:
    - A list of paths to .ptf files at the specified level.
    """
    base_level = base_path.rstrip(os.path.sep).count(os.path.sep)
    ptf_files = []

    for root, dirs, files in os.walk(base_path):
        current_level = root.count(os.path.sep)
        if current_level - base_level < level:
            # Continue walking
            pass
        elif current_level - base_level == level:
            # At the right level, look for .ptf files and add to the list
            ptf_files.extend([os.path.join(root, f) for f in files if f.lower().endswith('.ptf')])
        else:
            # Beyond the desired level, stop walking this branch
            dirs[:] = []

    return ptf_files

def run_js_in_ptf(d3plot_executable, ptf_path, js_path, raw_data_dir=None):
    """
    : Run a JS script inside a PTF file: open a .ptf file and execute the specified JS script file.

    Parameters:
    - ptf_path: The path to the .ptf file to process.
    - js_path: The path to the JavaScript file to run with d3plot.
    - d3plot_executable: The path to the d3plot executable.
    - raw_data_dir: Optional path to a directory to place the RAW_DATA folder.
                    If not specified, defaults to the same directory as the ptf file.
    """
    if raw_data_dir is None:
        raw_data_dir = os.path.dirname(ptf_path)

    # Run the d3plot executable with the JavaScript script
    subprocess.run([d3plot_executable, ptf_path, "-js=" + js_path, "-exit"], check=True, cwd=os.path.dirname(ptf_path))

# Set Paths
d3plot_executable = r"C:\Users\thomas.bush\AppData\Roaming\Ove Arup\v21.0_x64\d3plot21_0_x64.exe"
base_directory = r"C:\Users\thomas.bush\Documents\temp\lsdyna-test\data"
js_path = r"C:\Users\thomas.bush\Documents\temp\lsdyna-test\D3Plot_extract_raw_debris.js"

# Specify the level of directories you want to search for .ptf files
walk_levels = 1  

# Find all .ptf files at the specified depth
ptf_files_at_level = find_ptf_files_at_folder_depth(base_directory, walk_levels)

# Process each .ptf file
for ptf_path in ptf_files_at_level:
    run_js_in_ptf(d3plot_executable, ptf_path, js_path)