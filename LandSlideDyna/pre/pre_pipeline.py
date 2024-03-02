import os
import json

from numpy import extract

from ptf_extractor import PTFDataExtractor
from ptf_extractor import PTFImageExtractor
from intial_processing import InitialProcessPipeline
from final_processing import FinalProcessPipeline
from metadata import ModelMetadata

class PreprocessingPipeline:
    def __init__(self, model_dir, d3plot_executable, status_file=None, ):
        """Initialize the preprocessing pipeline with the given model directory and optional status file.

        Args:
            model_dir (str): The directory containing the models to be processed.
            status_file (str, optional): The path to the JSON file that stores the processing status.
                                         Defaults to None and will be set to 'preprocess_status.json' in the model_dir.
            d3plot_executable (str): The path to the d3plot executable.
        """
        self.model_dir = model_dir
        self.status_file = status_file if status_file else os.path.join(model_dir, 'preprocess_status.json')
        self.status_dict = self.load_status()
        self.d3plot_exe = d3plot_executable

    def load_status(self):
        """Load the preprocessing status from a JSON file."""
        if os.path.exists(self.status_file):
            with open(self.status_file, 'r') as file:
                status_dict = json.load(file)
        else:
            status_dict = {}
        return status_dict

    def save_status(self):
        """Save the preprocessing status to a JSON file."""
        with open(self.status_file, 'w') as file:
            json.dump(self.status_dict, file, indent=4)

    def check_and_update_status(self):
        """Check the preprocessing status of each model in the data directory and update the status file."""
        status = {}
        for model in os.listdir(self.model_dir):
            model_path = os.path.join(self.model_dir, model)
            if os.path.isdir(model_path):
                d3plot_ptf_path = os.path.join(model_path, '01_d3plot/ptf/')
                d3plot_image_path = os.path.join(model_path, '01_d3plot/image/')
                extract_path = os.path.join(model_path, '02_extract/')
                initial_process_path = os.path.join(model_path, '03_initial_process/')
                final_process_path = os.path.join(model_path, '04_final_process/')

                status[model] = {
                    "ignore": os.path.isfile(os.path.join(model_path, 'ignore.txt')),
                    "metadata": os.path.isfile(os.path.join(model_path, f"{model}_metadata.json")),
                    "d3plot_ptf": os.path.isdir(d3plot_ptf_path) and any(os.path.isfile(os.path.join(d3plot_ptf_path, file)) for file in os.listdir(d3plot_ptf_path) if file.endswith('.ptf')),
                    "d3plot_image": os.path.isdir(d3plot_image_path) and bool(os.listdir(d3plot_image_path)),
                    "extract": os.path.isdir(extract_path) and all(os.path.isfile(os.path.join(extract_path, filename)) for filename in [
                        f"{model}_nodal_velocities.csv",
                        f"{model}_nodes.csv",
                        f"{model}_shells.csv",
                        f"{model}_solid_thicknesses.csv",
                        f"{model}_solids.csv",
                        f"{model}_states.csv"
                    ]),
                    "initial_process": os.path.isdir(initial_process_path) and all(
                        os.path.isdir(os.path.join(initial_process_path, dirname)) and bool(
                            os.listdir(os.path.join(initial_process_path, dirname))) for dirname in ['elevation', 'thickness', 'velocity']),
                    "final_process": os.path.isdir(final_process_path) and all(
                        os.path.isdir(os.path.join(final_process_path, dirname)) and bool(
                            os.listdir(os.path.join(final_process_path, dirname))) for dirname in ['elevation', 'thickness', 'velocity'])
                }

        self.status_dict = status
        self.save_status()

    def save_status(self):
        """Save the preprocessing status to a JSON file."""
        with open(self.status_file, 'w') as file:
            json.dump(self.status_dict, file, indent=4)

    def process_models(self):
        """Process each model based on its preprocessing status."""
        for model, status in self.status_dict.items():
            model_path = os.path.join(self.model_dir, model)
            model_ptf_path = os.path.join(model_path, "01_d3plot", "ptf", f"{model}.ptf")
            d3plotexe = self.d3plot_exe

            if status.get("ignore", False):
                print(f"Skipping model {model} as it is marked to be ignored.")
                continue

            # Check if all conditions are True
            if all(status.get(key, False) for key in ["extract", "d3plot_image", "initial_process", "final_process"]):
                print(f"Model {model} already fully processed.")
                continue

            if not status.get("extract", False):
                print(f"Running extraction pipeline for model {model}.")
                extraction_pipeline = PTFDataExtractor(d3plotexe)
                extraction_pipeline.run(model_ptf_path)


            if not status.get("d3plot_image", False):
                pass


            if not status.get("initial_process", False):
                print(f"Running initial processing pipeline for model {model}.")
                initial_process_path = os.path.join(model_path, '03_initial_process/')
                extract_path = os.path.join(model_path, '02_extract/')
                initial_processing_pipeline = InitialProcessPipeline(model, extract_path, initial_process_path)
                initial_processing_pipeline.run()

            if not status.get("metadata", False):
                
                for model in os.listdir(self.model_dir):
                    print(f"Geting metadata for model {model}.")
                    meta = ModelMetadata(self.model_dir, model)
                    meta.save_metadata()
                    print(f"Finished metadata for model {model}.")

            if not status.get("final_process", False):
                pass

    def run(self):
        """Run the preprocessing pipeline."""
        self.check_and_update_status()
        self.process_models()
        self.check_and_update_status()
        print("Preprocessing pipeline completed.")