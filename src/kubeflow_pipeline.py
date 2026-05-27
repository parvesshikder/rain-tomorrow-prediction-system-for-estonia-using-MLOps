# src/kubeflow_pipeline.py

import subprocess
import sys
import yaml
from datetime import datetime

def run_step(step_name: str, command: list):
    """Run a pipeline step and report success or failure."""
    print(f"\n{'='*50}")
    print(f"STEP: {step_name}")
    print(f"{'='*50}")

    result = subprocess.run(command, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"FAILED: {step_name}")
        sys.exit(1)
    else:
        print(f"COMPLETED: {step_name}")

def run_pipeline():
    # Load params
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    print("\n" + "="*50)
    print(f"PIPELINE: {params['project']['title']}")
    print(f"STARTED : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)

    # Define pipeline steps
    steps = [
        (
            "Step 1: Fetch Data",
            [sys.executable, "-m", "src.fetch_data"]
        ),
        (
            "Step 2: Preprocess Data",
            [sys.executable, "-m", "src.preprocess"]
        ),
        (
            "Step 3: Train Model",
            [sys.executable, "-m", "src.train"]
        ),
        (
            "Step 4: Evaluate Model",
            [sys.executable, "-m", "src.evaluate"]
        ),
    ]

    # Run each step
    for step_name, command in steps:
        run_step(step_name, command)

    print("\n" + "="*50)
    print(f"PIPELINE COMPLETED SUCCESSFULLY")
    print(f"FINISHED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)
    print(f"\nModel saved to  : {params['artifacts']['model_path']}")
    print(f"Metrics saved to: {params['evaluation']['metrics_path']}")
    print(f"MLflow tracking : {params['tracking']['mlflow_tracking_uri']}")

if __name__ == "__main__":
    run_pipeline()
