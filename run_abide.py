import subprocess
import sys

# Path to the dataset configuration file
config_file = "configs/abide_schaefer100/TUs_graph_classification_ContrastPool_abide_schaefer100_100k.json"

# Arguments adapted from baseline.sh (--gpu_id -1 used for macOS CPU compatibility)
command = [
    sys.executable, "main.py",
    "--config", config_file,
    "--gpu_id", "-1",
    "--node_feat_transform", "pearson",
    "--max_time", "60",
    "--init_lr", "1e-2",
    "--threshold", "0.0",
    "--batch_size", "20",
    "--dropout", "0.0",
    "--contrast",
    "--pool_ratio", "0.5",
    "--lambda1", "1e-3",
    "--L", "2"
]

if __name__ == "__main__":
    print(f"Starting run for dataset configuration: {config_file}")
    
    # Execute main.py
    process = subprocess.run(command)
    
    if process.returncode == 0:
        print("\nTraining run for ABIDE finished successfully!")
    else:
        print(f"\nExecution failed with exit code: {process.returncode}")




        