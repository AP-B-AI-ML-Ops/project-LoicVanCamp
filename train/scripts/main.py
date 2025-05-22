import subprocess

print("=== Preprocessing data ===")
subprocess.run("python scripts/preprocess_data.py --raw_data_path data/StudentsPerformance.csv --dest_path models", shell=True, check=True)

print("=== Model training ===")
subprocess.run("python scripts/train.py --data_path models", shell=True, check=True)

print("=== Hyperparameter optimization ===")
subprocess.run("python scripts/hpo.py --data_dir models --num_trials 20", shell=True, check=True)

print("=== Register best model ===")
subprocess.run("python scripts/register_model.py --data_path models --top_n 5", shell=True, check=True)