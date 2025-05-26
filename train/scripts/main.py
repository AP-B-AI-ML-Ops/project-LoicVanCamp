"""Main pipeline script for running data preprocessing, training, HPO, and model registration.

This script sequentially runs all steps of the ML pipeline using subprocess calls:
- Data preprocessing
- Model training
- Hyperparameter optimization
- Registering the best model
"""

import subprocess


def run_pipeline():
    """Run the full ML pipeline by executing each step as a subprocess."""
    print("=== Preprocessing data ===")
    subprocess.run(
        [
            "python",
            "scripts/preprocess_data.py",
            "--raw_data_path",
            "data/StudentsPerformance.csv",
        ],
        check=True,
    )

    print("=== Model training ===")
    subprocess.run(
        ["python", "scripts/train.py"],
        check=True,
    )

    print("=== Hyperparameter optimization ===")
    subprocess.run(
        [
            "python",
            "scripts/hpo.py",
            "--num_trials",
            "20",
        ],
        check=True,
    )

    print("=== Register best model ===")
    subprocess.run(
        [
            "python",
            "scripts/register_model.py",
            "--top_n",
            "5",
        ],
        check=True,
    )


if __name__ == "__main__":
    run_pipeline()
