import argparse
import os
import glob
from pathlib import Path
import subprocess # New import
import sys        # New import
from typing import Dict # Added for type hint

# Define the base directory for Hugging Face models
HF_MODELS_DIR = Path("huggingface/models")

def find_trainable_models(models_base_dir: Path) -> Dict[str, Path]:
    """
    Finds models that have a 'train.py' script in their directory.

    Args:
        models_base_dir: The base directory to search for models (e.g., huggingface/models).

    Returns:
        A dictionary mapping model name (directory name) to the Path of its train.py script.
    """
    trainable_models = {}
    if not models_base_dir.is_dir():
        print(f"Warning: Models base directory '{models_base_dir}' not found.")
        return trainable_models

    for model_dir in models_base_dir.iterdir():
        if model_dir.is_dir():
            train_script_path = model_dir / "train.py"
            if train_script_path.is_file():
                trainable_models[model_dir.name] = train_script_path
    return trainable_models

def main():
    parser = argparse.ArgumentParser(description="Master script to train existing models.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="all",
        help="Name of the model to train (e.g., 'emotion-detector'), or 'all' to train all models with a train.py script. 'list' to show available."
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Root directory for datasets. If provided, model-specific subdirectories might be used (e.g., dataset_root/imagenet_subset)."
    )
    # Future arguments could include: --epochs, --batch_size, etc.
    # These would need to be passed appropriately to the individual train.py scripts.

    args = parser.parse_args()

    print(f"Searching for trainable models in: {HF_MODELS_DIR.resolve()}")
    available_to_train = find_trainable_models(HF_MODELS_DIR)

    if not available_to_train and args.model_name != "list": # Allow 'list' even if no models found
        print("No models with 'train.py' scripts found.")
        return

    if args.model_name == "list":
        print("Available models with training scripts:")
        if available_to_train:
            for name in available_to_train.keys():
                print(f"- {name}")
        else:
            print("No models with 'train.py' scripts found in the specified directory.")
        return

    models_to_train = []
    if args.model_name == "all":
        models_to_train = list(available_to_train.keys())
        if not models_to_train:
             print("No models available to train with 'all' option.")
             return
        print(f"Attempting to train all available models: {', '.join(models_to_train)}")
    elif args.model_name in available_to_train:
        models_to_train = [args.model_name]
        print(f"Attempting to train selected model: {args.model_name}")
    else:
        print(f"Error: Model '{args.model_name}' not found or does not have a 'train.py' script.")
        print("Available models with training scripts:")
        if available_to_train:
            for name in available_to_train.keys():
                print(f"- {name}")
        else:
            print("No models with 'train.py' scripts found to list.")
        return

    if not models_to_train:
        print("No models selected for training.")
        return

    print("\n--- Initiating Training ---")
    for model_key in models_to_train:
        train_script_path = available_to_train[model_key]
        model_dir = train_script_path.parent # Directory of the train.py script

        print(f"Executing training script for '{model_key}': {train_script_path.resolve()}")

        command = [sys.executable, str(train_script_path.resolve())]

        # Example of how to pass model-specific arguments if needed in the future.
        # For now, we'll focus on the mobilenet dataset path.
        if model_key == "mobilenet" and args.dataset_root:
            # Assuming mobilenet/train.py can accept a --dataset_path argument
            # This is a hypothetical argument for mobilenet's train.py
            # We would need to modify mobilenet/train.py to accept it.
            # For now, this shows the pattern.
            # command.extend(["--dataset_path", str(Path(args.dataset_root) / "imagenet_subset_for_mobilenet")])
            # Since mobilenet/train.py has a hardcoded path, we can't easily override it yet
            # without modifying that script. We'll note this.
            print(f"  Note: mobilenet/train.py has a hardcoded dataset path. Parameter passing for it is not fully implemented yet in this master script.")

        elif model_key == "emotion-detector" and args.dataset_root:
            # emotion-detector/train.py uses datasets.load_dataset('emotion')
            # It could be modified to accept a path to a custom dataset.
            # command.extend(["--dataset_name_or_path", str(Path(args.dataset_root) / "emotion_dataset")])
            print(f"  Note: emotion-detector/train.py uses datasets.load_dataset('emotion'). Custom dataset path not yet implemented via this script.")

        elif model_key == "tinybert" and args.dataset_root:
            # tinybert/train.py uses datasets.load_dataset('imdb')
            # command.extend(["--dataset_name_or_path", str(Path(args.dataset_root) / "imdb_dataset")])
            print(f"  Note: tinybert/train.py uses datasets.load_dataset('imdb'). Custom dataset path not yet implemented via this script.")

        print(f"  Running command: {' '.join(command)}")
        print(f"  Working directory: {model_dir.resolve()}")

        try:
            # Execute the training script
            # Setting cwd ensures that the script's relative paths for saving models/logs work as intended.
            process = subprocess.run(command, cwd=model_dir.resolve(), check=True, capture_output=True, text=True)
            print(f"  --- Output for {model_key} ---")
            print(process.stdout)
            if process.stderr:
                print(f"  --- Errors (if any) for {model_key} ---")
                print(process.stderr)
            print(f"  Successfully trained {model_key}.")
        except subprocess.CalledProcessError as e:
            print(f"  Error training {model_key}:")
            print(f"  Return code: {e.returncode}")
            print(f"  --- STDOUT ---")
            print(e.stdout)
            print(f"  --- STDERR ---")
            print(e.stderr)
        except FileNotFoundError:
            print(f"  Error: Python executable or script not found. Ensure Python is in PATH and script exists.")
        except Exception as e:
            print(f"  An unexpected error occurred while training {model_key}: {e}")
        print("-" * 50)

    print("\n--- All Training Attempts Complete ---")

if __name__ == "__main__":
    main()
