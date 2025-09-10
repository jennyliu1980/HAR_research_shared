#!/usr/bin/env python3
"""
Script to download and prepare datasets for the HAR self-supervised learning paper.

Paper: "An Improved Masking Strategy for Self-supervised Masked Reconstruction
        in Human Activity Recognition"
GitHub: https://github.com/diheal/channelMasking

Usage:
    python prepare_datasets.py --dataset all
    python prepare_datasets.py --dataset ucihar --output_dir datasets/sub
    python prepare_datasets.py --dataset motion --output_dir datasets/sub
    python prepare_datasets.py --dataset uschad --output_dir datasets/sub
"""

import os
import zipfile
import urllib.request
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from scipy import stats
from scipy.io import loadmat
from sklearn.preprocessing import LabelEncoder
import glob


def download_ucihar(output_dir):
    """Download and prepare UCI-HAR dataset"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already extracted
    extract_dir = output_dir / "UCI HAR Dataset"
    if extract_dir.exists():
        print(f"✓ UCI-HAR dataset already exists at {extract_dir}")
        print("  Skipping download...")
        return extract_dir

    # UCI-HAR download URL
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
    zip_path = output_dir / "UCI_HAR_Dataset.zip"

    # Download if not exists
    if not zip_path.exists():
        print(f"Downloading UCI-HAR dataset...")
        print(f"URL: {url}")
        urllib.request.urlretrieve(url, zip_path)
        print(f"Downloaded to {zip_path}")
    else:
        print(f"✓ Zip file already exists at {zip_path}")

    # Extract
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    print(f"Extracted to {extract_dir}")

    print("\nUCI-HAR dataset structure:")
    print(f"  Train samples: 7352")
    print(f"  Test samples: 2947")
    print(f"  Total samples: 10299")
    print(f"  Timesteps: 128")
    print(f"  Channels: 9 (3-axis accelerometer + 3-axis gyroscope + 3-axis total acceleration)")
    print(f"  Classes: 6 (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING)")
    print(f"  Subjects: 30")

    return extract_dir


def prepare_motionsense(output_dir):
    """
    Download and process MotionSense dataset automatically.
    Downloads from GitHub and processes into required format.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("MotionSense Dataset Preparation")
    print("=" * 60)

    # Check if already processed
    if all((output_dir / f).exists() for f in
           ["Motion_X_1s.npy", "Motion_Y_1s.npy", "Motion_Subject_1s.npy"]):
        print("\n✓ MotionSense data already processed!")
        verify_motionsense_data(output_dir)
        return output_dir

    # Download MotionSense dataset
    urls = {
        'A_DeviceMotion_data': 'https://github.com/mmalekzadeh/motion-sense/raw/master/data/A_DeviceMotion_data.zip',
        'data_subjects_info': 'https://github.com/mmalekzadeh/motion-sense/raw/master/data/data_subjects_info.csv'
    }

    # Create raw data directory
    raw_dir = output_dir / "raw_motion"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Download device motion data
    zip_path = raw_dir / "A_DeviceMotion_data.zip"
    if not zip_path.exists():
        print("Downloading MotionSense device motion data...")
        urllib.request.urlretrieve(urls['A_DeviceMotion_data'], zip_path)
        print(f"Downloaded to {zip_path}")

    # Extract
    extract_dir = raw_dir / "A_DeviceMotion_data"
    if not extract_dir.exists():
        print("Extracting data...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(raw_dir)

    # Download subject info
    info_path = raw_dir / "data_subjects_info.csv"
    if not info_path.exists():
        print("Downloading subject info...")
        urllib.request.urlretrieve(urls['data_subjects_info'], info_path)

    # Process the data
    print("\nProcessing MotionSense data...")
    process_motionsense_data(extract_dir, info_path, output_dir)

    # Verify
    print("\nVerifying processed data...")
    verify_motionsense_data(output_dir)

    return output_dir


def process_motionsense_data(data_dir, info_path, output_dir):
    """Process MotionSense data into 1-second windows at 50Hz"""

    # Load subject info
    subject_info = pd.read_csv(info_path)

    # Activity labels (following paper's order)
    activities = ['dws', 'ups', 'sit', 'std', 'wlk', 'jog']
    activity_map = {act: i for i, act in enumerate(activities)}

    # Initialize lists for processed data
    all_segments = []
    all_labels = []
    all_subjects = []

    # data_dir is already the correct directory from prepare_motionsense
    data_dir = Path(data_dir)

    print(f"Looking for data in: {data_dir}")

    # List all directories (activity folders like dws_1, jog_9, etc.)
    folders = [f for f in data_dir.iterdir() if f.is_dir()]
    print(f"Found {len(folders)} folders")

    # Process each activity folder
    files_processed = 0
    subjects_with_data = set()

    for folder in folders:
        folder_name = folder.name

        # Skip __MACOSX folder
        if folder_name.startswith('_'):
            continue

        # Parse activity from folder name (e.g., "dws_1" -> "dws")
        parts = folder_name.split('_')
        if len(parts) < 2:
            continue

        activity = parts[0]

        if activity not in activities:
            continue

        # List CSV files in this folder (sub_1.csv through sub_24.csv)
        csv_files = list(folder.glob("*.csv"))

        if len(csv_files) > 0:
            print(f"Processing {folder_name}: {len(csv_files)} files")

        for csv_file in csv_files:
            # Parse subject ID from filename (e.g., "sub_1.csv" -> 1)
            filename = csv_file.stem  # filename without extension
            if not filename.startswith('sub_'):
                continue

            try:
                subject_id = int(filename.split('_')[1])
            except (IndexError, ValueError):
                continue

            try:
                # Load data
                df = pd.read_csv(csv_file)

                # Use gravity as accelerometer and rotationRate as gyroscope
                # Based on the actual columns in MotionSense data
                accel_cols = ['gravity.x', 'gravity.y', 'gravity.z']
                gyro_cols = ['rotationRate.x', 'rotationRate.y', 'rotationRate.z']

                # Verify columns exist
                if not all(col in df.columns for col in accel_cols + gyro_cols):
                    print(f"  Missing columns in {csv_file.name}")
                    print(f"    Expected: {accel_cols + gyro_cols}")
                    print(f"    Available: {list(df.columns)}")
                    continue

                # Extract sensor data
                sensor_data = df[accel_cols + gyro_cols].values

                # Remove NaN values
                sensor_data = sensor_data[~np.isnan(sensor_data).any(axis=1)]

                if len(sensor_data) < 50:  # Need at least 1 second of data
                    continue

                # Create 1-second windows (50 samples at 50Hz)
                window_size = 50
                stride = 25  # 50% overlap

                for i in range(0, len(sensor_data) - window_size + 1, stride):
                    segment = sensor_data[i:i + window_size]

                    if segment.shape[0] == window_size:
                        all_segments.append(segment)
                        all_labels.append(activity_map[activity])
                        all_subjects.append(subject_id)
                        subjects_with_data.add(subject_id)

                files_processed += 1
                if files_processed % 50 == 0:
                    print(f"  Processed {files_processed} files...")

            except Exception as e:
                print(f"  Error processing {csv_file.name}: {e}")
                continue

    print(f"\nProcessing complete:")
    print(f"  Files processed: {files_processed}")
    print(f"  Subjects with data: {len(subjects_with_data)}/24")
    print(f"  Total segments: {len(all_segments)}")

    # Check if we have any data
    if len(all_segments) == 0:
        print("\n✗ No data was successfully processed!")
        return None, None, None

    # Convert to numpy arrays
    X = np.array(all_segments, dtype=np.float32)  # Shape: (n_samples, 50, 6)
    y = np.array(all_labels, dtype=np.int32)
    subjects = np.array(all_subjects, dtype=np.int32)

    print(f"\nProcessed data shape: {X.shape}")
    print(f"Number of samples: {len(X)}")
    print(f"Activities distribution:")
    for act, idx in activity_map.items():
        count = np.sum(y == idx)
        print(f"  {act}: {count} samples")

    # Convert labels to one-hot encoding
    n_classes = len(activities)
    y_onehot = np.zeros((len(y), n_classes), dtype=np.float32)
    if len(y) > 0:
        y_onehot[np.arange(len(y)), y] = 1

    # Save processed data
    output_dir = Path(output_dir)
    np.save(output_dir / "Motion_X_1s.npy", X)
    np.save(output_dir / "Motion_Y_1s.npy", y_onehot)
    np.save(output_dir / "Motion_Subject_1s.npy", subjects.reshape(-1, 1))

    print(f"\nSaved processed data to {output_dir}")
    print(f"  Motion_X_1s.npy: {X.shape}")
    print(f"  Motion_Y_1s.npy: {y_onehot.shape}")
    print(f"  Motion_Subject_1s.npy: {subjects.reshape(-1, 1).shape}")

    return X, y_onehot, subjects


def verify_motionsense_data(output_dir):
    """Verify the processed MotionSense data"""

    output_dir = Path(output_dir)

    try:
        X = np.load(output_dir / "Motion_X_1s.npy")
        Y = np.load(output_dir / "Motion_Y_1s.npy")
        S = np.load(output_dir / "Motion_Subject_1s.npy")

        print(f"✓ Data loaded successfully")
        print(f"  X shape: {X.shape} (samples, timesteps, features)")
        print(f"  Y shape: {Y.shape} (samples, classes)")
        print(f"  Subjects shape: {S.shape}")

        # Check data properties
        print(f"\nData properties:")
        print(f"  Sampling rate: 50 Hz")
        print(f"  Window size: 1 second (50 timesteps)")
        print(f"  Features: 6 (3-axis accel + 3-axis gyro)")
        print(f"  Classes: {Y.shape[1]}")
        print(f"  Subjects: {len(np.unique(S))}")

        return True

    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return False


def prepare_uschad(output_dir):
    """
    Process USC-HAD dataset from raw .mat files.

    Note: USC-HAD dataset needs to be manually downloaded from:
    http://sipi.usc.edu/had/
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("USC-HAD Dataset Preparation")
    print("=" * 60)

    # Check if already processed
    if all((output_dir / f).exists() for f in
           ["USCHAD_X.npy", "USCHAD_Y.npy", "USCHAD_Subject.npy"]):
        print("\n✓ USC-HAD data already processed!")
        verify_uschad_data(output_dir)
        return output_dir

    # Check for raw data
    raw_dir = output_dir / "USC-HAD"
    if not raw_dir.exists():
        print("\n✗ USC-HAD raw data not found!")
        print("\nPlease follow these steps:")
        print("1. Download USC-HAD from: http://sipi.usc.edu/had/")
        print("2. Extract the downloaded file to:", raw_dir)
        print("3. The folder should contain Subject1.mat through Subject14.mat")
        print("4. Run this script again")
        return None

    # Process the data
    print("\nProcessing USC-HAD data...")
    process_uschad_data(raw_dir, output_dir)

    # Verify
    print("\nVerifying processed data...")
    verify_uschad_data(output_dir)

    return output_dir


def process_uschad_data(raw_dir, output_dir):
    """Process USC-HAD .mat files into required format"""

    # USC-HAD has 12 activities
    activities = [
        'WalkingForward', 'WalkingLeft', 'WalkingRight', 'WalkingUpstairs',
        'WalkingDownstairs', 'Running', 'Jumping', 'Sitting', 'Standing',
        'Sleeping', 'ElevatorUp', 'ElevatorDown'
    ]

    # Initialize lists for all data
    all_segments = []
    all_labels = []
    all_subjects = []

    # Process each subject (1-14)
    for subject_id in range(1, 15):
        # Check for subject folder (not file)
        subject_folder = raw_dir / f"Subject{subject_id}"

        if not subject_folder.exists():
            print(f"Warning: {subject_folder} not found, skipping...")
            continue

        print(f"Processing subject {subject_id}/14...")

        # List all .mat files in the subject folder
        mat_files = list(subject_folder.glob("*.mat"))

        if not mat_files:
            print(f"  No .mat files found in {subject_folder}")
            continue

        print(f"  Found {len(mat_files)} trial files")

        # Process each trial file
        for mat_file in mat_files:
            try:
                # Parse activity from filename (e.g., "a1t1.mat" -> activity 1)
                filename = mat_file.stem  # filename without extension

                # More robust parsing
                if not filename.startswith('a'):
                    continue

                # Find where 't' appears
                t_index = filename.find('t')
                if t_index == -1:
                    continue

                # Extract activity number (everything between 'a' and 't')
                try:
                    activity_str = filename[1:t_index]
                    activity_num = int(activity_str)
                except ValueError:
                    print(f"  Could not parse activity from {filename}")
                    continue

                # Skip if activity number is out of range
                if activity_num < 1 or activity_num > 12:
                    print(f"  Skipping {filename}: activity {activity_num} out of range")
                    continue

                # Load .mat file
                mat_data = loadmat(mat_file)

                # Find the sensor data
                # USC-HAD typically stores data with the filename as key
                sensor_data = None

                # Try to find data using filename as key first
                if filename in mat_data:
                    sensor_data = mat_data[filename]
                else:
                    # Otherwise look for any array that looks like sensor data
                    for key in mat_data.keys():
                        if not key.startswith('__'):  # Skip metadata keys
                            data = mat_data[key]
                            # Check if it looks like sensor data (2D array with 6 columns)
                            if isinstance(data, np.ndarray) and len(data.shape) >= 2:
                                sensor_data = data
                                break

                if sensor_data is None:
                    print(f"  No sensor data found in {mat_file.name}")
                    continue

                # Ensure data is 2D array
                if len(sensor_data.shape) == 3:
                    # If 3D, take the first 2D slice or reshape
                    if sensor_data.shape[0] == 1:
                        sensor_data = sensor_data[0, :, :]
                    elif sensor_data.shape[2] == 1:
                        sensor_data = sensor_data[:, :, 0]
                    else:
                        sensor_data = sensor_data.reshape(-1, sensor_data.shape[-1])
                elif len(sensor_data.shape) == 1:
                    print(f"  Data is 1D in {mat_file.name}, skipping")
                    continue

                # USC-HAD has 6 channels (3 acc + 3 gyro)
                # Check if we need to transpose
                if sensor_data.shape[0] == 6 and sensor_data.shape[1] > 100:
                    sensor_data = sensor_data.T
                elif sensor_data.shape[1] != 6:
                    print(f"  Warning: Expected 6 channels, got shape {sensor_data.shape} in {mat_file.name}")
                    continue

                # Create 1-second windows at 100Hz
                window_size = 100  # 100Hz * 1 second
                stride = 50  # 50% overlap

                num_windows = 0
                for i in range(0, len(sensor_data) - window_size + 1, stride):
                    segment = sensor_data[i:i + window_size]

                    if segment.shape[0] == window_size:
                        # Activity numbers in USC-HAD are 1-based, convert to 0-based
                        label = activity_num - 1

                        # Only keep valid activities (0-11)
                        if 0 <= label < 12:
                            all_segments.append(segment)
                            all_labels.append(label)
                            all_subjects.append(subject_id)
                            num_windows += 1

                if num_windows > 0:
                    print(f"    {mat_file.name}: {num_windows} windows")

            except Exception as e:
                print(f"  Error processing {mat_file.name}: {e}")
                import traceback
                if subject_id == 1 and 'a1t1' in mat_file.name:  # Debug first file
                    traceback.print_exc()
                continue

    # Convert to numpy arrays
    if len(all_segments) == 0:
        print("\nNo data was successfully processed!")
        print("Debugging: Check if .mat files have the expected structure")
        return

    X = np.array(all_segments, dtype=np.float32)  # Shape: (n_samples, 100, 6)
    y = np.array(all_labels, dtype=np.int32)
    subjects = np.array(all_subjects, dtype=np.int32)

    print(f"\nProcessed data shape: {X.shape}")
    print(f"Number of samples: {len(X)}")
    print(f"Activities distribution:")
    for i in range(12):
        count = np.sum(y == i)
        if i < len(activities):
            print(f"  {activities[i]}: {count} samples")
        else:
            print(f"  Activity {i}: {count} samples")

    # Convert labels to one-hot encoding
    n_classes = 12
    y_onehot = np.zeros((len(y), n_classes), dtype=np.float32)
    y_onehot[np.arange(len(y)), y] = 1

    # Save processed data
    output_dir = Path(output_dir)
    np.save(output_dir / "USCHAD_X.npy", X)
    np.save(output_dir / "USCHAD_Y.npy", y_onehot)
    np.save(output_dir / "USCHAD_Subject.npy", subjects.reshape(-1, 1))

    print(f"\nSaved processed data to {output_dir}")
    print(f"  USCHAD_X.npy: {X.shape}")
    print(f"  USCHAD_Y.npy: {y_onehot.shape}")
    print(f"  USCHAD_Subject.npy: {subjects.reshape(-1, 1).shape}")
    print("\nNote: dataset.py will split subjects 1-10 for train, 11-14 for test")


def verify_uschad_data(output_dir):
    """Verify the processed USC-HAD data"""

    output_dir = Path(output_dir)

    try:
        X = np.load(output_dir / "USCHAD_X.npy")
        Y = np.load(output_dir / "USCHAD_Y.npy")
        S = np.load(output_dir / "USCHAD_Subject.npy")

        print(f"✓ Data loaded successfully")
        print(f"  X shape: {X.shape} (samples, timesteps, features)")
        print(f"  Y shape: {Y.shape} (samples, classes)")
        print(f"  Subjects shape: {S.shape}")

        # Check data properties
        print(f"\nData properties:")
        print(f"  Sampling rate: 100 Hz")
        print(f"  Window size: 1 second (100 timesteps)")
        print(f"  Features: 6 (3-axis accel + 3-axis gyro)")
        print(f"  Classes: {Y.shape[1]}")
        print(f"  Subjects: {len(np.unique(S))}")

        # Check subject split
        train_subjects = S[S.flatten() <= 10]
        test_subjects = S[S.flatten() > 10]
        print(f"\nDataset split (as per paper):")
        print(f"  Train subjects (1-10): {len(train_subjects)} samples")
        print(f"  Test subjects (11-14): {len(test_subjects)} samples")

        return True

    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return False


def verify_dataset(dataset_dir, dataset_name):
    """Verify that dataset files exist and have correct shapes"""

    dataset_dir = Path(dataset_dir)

    if dataset_name == 'ucihar':
        # Check for raw UCI-HAR directory
        uci_dir = dataset_dir / "UCI HAR Dataset"
        if uci_dir.exists():
            print(f"\n✓ UCI-HAR raw data found at {uci_dir}")

            # Check key files
            train_dir = uci_dir / "train" / "Inertial Signals"
            test_dir = uci_dir / "test" / "Inertial Signals"

            if train_dir.exists() and test_dir.exists():
                print("✓ Train and test Inertial Signals directories found")

                # Check for a sample signal file
                sample_file = train_dir / "body_acc_x_train.txt"
                if sample_file.exists():
                    data = np.loadtxt(sample_file)
                    print(f"✓ Sample signal shape: {data.shape} (should be (7352, 128))")

                return True
            else:
                print("✗ Missing Inertial Signals directories")
                return False
        else:
            print(f"✗ UCI-HAR dataset not found at {uci_dir}")
            print("  Please run: python prepare_datasets.py --dataset ucihar")
            return False

    elif dataset_name == 'motion':
        return verify_motionsense_data(dataset_dir)

    elif dataset_name == 'uschad':
        return verify_uschad_data(dataset_dir)

    return False


def main():
    parser = argparse.ArgumentParser(
        description='Prepare datasets for HAR self-supervised learning\n' +
                    'Based on: https://github.com/diheal/channelMasking',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['ucihar', 'motion', 'uschad', 'all'],
        default='all',
        help='Which dataset to prepare'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='datasets/sub',
        help='Output directory for datasets'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Only verify existing datasets'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("HAR Dataset Preparation Script")
    print("Based on: An Improved Masking Strategy for Self-supervised")
    print("          Masked Reconstruction in Human Activity Recognition")
    print("GitHub: https://github.com/diheal/channelMasking")
    print("=" * 60)

    if args.verify:
        # Verify mode
        print("\nVerifying datasets...")

        datasets = ['ucihar', 'motion', 'uschad'] if args.dataset == 'all' else [args.dataset]

        for dataset in datasets:
            print(f"\n--- {dataset.upper()} ---")
            verify_dataset(args.output_dir, dataset)

    else:
        # Preparation mode
        if args.dataset in ['ucihar', 'all']:
            print("\n--- UCI-HAR Dataset ---")
            download_ucihar(args.output_dir)

        if args.dataset in ['motion', 'all']:
            print("\n--- MotionSense Dataset ---")
            prepare_motionsense(args.output_dir)

        if args.dataset in ['uschad', 'all']:
            print("\n--- USC-HAD Dataset ---")
            prepare_uschad(args.output_dir)

        print("\n" + "=" * 60)
        print("Dataset preparation complete!")
        print("\nNext steps:")
        print("1. Verify datasets: python prepare_datasets.py --verify")
        print("2. Run pretraining: python main_wandb.py --dataset [ucihar/motion/uschad]")
        print("3. Run evaluation: python evaluate_wandb.py --dataset [ucihar/motion/uschad]")
        print("4. Run experiments:")
        print("   - UCI-HAR: python run_experiments.py")
        print("   - MotionSense: python run_experiments_motionsense.py")
        print("   - USC-HAD: python run_experiments_uschad.py")
        print("=" * 60)


if __name__ == '__main__':
    main()