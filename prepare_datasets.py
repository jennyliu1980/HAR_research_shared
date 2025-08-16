#!/usr/bin/env python3
"""
Script to download and prepare datasets for the HAR self-supervised learning paper.

Paper: "An Improved Masking Strategy for Self-supervised Masked Reconstruction
        in Human Activity Recognition"

Usage:
    python prepare_datasets.py --dataset ucihar --output_dir datasets/sub
"""

import os
import zipfile
import urllib.request
import numpy as np
from pathlib import Path
import argparse


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
    Prepare MotionSense dataset instructions.

    Note: MotionSense dataset needs to be manually downloaded from:
    https://github.com/mmalekzadeh/motion-sense
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("MotionSense Dataset Preparation")
    print("=" * 60)
    print("\nMotionSense dataset needs to be manually downloaded and processed.")
    print("\nSteps:")
    print("1. Download the dataset from: https://github.com/mmalekzadeh/motion-sense")
    print("2. Process the raw data into 1-second windows at 50Hz sampling rate")
    print("3. Save as the following .npy files in", output_dir)
    print("   - Motion_X_1s.npy: shape (n_samples, 50, 6)")
    print("   - Motion_Y_1s.npy: shape (n_samples, n_classes) one-hot encoded")
    print("   - Motion_Subject_1s.npy: shape (n_samples, 1)")
    print("\nExpected data characteristics:")
    print("  Sampling rate: 50Hz")
    print("  Window size: 1 second (50 timesteps)")
    print("  Channels: 6 (3-axis accelerometer + 3-axis gyroscope)")
    print("  Classes: 6 activities")
    print("  Subjects: 24")


def prepare_uschad(output_dir):
    """
    Prepare USC-HAD dataset instructions.

    Note: USC-HAD dataset needs to be manually downloaded from:
    http://sipi.usc.edu/had/
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("USC-HAD Dataset Preparation")
    print("=" * 60)
    print("\nUSC-HAD dataset needs to be manually downloaded and processed.")
    print("\nSteps:")
    print("1. Download the dataset from: http://sipi.usc.edu/had/")
    print("2. Process the raw .mat files into 1-second windows")
    print("3. Split into train/val/test sets")
    print("4. Save as the following .npy files in", output_dir)
    print("   - USCHAD_X_Train_1s.npy")
    print("   - USCHAD_Y_Train_1s.npy (one-hot encoded)")
    print("   - USCHAD_X_Val_1s.npy")
    print("   - USCHAD_Y_Val_1s.npy")
    print("   - USCHAD_X_Test_1s.npy")
    print("   - USCHAD_Y_Test_1s.npy")
    print("\nExpected data characteristics:")
    print("  Sampling rate: 100Hz")
    print("  Window size: 1 second")
    print("  Channels: 6 (3-axis accelerometer + 3-axis gyroscope)")
    print("  Classes: 12 activities")
    print("  Subjects: 14")


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
        required_files = [
            'Motion_X_1s.npy',
            'Motion_Y_1s.npy',
            'Motion_Subject_1s.npy'
        ]

        all_exist = True
        for file in required_files:
            path = dataset_dir / file
            if path.exists():
                data = np.load(path)
                print(f"✓ {file}: shape {data.shape}")
            else:
                print(f"✗ {file}: not found")
                all_exist = False

        return all_exist

    elif dataset_name == 'uschad':
        required_files = [
            'USCHAD_X_Train_1s.npy',
            'USCHAD_Y_Train_1s.npy',
            'USCHAD_X_Val_1s.npy',
            'USCHAD_Y_Val_1s.npy',
            'USCHAD_X_Test_1s.npy',
            'USCHAD_Y_Test_1s.npy'
        ]

        all_exist = True
        for file in required_files:
            path = dataset_dir / file
            if path.exists():
                data = np.load(path)
                print(f"✓ {file}: shape {data.shape}")
            else:
                print(f"✗ {file}: not found")
                all_exist = False

        return all_exist

    return False


def main():
    parser = argparse.ArgumentParser(
        description='Prepare datasets for HAR self-supervised learning'
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
            prepare_motionsense(args.output_dir)

        if args.dataset in ['uschad', 'all']:
            prepare_uschad(args.output_dir)

        print("\n" + "=" * 60)
        print("Dataset preparation complete!")
        print("\nNext steps:")
        print("1. Verify datasets: python prepare_datasets.py --verify")
        print("2. Run pretraining: python main.py --dataset ucihar")
        print("3. Run evaluation: python evaluate.py --dataset ucihar")
        print("=" * 60)


if __name__ == '__main__':
    main()