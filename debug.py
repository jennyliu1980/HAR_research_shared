# Test if the updated script will work
from pathlib import Path
import numpy as np
import pandas as pd

data_dir = Path(
    r"C:\Users\nikit\PycharmProjects\jenny_HAR\HAR_research_shared\datasets\sub\raw_motion\A_DeviceMotion_data")

# Activity labels
activities = ['dws', 'ups', 'sit', 'std', 'wlk', 'jog']
activity_map = {act: i for i, act in enumerate(activities)}

# Process one folder as a test
test_folder = data_dir / "jog_9"
if test_folder.exists():
    csv_files = list(test_folder.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files in jog_9")

    # Process first file
    if csv_files:
        df = pd.read_csv(csv_files[0])

        # Use the correct columns
        accel_cols = ['gravity.x', 'gravity.y', 'gravity.z']
        gyro_cols = ['rotationRate.x', 'rotationRate.y', 'rotationRate.z']

        # Extract data
        sensor_data = df[accel_cols + gyro_cols].values
        sensor_data = sensor_data[~np.isnan(sensor_data).any(axis=1)]

        print(f"Sensor data shape: {sensor_data.shape}")

        # Create windows
        window_size = 50
        stride = 25
        segments = []

        for i in range(0, len(sensor_data) - window_size + 1, stride):
            segment = sensor_data[i:i + window_size]
            if segment.shape[0] == window_size:
                segments.append(segment)

        print(f"Created {len(segments)} windows from first file")
        print(f"Each window shape: {segments[0].shape if segments else 'None'}")