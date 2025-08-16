import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_data(dir, data_name, transformer=None, divide_seed=None):
    """
    Load and preprocess HAR datasets according to the paper's specifications.

    Paper: "An Improved Masking Strategy for Self-supervised Masked Reconstruction
           in Human Activity Recognition"

    Args:
        dir: Dataset directory path
        data_name: Dataset name ('ucihar', 'motion', 'uschad')
        transformer: Whether to apply standardization
        divide_seed: Random seed for train/val/test split

    Returns:
        x_train, y_train, x_val, y_val, x_test, y_test
    """

    if data_name not in ['uschad', 'ucihar', 'motion']:
        raise ValueError(f"Dataset {data_name} is not supported")

    # ==================== UCI-HAR Dataset ====================
    if data_name == 'ucihar':
        # Check if preprocessed .npy files exist
        x_path = os.path.join(dir, 'UCI_X.npy')
        y_path = os.path.join(dir, 'UCI_Y.npy')
        s_path = os.path.join(dir, 'UCI_Subject.npy')

        if not all(os.path.exists(p) for p in [x_path, y_path, s_path]):
            print("Generating UCI-HAR .npy files from raw data...")

            # Path to UCI HAR Dataset folder
            uci_dir = os.path.join(dir, 'UCI HAR Dataset')
            if not os.path.exists(uci_dir):
                raise FileNotFoundError(f"Please download and extract UCI HAR Dataset to {uci_dir}")

            # Load raw inertial signals (9 channels as per paper)
            signal_names = [
                'body_acc_x', 'body_acc_y', 'body_acc_z',
                'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
                'total_acc_x', 'total_acc_y', 'total_acc_z'
            ]

            def load_signals(subset):
                """Load inertial signals for train or test subset"""
                signals_path = os.path.join(uci_dir, subset, 'Inertial Signals')
                signals = []

                for signal_name in signal_names:
                    file_path = os.path.join(signals_path, f'{signal_name}_{subset}.txt')
                    signal = np.loadtxt(file_path)  # Shape: (n_samples, 128)
                    signals.append(signal)

                # Stack to shape (n_samples, 128, 9)
                return np.stack(signals, axis=-1)

            # Load train and test data
            X_train = load_signals('train')  # (7352, 128, 9)
            X_test = load_signals('test')  # (2947, 128, 9)

            # Load labels (1-6 for 6 activities)
            y_train = np.loadtxt(os.path.join(uci_dir, 'train', 'y_train.txt'), dtype=int)
            y_test = np.loadtxt(os.path.join(uci_dir, 'test', 'y_test.txt'), dtype=int)

            # Convert to one-hot encoding
            def to_one_hot(y, n_classes=6):
                n_samples = len(y)
                y_one_hot = np.zeros((n_samples, n_classes))
                y_one_hot[np.arange(n_samples), y - 1] = 1  # -1 because labels are 1-6
                return y_one_hot

            y_train = to_one_hot(y_train)
            y_test = to_one_hot(y_test)

            # Load subject IDs (1-30)
            subject_train = np.loadtxt(os.path.join(uci_dir, 'train', 'subject_train.txt'), dtype=int)
            subject_test = np.loadtxt(os.path.join(uci_dir, 'test', 'subject_test.txt'), dtype=int)

            # Combine all data
            x_data = np.concatenate([X_train, X_test], axis=0)  # (10299, 128, 9)
            y_data = np.concatenate([y_train, y_test], axis=0)  # (10299, 6)
            subject_index = np.concatenate([subject_train, subject_test], axis=0).reshape(-1, 1)

            # Save preprocessed data
            np.save(x_path, x_data)
            np.save(y_path, y_data)
            np.save(s_path, subject_index)

            print(f"Saved UCI-HAR data:")
            print(f"  X shape: {x_data.shape} (samples, timesteps, channels)")
            print(f"  Y shape: {y_data.shape} (samples, classes)")
            print(f"  Subjects: {len(np.unique(subject_index))} unique subjects")

        # Load preprocessed data
        x_data = np.load(x_path)
        y_data = np.load(y_path)
        subject_index = np.load(s_path)

        # Shuffle data with fixed seed for reproducibility
        np.random.seed(888)
        indices = np.random.permutation(len(x_data))
        x_data = x_data[indices]
        y_data = y_data[indices]
        subject_index = subject_index[indices]

        # Subject-wise split according to paper (60% train, 20% val, 20% test)
        # Paper mentions 5 different splits for cross-validation
        divide = divide_seed

        if divide == 1:
            test_subject = set([1, 2, 3, 4, 5, 6])  # 20%
            val_subject = set([7, 8, 9, 10, 11, 12])  # 20%
            train_subject = set(range(13, 31))  # 60%
        elif divide == 2:
            test_subject = set([25, 26, 27, 28, 29, 30])
            val_subject = set(range(19, 25))
            train_subject = set(range(1, 19))
        elif divide == 3:
            test_subject = set(range(7, 13))
            val_subject = set(range(25, 31))
            train_subject = set(list(range(1, 7)) + list(range(13, 25)))
        elif divide == 4:
            test_subject = set(range(13, 19))
            val_subject = set(range(19, 25))
            train_subject = set(list(range(1, 13)) + list(range(25, 31)))
        elif divide == 5:
            test_subject = set(range(19, 25))
            val_subject = set(range(1, 7))
            train_subject = set(list(range(7, 19)) + list(range(25, 31)))
        else:
            # Random split with seed
            subjects = list(range(1, 31))
            np.random.seed(divide if divide else 42)
            np.random.shuffle(subjects)

            n_test = 6  # 20% of 30
            n_val = 6  # 20% of 30

            test_subject = set(subjects[:n_test])
            val_subject = set(subjects[n_test:n_test + n_val])
            train_subject = set(subjects[n_test + n_val:])

        # Create masks for splitting
        train_mask = np.array([s[0] in train_subject for s in subject_index])
        val_mask = np.array([s[0] in val_subject for s in subject_index])
        test_mask = np.array([s[0] in test_subject for s in subject_index])

        x_train = x_data[train_mask]
        y_train = y_data[train_mask]
        x_val = x_data[val_mask]
        y_val = y_data[val_mask]
        x_test = x_data[test_mask]
        y_test = y_data[test_mask]

        # Apply standardization if requested
        if transformer:
            n_samples, n_timesteps, n_features = x_train.shape

            # Fit scaler on training data
            scaler = StandardScaler()
            x_train_flat = x_train.reshape(-1, n_features)
            x_train_flat = scaler.fit_transform(x_train_flat)
            x_train = x_train_flat.reshape(n_samples, n_timesteps, n_features)

            # Transform validation data
            n_val = x_val.shape[0]
            x_val_flat = x_val.reshape(-1, n_features)
            x_val_flat = scaler.transform(x_val_flat)
            x_val = x_val_flat.reshape(n_val, n_timesteps, n_features)

            # Transform test data
            n_test = x_test.shape[0]
            x_test_flat = x_test.reshape(-1, n_features)
            x_test_flat = scaler.transform(x_test_flat)
            x_test = x_test_flat.reshape(n_test, n_timesteps, n_features)

        return x_train, y_train, x_val, y_val, x_test, y_test

    # ==================== MotionSense Dataset ====================
    elif data_name == 'motion':
        # Load preprocessed MotionSense data (1s windows at 50Hz)
        x_path = os.path.join(dir, 'Motion_X_1s.npy')
        y_path = os.path.join(dir, 'Motion_Y_1s.npy')
        s_path = os.path.join(dir, 'Motion_Subject_1s.npy')

        if not all(os.path.exists(p) for p in [x_path, y_path, s_path]):
            raise FileNotFoundError(
                f"MotionSense preprocessed files not found. "
                f"Please preprocess the dataset into 1s windows at 50Hz and save as .npy files."
            )

        x_data = np.load(x_path)  # Expected shape: (n_samples, 50, 6) for 50Hz, 6 channels
        y_data = np.load(y_path)  # One-hot encoded labels
        subject_index = np.load(s_path)

        # Shuffle with fixed seed
        np.random.seed(888)
        indices = np.random.permutation(len(x_data))
        x_data = x_data[indices]
        y_data = y_data[indices]
        subject_index = subject_index[indices]

        # Get unique subjects
        unique_subjects = np.unique(subject_index)
        n_subjects = len(unique_subjects)

        # Subject-wise split (60% train, 20% val, 20% test)
        divide = divide_seed

        if divide in [1, 2, 3, 4, 5]:
            # Predefined splits for reproducibility
            n_test = int(n_subjects * 0.2)
            n_val = int(n_subjects * 0.2)

            np.random.seed(divide)
            shuffled_subjects = np.random.permutation(unique_subjects)

            test_subject = set(shuffled_subjects[:n_test])
            val_subject = set(shuffled_subjects[n_test:n_test + n_val])
            train_subject = set(shuffled_subjects[n_test + n_val:])
        else:
            # Random split with custom seed
            np.random.seed(divide if divide else 42)
            shuffled_subjects = np.random.permutation(unique_subjects)

            n_test = int(n_subjects * 0.2)
            n_val = int(n_subjects * 0.2)

            test_subject = set(shuffled_subjects[:n_test])
            val_subject = set(shuffled_subjects[n_test:n_test + n_val])
            train_subject = set(shuffled_subjects[n_test + n_val:])

        # Create masks
        train_mask = np.array([s in train_subject for s in subject_index.flatten()])
        val_mask = np.array([s in val_subject for s in subject_index.flatten()])
        test_mask = np.array([s in test_subject for s in subject_index.flatten()])

        x_train = x_data[train_mask]
        y_train = y_data[train_mask]
        x_val = x_data[val_mask]
        y_val = y_data[val_mask]
        x_test = x_data[test_mask]
        y_test = y_data[test_mask]

        # Apply standardization if requested
        if transformer:
            n_samples, n_timesteps, n_features = x_train.shape

            scaler = StandardScaler()
            x_train_flat = x_train.reshape(-1, n_features)
            x_train_flat = scaler.fit_transform(x_train_flat)
            x_train = x_train_flat.reshape(n_samples, n_timesteps, n_features)

            n_val = x_val.shape[0]
            x_val_flat = x_val.reshape(-1, n_features)
            x_val_flat = scaler.transform(x_val_flat)
            x_val = x_val_flat.reshape(n_val, n_timesteps, n_features)

            n_test = x_test.shape[0]
            x_test_flat = x_test.reshape(-1, n_features)
            x_test_flat = scaler.transform(x_test_flat)
            x_test = x_test_flat.reshape(n_test, n_timesteps, n_features)

        return x_train, y_train, x_val, y_val, x_test, y_test

    # ==================== USC-HAD Dataset ====================
    elif data_name == 'uschad':
        # USC-HAD should already be preprocessed into 1s windows
        x_train = np.load(os.path.join(dir, 'USCHAD_X_Train_1s.npy'))
        y_train = np.load(os.path.join(dir, 'USCHAD_Y_Train_1s.npy'))
        x_val = np.load(os.path.join(dir, 'USCHAD_X_Val_1s.npy'))
        y_val = np.load(os.path.join(dir, 'USCHAD_Y_Val_1s.npy'))
        x_test = np.load(os.path.join(dir, 'USCHAD_X_Test_1s.npy'))
        y_test = np.load(os.path.join(dir, 'USCHAD_Y_Test_1s.npy'))

        # Apply standardization if requested
        if transformer:
            n_samples, n_timesteps, n_features = x_train.shape

            scaler = StandardScaler()
            x_train_flat = x_train.reshape(-1, n_features)
            x_train_flat = scaler.fit_transform(x_train_flat)
            x_train = x_train_flat.reshape(n_samples, n_timesteps, n_features)

            test_samples = x_test.shape[0]
            x_test_flat = x_test.reshape(-1, n_features)
            x_test_flat = scaler.transform(x_test_flat)
            x_test = x_test_flat.reshape(test_samples, n_timesteps, n_features)

            val_samples = x_val.shape[0]
            x_val_flat = x_val.reshape(-1, n_features)
            x_val_flat = scaler.transform(x_val_flat)
            x_val = x_val_flat.reshape(val_samples, n_timesteps, n_features)

        return x_train, y_train, x_val, y_val, x_test, y_test

    else:
        raise NotImplementedError(f"Dataset {data_name} not implemented")