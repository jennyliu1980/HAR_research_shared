import os
import numpy as np
from sklearn.preprocessing import StandardScaler


def get_data(dir, data_name, transformer=None, normalize_per_channel=True):
    """
    Load HAR datasets without validation set, following original TensorFlow implementation.

    Args:
        dir: dataset directory
        data_name: dataset name ('ucihar', 'motion', 'uschad')
        transformer: whether to apply standardization
        normalize_per_channel: if True, normalize each channel independently (new param)

    Returns exactly 4 values:
        x_train, y_train, x_test, y_test
    """

    if data_name not in ['uschad', 'ucihar', 'motion']:
        raise ValueError(f"Dataset {data_name} is not supported")

    # ==================== UCI-HAR Dataset ====================
    if data_name == 'ucihar':
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
                signal = np.loadtxt(file_path)
                signals.append(signal)
            return np.stack(signals, axis=-1)

        # Load train and test data using ORIGINAL split
        print(f"Loading UCI-HAR with original train/test split...")
        X_train = load_signals('train')  # (7352, 128, 9)
        X_test = load_signals('test')  # (2947, 128, 9)

        # Load labels (1-6 for 6 activities)
        y_train = np.loadtxt(os.path.join(uci_dir, 'train', 'y_train.txt'), dtype=int)
        y_test = np.loadtxt(os.path.join(uci_dir, 'test', 'y_test.txt'), dtype=int)

        # Convert to one-hot encoding
        def to_one_hot(y, n_classes=6):
            n_samples = len(y)
            y_one_hot = np.zeros((n_samples, n_classes))
            y_one_hot[np.arange(n_samples), y - 1] = 1
            return y_one_hot

        y_train = to_one_hot(y_train)
        y_test = to_one_hot(y_test)

        # Apply standardization if requested
        if transformer:
            if normalize_per_channel:
                # Normalize each channel independently (likely what the paper does)
                print("Normalizing each channel independently...")
                n_channels = X_train.shape[2]
                for i in range(n_channels):
                    scaler = StandardScaler()
                    # Fit on train data for this channel
                    X_train[:, :, i] = scaler.fit_transform(
                        X_train[:, :, i].reshape(-1, 1)
                    ).reshape(X_train[:, :, i].shape)
                    # Apply same transformation to test
                    X_test[:, :, i] = scaler.transform(
                        X_test[:, :, i].reshape(-1, 1)
                    ).reshape(X_test[:, :, i].shape)
            else:
                # Original approach - normalize all together
                print("Normalizing all features together...")
                scaler = StandardScaler()
                X_train_flat = X_train.reshape(-1, 9)
                X_train_flat = scaler.fit_transform(X_train_flat)
                X_train = X_train_flat.reshape(-1, 128, 9)

                X_test_flat = X_test.reshape(-1, 9)
                X_test_flat = scaler.transform(X_test_flat)
                X_test = X_test_flat.reshape(-1, 128, 9)

        print(f"UCI-HAR loaded: train={X_train.shape}, test={X_test.shape}")
        return X_train, y_train, X_test, y_test

    # ==================== MotionSense Dataset ====================
    elif data_name == 'motion':
        # Following paper's setup: 80% train, 20% test (subject-wise)
        x_path = os.path.join(dir, 'Motion_X_1s.npy')
        y_path = os.path.join(dir, 'Motion_Y_1s.npy')
        s_path = os.path.join(dir, 'Motion_Subject_1s.npy')

        if not all(os.path.exists(p) for p in [x_path, y_path, s_path]):
            raise FileNotFoundError("MotionSense preprocessed files not found.")

        x_data = np.load(x_path)
        y_data = np.load(y_path)
        subject_index = np.load(s_path)

        # 80-20 split as per paper
        unique_subjects = np.unique(subject_index)
        n_test = int(len(unique_subjects) * 0.2)

        np.random.seed(42)
        np.random.shuffle(unique_subjects)

        test_subjects = set(unique_subjects[:n_test])
        train_subjects = set(unique_subjects[n_test:])

        train_mask = np.array([s in train_subjects for s in subject_index.flatten()])
        test_mask = np.array([s in test_subjects for s in subject_index.flatten()])

        x_train = x_data[train_mask]
        y_train = y_data[train_mask]
        x_test = x_data[test_mask]
        y_test = y_data[test_mask]

        if transformer:
            if normalize_per_channel:
                n_channels = x_train.shape[2]
                for i in range(n_channels):
                    scaler = StandardScaler()
                    x_train[:, :, i] = scaler.fit_transform(
                        x_train[:, :, i].reshape(-1, 1)
                    ).reshape(x_train[:, :, i].shape)
                    x_test[:, :, i] = scaler.transform(
                        x_test[:, :, i].reshape(-1, 1)
                    ).reshape(x_test[:, :, i].shape)
            else:
                scaler = StandardScaler()
                x_train = scaler.fit_transform(x_train.reshape(-1, 6)).reshape(x_train.shape)
                x_test = scaler.transform(x_test.reshape(-1, 6)).reshape(x_test.shape)

        return x_train, y_train, x_test, y_test

    # ==================== USC-HAD Dataset ====================
    elif data_name == 'uschad':
        # Following paper: subjects 1-10 train, 11-14 test
        x_path = os.path.join(dir, 'USCHAD_X.npy')
        y_path = os.path.join(dir, 'USCHAD_Y.npy')
        s_path = os.path.join(dir, 'USCHAD_Subject.npy')

        if not all(os.path.exists(p) for p in [x_path, y_path, s_path]):
            raise FileNotFoundError("USC-HAD preprocessed files not found.")

        x_data = np.load(x_path)
        y_data = np.load(y_path)
        subject_index = np.load(s_path)

        # Fixed split as per paper
        train_subjects = set(range(1, 11))  # subjects 1-10
        test_subjects = set(range(11, 15))  # subjects 11-14

        train_mask = np.array([s in train_subjects for s in subject_index.flatten()])
        test_mask = np.array([s in test_subjects for s in subject_index.flatten()])

        x_train = x_data[train_mask]
        y_train = y_data[train_mask]
        x_test = x_data[test_mask]
        y_test = y_data[test_mask]

        if transformer:
            if normalize_per_channel:
                n_channels = x_train.shape[2]
                for i in range(n_channels):
                    scaler = StandardScaler()
                    x_train[:, :, i] = scaler.fit_transform(
                        x_train[:, :, i].reshape(-1, 1)
                    ).reshape(x_train[:, :, i].shape)
                    x_test[:, :, i] = scaler.transform(
                        x_test[:, :, i].reshape(-1, 1)
                    ).reshape(x_test[:, :, i].shape)
            else:
                scaler = StandardScaler()
                x_train = scaler.fit_transform(x_train.reshape(-1, 6)).reshape(x_train.shape)
                x_test = scaler.transform(x_test.reshape(-1, 6)).reshape(x_test.shape)

        return x_train, y_train, x_test, y_test