import numpy as np
import torch
import torch.nn as nn
import os
from utils import span_mask, save_model
from encoder import Encoder


def get_base(dir, data_name, my_type, time_mask, channel_mask, alpha, divide=None, epoch=None):
    """Load pretrained model - handles both regular and robust models"""
    if not os.path.exists(dir):
        raise ValueError("the path is not exist")

    dir_pre = os.path.join(dir, data_name)

    # Handle robust suffix
    is_robust = False
    if my_type.endswith('_robust'):
        is_robust = True
        base_type = my_type.replace('_robust', '')
    else:
        base_type = my_type

    if data_name == 'uschad':
        if base_type == 'time':
            dir_suf = 'time{}'.format(time_mask)
        elif base_type == 'spantime':
            dir_suf = 'spantime{}'.format(time_mask)
        elif base_type == 'spantime_channel':
            dir_suf = 'spantime{}_channel{}_alpha{}'.format(time_mask, channel_mask, alpha)
        elif base_type == 'time_channel':
            dir_suf = 'time{}_channel{}_alpha{}'.format(time_mask, channel_mask, alpha)
        elif base_type == 'channel':
            dir_suf = 'channel{}'.format(channel_mask)
        else:
            raise ValueError("the type is not exist")
    else:
        if base_type == 'time':
            dir_suf = 'time{}_divide{}'.format(time_mask, divide)
        elif base_type == 'spantime':
            dir_suf = 'spantime{}_divide{}'.format(time_mask, divide)
        elif base_type == 'spantime_channel':
            dir_suf = 'spantime{}_channel{}_divide{}_alpha{}'.format(time_mask, channel_mask, divide, alpha)
        elif base_type == 'time_channel':
            dir_suf = 'time{}_channel{}_divide{}_alpha{}'.format(time_mask, channel_mask, divide, alpha)
        elif base_type == 'channel':
            dir_suf = 'channel{}_divide{}'.format(channel_mask, divide)
        else:
            raise ValueError("the type is not exist")

    if epoch is not None and epoch != 150:
        dir_suf += '_epoch{}'.format(epoch)

    # Add robust suffix if needed
    if is_robust:
        dir_suf += '_robust'

    model_path = os.path.join(dir_pre, dir_suf)
    print("Loading pretrained model from: {}".format(model_path))

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return torch.load(model_path)


def get_evaluate(base, n_outputs):
    """Complex evaluation head - likely what the paper uses"""
    base_encoder = base.encoder
    base_encoder.requires_grad_(False)  # Freeze encoder

    class ComplexClassifier(nn.Module):
        def __init__(self, encoder, n_outputs):
            super().__init__()
            self.encoder = encoder
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.flatten = nn.Flatten()

            # Complex classifier with BatchNorm and Dropout
            self.classifier = nn.Sequential(
                nn.Linear(encoder.d_model, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, n_outputs)
                # No softmax - CrossEntropyLoss expects raw logits
            )

        def forward(self, x):
            x = self.encoder(x)  # (batch, seq_len, d_model)
            x = x.transpose(1, 2)  # (batch, d_model, seq_len)
            x = self.pool(x)  # (batch, d_model, 1)
            x = self.flatten(x)  # (batch, d_model)
            x = self.classifier(x)  # (batch, n_outputs) - raw logits
            return x

    return ComplexClassifier(base_encoder, n_outputs)


def get_evaluate_simple(base, n_outputs):
    """Simple evaluation head - minimal architecture"""
    base_encoder = base.encoder
    base_encoder.requires_grad_(False)  # Freeze encoder

    class SimpleClassifier(nn.Module):
        def __init__(self, encoder, n_outputs):
            super().__init__()
            self.encoder = encoder
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(encoder.d_model, n_outputs)
            # No softmax - CrossEntropyLoss expects raw logits

        def forward(self, x):
            x = self.encoder(x)  # (batch, seq_len, d_model)
            x = x.transpose(1, 2)  # (batch, d_model, seq_len)
            x = self.pool(x)  # (batch, d_model, 1)
            x = self.flatten(x)  # (batch, d_model)
            x = self.fc(x)  # (batch, n_outputs) - raw logits
            return x

    return SimpleClassifier(base_encoder, n_outputs)


class PretrainModel(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, maximum_position_encoding, n_features):
        super(PretrainModel, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, maximum_position_encoding, n_features=n_features)
        # Simple decoder - just one linear layer like original
        self.decoder = nn.Linear(d_model, n_features)

    def forward(self, x):
        encoded = self.encoder(x)  # (batch_size, seq_len, d_model)
        decoded = self.decoder(encoded)  # (batch_size, seq_len, n_features)
        return decoded


def get_pretrain_model(num_layers, d_model, num_heads, dff, maximum_position_encoding, n_features):
    return PretrainModel(num_layers, d_model, num_heads, dff, maximum_position_encoding, n_features)


def pre_train(model, data_name, x_train, epoch, batch_size, optimizer, loss_func, my_type, n_timesteps, time_mask,
              n_features=None, channel_mask=None, alpha=None, divide=None):
    """Legacy pretraining function for compatibility"""
    print("To begin the model, data:{} epoch:{} batchsize:{} type:{}".format(data_name, epoch, batch_size, my_type))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()

    # Convert numpy to torch tensor
    x_train = torch.from_numpy(x_train).float()
    dataset = torch.utils.data.TensorDataset(x_train)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    cur_loss = 1e4
    for i in range(epoch):
        loss_batch = []
        for batch in train_loader:
            x = batch[0].to(device)
            x_np = x.cpu().numpy().copy()

            time_index = None
            y_time = None
            if my_type in ['time', 'time_channel']:
                time_index = np.random.choice(n_timesteps, int(n_timesteps * time_mask * 0.01), replace=False)
                y_time = torch.from_numpy(x_np[:, time_index, :]).float().to(device)
                x_np[:, time_index, :] = 0
            elif my_type in ['spantime', 'spantime_channel']:
                time_index = span_mask(n_timesteps, goal_num_predict=int(n_timesteps * time_mask * 0.01))
                y_time = torch.from_numpy(x_np[:, time_index, :]).float().to(device)
                x_np[:, time_index, :] = 0

            y_channel, channel_index = None, None
            if my_type in ['spantime_channel', 'time_channel', 'channel']:
                channel_index = np.random.choice(n_features, channel_mask, replace=False)
                y_channel = torch.from_numpy(x_np[:, :, channel_index]).float().to(device)
                x_np[:, :, channel_index] = 0

            x_mask = torch.from_numpy(x_np).float().to(device)

            loss = train_step(model, my_type, optimizer, loss_func, x_mask, y_time, time_index, y_channel,
                              channel_index, alpha)

            loss_batch.append(loss.item())

        epoch_loss_last = np.mean(loss_batch)
        print('epoch:{} ==> loss:{}'.format(i + 1, epoch_loss_last))

        if i > int(epoch * 2 // 3) and epoch_loss_last < cur_loss:
            model_dir = save_model(data_name, my_type, time_mask, channel_mask, alpha, divide, model, epoch)
            cur_loss = epoch_loss_last
            print("epoch{} the model is saved in {}".format(i + 1, model_dir))


def train_step(model, my_type, optimizer, loss_func, x, y_time, time_index, y_channel=None, channel_index=None,
               alpha=None):
    optimizer.zero_grad()

    out = model(x)

    if my_type in ['time', 'spantime', 'spantime_channel', 'time_channel']:
        y_t = out[:, time_index, :]
    if my_type in ['channel', 'time_channel', 'spantime_channel']:
        y_c = out[:, :, channel_index]

    if my_type in ['time', 'spantime']:
        loss = loss_func(y_t, y_time)
    elif my_type in ['channel']:
        loss = loss_func(y_c, y_channel)
    elif my_type in ['spantime_channel', 'time_channel']:
        loss = alpha * loss_func(y_t, y_time) + (1 - alpha) * loss_func(y_c, y_channel)

    loss.backward()
    optimizer.step()

    return loss