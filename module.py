import numpy as np
import torch
import torch.nn as nn
import os
from utils import span_mask, save_model
from encoder import Encoder


def get_base(dir, data_name, my_type, time_mask, channel_mask, alpha, divide=None, epoch=None):
    if not os.path.exists(dir):
        raise ValueError("the path is not exist")
    dir_pre = os.path.join(dir, data_name)
    dir_suf = None
    if data_name == 'uschad':
        if my_type == 'time':
            dir_suf = 'time{}'.format(time_mask)
        elif my_type == 'spantime':
            dir_suf = 'spantime{}'.format(time_mask)
        elif my_type == 'spantime_channel':
            dir_suf = 'spantime{}_channel{}_alpha{}'.format(time_mask, channel_mask, alpha)
        elif my_type == 'time_channel':
            dir_suf = 'time{}_channel{}_alpha{}'.format(time_mask, channel_mask, alpha)
        elif my_type == 'channel':
            dir_suf = 'channel{}'.format(channel_mask)
        else:
            raise ValueError("the type is not exist")
    else:
        if my_type == 'time':
            dir_suf = 'time{}_divide{}'.format(time_mask, divide)
        elif my_type == 'spantime':
            dir_suf = 'spantime{}_divide{}'.format(time_mask, divide)
        elif my_type == 'spantime_channel':
            dir_suf = 'spantime{}_channel{}_divide{}_alpha{}'.format(time_mask, channel_mask, divide, alpha)
        elif my_type == 'time_channel':
            dir_suf = 'time{}_channel{}_divide{}_alpha{}'.format(time_mask, channel_mask, divide, alpha)
        elif my_type == 'channel':
            dir_suf = 'channel{}_divide{}'.format(channel_mask, divide)
        else:
            raise ValueError("the type is not exist")
    if epoch != 150:
        dir_suf += '_epoch{}'.format(epoch)

    print("the model path is: {}".format(os.path.join(dir_pre, dir_suf)))
    return torch.load(os.path.join(dir_pre, dir_suf))


def get_evaluate(base, n_outputs):
    base_encoder = base.encoder
    base_encoder.requires_grad_(False)

    return nn.Sequential(
        base_encoder,
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Linear(base_encoder.d_model, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(128, n_outputs),
        nn.Softmax(dim=1)
    )


class PretrainModel(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, maximum_position_encoding, n_features):
        super(PretrainModel, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, maximum_position_encoding, n_features=n_features)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, n_features)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, features)
        encoded = self.encoder(x)  # (batch_size, seq_len, d_model)

        # Reshape for BatchNorm1d
        batch_size, seq_len, d_model = encoded.shape
        encoded_reshaped = encoded.view(-1, d_model)
        decoded = self.decoder(encoded_reshaped)
        decoded = decoded.view(batch_size, seq_len, -1)

        return decoded


def get_pretrain_model(num_layers, d_model, num_heads, dff, maximum_position_encoding, n_features):
    return PretrainModel(num_layers, d_model, num_heads, dff, maximum_position_encoding, n_features)


def pre_train(model, data_name, x_train, epoch, batch_size, optimizer, loss_func, my_type, n_timesteps, time_mask,
              n_features=None, channel_mask=None, alpha=None, divide=None):
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
        alpha = alpha * 0.01
        loss = alpha * loss_func(y_t, y_time) + (1 - alpha) * loss_func(y_c, y_channel)

    loss.backward()
    optimizer.step()

    return loss