import torch
import numpy as np
import wandb
from module import get_pretrain_model, PretrainModel
from dataset import get_data
from utils import span_mask, save_model
import argparse
import time
from pathlib import Path

parser = argparse.ArgumentParser(description='Self-supervised pretraining for HAR with WandB')

# Dataset arguments
parser.add_argument('--dir', type=str, default='datasets/sub', help='dataset path')
parser.add_argument('--dataset', type=str, default='ucihar',
                    choices=['ucihar', 'motion', 'uschad'], help='dataset name')
parser.add_argument('--seed', type=int, default=100,
                    help='random seed for dataset division')

# Masking strategy arguments
parser.add_argument('--type', type=str, default='channel',
                    choices=['time', 'spantime', 'spantime_channel', 'time_channel', 'channel'],
                    help='masking strategy type')
parser.add_argument('--channel_mask', type=int, default=3,
                    help='number of channels to mask')
parser.add_argument('--time_mask', type=int, default=30,
                    help='percentage of time steps to mask')
parser.add_argument('--alpha', type=float, default=0.5,
                    help='weight balance between time and channel loss (0-1)')

# Model architecture arguments
parser.add_argument('--num_layers', type=int, default=3,
                    help='number of transformer encoder layers')
parser.add_argument('--num_heads', type=int, default=4,
                    help='number of attention heads')
parser.add_argument('--dff', type=int, default=256,
                    help='dimension of feedforward network')
parser.add_argument('--d_model', type=int, default=128,
                    help='dimension of transformer embeddings')

# Training arguments
parser.add_argument('--batch_size', type=int, default=256,
                    help='batch size for pretraining')
parser.add_argument('--epoch', type=int, default=150,
                    help='number of pretraining epochs')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate')

# WandB arguments
parser.add_argument('--wandb_project', type=str, default='har-masking',
                    help='WandB project name')
parser.add_argument('--wandb_entity', type=str, default=None,
                    help='WandB entity/team name')
parser.add_argument('--wandb_name', type=str, default=None,
                    help='WandB run name')
parser.add_argument('--no_wandb', action='store_true',
                    help='Disable WandB logging')


def train_step(model, my_type, optimizer, loss_func, x, y_time, time_index,
               y_channel=None, channel_index=None, alpha=None):
    """Single training step with loss calculation"""
    optimizer.zero_grad()

    out = model(x)

    # Calculate masked predictions
    if my_type in ['time', 'spantime', 'spantime_channel', 'time_channel']:
        y_t = out[:, time_index, :]
    if my_type in ['channel', 'time_channel', 'spantime_channel']:
        y_c = out[:, :, channel_index]

    # Calculate loss based on masking type
    if my_type in ['time', 'spantime']:
        loss = loss_func(y_t, y_time)
        time_loss = loss.item()
        channel_loss = 0
    elif my_type in ['channel']:
        loss = loss_func(y_c, y_channel)
        time_loss = 0
        channel_loss = loss.item()
    elif my_type in ['spantime_channel', 'time_channel']:
        alpha_scaled = alpha * 0.01
        time_loss_val = loss_func(y_t, y_time)
        channel_loss_val = loss_func(y_c, y_channel)
        loss = alpha_scaled * time_loss_val + (1 - alpha_scaled) * channel_loss_val
        time_loss = time_loss_val.item()
        channel_loss = channel_loss_val.item()

    loss.backward()
    optimizer.step()

    return loss, time_loss, channel_loss


def pretrain_with_wandb(model, data_name, x_train, args):
    """Pretraining with WandB logging"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()

    # Setup optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = torch.nn.MSELoss()

    # Convert numpy to torch tensor
    x_train = torch.from_numpy(x_train).float()
    dataset = torch.utils.data.TensorDataset(x_train)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    n_timesteps = x_train.shape[1]
    n_features = x_train.shape[2]

    best_loss = float('inf')
    best_epoch = 0

    print(f"Training on {device}")
    print(f"Total batches per epoch: {len(train_loader)}")

    for epoch in range(args.epoch):
        epoch_losses = []
        epoch_time_losses = []
        epoch_channel_losses = []

        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            x = batch[0].to(device)
            x_np = x.cpu().numpy().copy()

            # Apply masking based on type
            time_index = None
            y_time = None
            if args.type in ['time', 'time_channel']:
                time_index = np.random.choice(n_timesteps,
                                              int(n_timesteps * args.time_mask * 0.01),
                                              replace=False)
                y_time = torch.from_numpy(x_np[:, time_index, :]).float().to(device)
                x_np[:, time_index, :] = 0
            elif args.type in ['spantime', 'spantime_channel']:
                time_index = span_mask(n_timesteps,
                                       goal_num_predict=int(n_timesteps * args.time_mask * 0.01))
                y_time = torch.from_numpy(x_np[:, time_index, :]).float().to(device)
                x_np[:, time_index, :] = 0

            y_channel, channel_index = None, None
            if args.type in ['spantime_channel', 'time_channel', 'channel']:
                channel_index = np.random.choice(n_features, args.channel_mask, replace=False)
                y_channel = torch.from_numpy(x_np[:, :, channel_index]).float().to(device)
                x_np[:, :, channel_index] = 0

            x_mask = torch.from_numpy(x_np).float().to(device)

            # Training step
            loss, time_loss, channel_loss = train_step(
                model, args.type, optimizer, loss_func, x_mask,
                y_time, time_index, y_channel, channel_index, args.alpha
            )

            epoch_losses.append(loss.item())
            epoch_time_losses.append(time_loss)
            epoch_channel_losses.append(channel_loss)

            # Log batch metrics
            if not args.no_wandb and batch_idx % 10 == 0:
                wandb.log({
                    'batch_loss': loss.item(),
                    'batch_time_loss': time_loss,
                    'batch_channel_loss': channel_loss,
                    'batch': epoch * len(train_loader) + batch_idx
                })

        # Epoch metrics
        epoch_time = time.time() - epoch_start
        avg_loss = np.mean(epoch_losses)
        avg_time_loss = np.mean(epoch_time_losses)
        avg_channel_loss = np.mean(epoch_channel_losses)

        print(f'Epoch [{epoch + 1}/{args.epoch}] - '
              f'Loss: {avg_loss:.4f} - '
              f'Time Loss: {avg_time_loss:.4f} - '
              f'Channel Loss: {avg_channel_loss:.4f} - '
              f'Time: {epoch_time:.1f}s')

        # Log epoch metrics
        if not args.no_wandb:
            log_dict = {
                'epoch': epoch + 1,
                'epoch_loss': avg_loss,
                'epoch_time_loss': avg_time_loss,
                'epoch_channel_loss': avg_channel_loss,
                'epoch_time': epoch_time,
                'learning_rate': optimizer.param_groups[0]['lr']
            }
            wandb.log(log_dict)

        # Save best model (after 2/3 of epochs)
        if epoch > int(args.epoch * 2 / 3) and avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1

            model_dir = save_model(
                data_name, args.type, args.time_mask, args.channel_mask,
                args.alpha, args.seed if data_name != 'uschad' else None,
                model, args.epoch
            )

            print(f"  -> New best model saved at epoch {best_epoch} (loss: {best_loss:.4f})")

            if not args.no_wandb:
                wandb.log({
                    'best_loss': best_loss,
                    'best_epoch': best_epoch
                })

    return best_loss, best_epoch


if __name__ == '__main__':
    args = parser.parse_args()

    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Initialize WandB
    if not args.no_wandb:
        # Create run name
        if args.wandb_name is None:
            args.wandb_name = f"{args.dataset}_{args.type}_tm{args.time_mask}_cm{args.channel_mask}_a{args.alpha}_s{args.seed}"

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config=vars(args),
            tags=[args.dataset, args.type, f"seed{args.seed}"]
        )

    print("=" * 60)
    print("Self-supervised Pretraining for HAR")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Masking: {args.type}")
    if args.type in ['time', 'spantime', 'time_channel', 'spantime_channel']:
        print(f"  Time mask: {args.time_mask}%")
    if args.type in ['channel', 'time_channel', 'spantime_channel']:
        print(f"  Channel mask: {args.channel_mask}")
    if args.type in ['time_channel', 'spantime_channel']:
        print(f"  Alpha: {args.alpha}")
    print("=" * 60)

    # Load dataset
    print(f"\nLoading {args.dataset} dataset...")
    x_train, y_train, _, _, _, _ = get_data(
        args.dir, args.dataset, transformer=True, divide_seed=args.seed
    )

    n_samples, n_timesteps, n_features = x_train.shape
    n_outputs = y_train.shape[1]

    print(f"Data shape: {x_train.shape}")
    print(f"  Samples: {n_samples}")
    print(f"  Timesteps: {n_timesteps}")
    print(f"  Features: {n_features}")
    print(f"  Classes: {n_outputs}")

    # Create model
    print(f"\nInitializing model...")
    model = get_pretrain_model(
        args.num_layers, args.d_model, args.num_heads, args.dff,
        maximum_position_encoding=n_timesteps, n_features=n_features
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Log model architecture to WandB
    #if not args.no_wandb:
       # wandb.watch(model, log='all', log_freq=100)

    # Start pretraining
    print(f"\nStarting pretraining...")
    print(f"  Epochs: {args.epoch}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print("=" * 60)

    best_loss, best_epoch = pretrain_with_wandb(model, args.dataset, x_train, args)

    print("\n" + "=" * 60)
    print("Pretraining completed!")
    print(f"Best loss: {best_loss:.4f} at epoch {best_epoch}")
    print("=" * 60)

    if not args.no_wandb:
        wandb.finish()