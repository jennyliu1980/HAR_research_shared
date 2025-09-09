import torch
import numpy as np
import wandb
from module import get_pretrain_model
from dataset import get_data
from utils import span_mask, save_model
import argparse
import time
import os
import random

parser = argparse.ArgumentParser(description='Self-supervised pretraining for HAR')

# Dataset arguments
parser.add_argument('--dir', type=str, default='datasets/sub', help='dataset path')
parser.add_argument('--dataset', type=str, default='ucihar',
                    choices=['ucihar', 'motion', 'uschad'])

# Masking strategy arguments
parser.add_argument('--type', type=str, default='channel',
                    choices=['time', 'spantime', 'spantime_channel', 'time_channel', 'channel'])
parser.add_argument('--channel_mask', type=int, default=3)
parser.add_argument('--time_mask', type=int, default=15)
parser.add_argument('--alpha', type=float, default=0.5)

# Model architecture arguments
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--num_heads', type=int, default=4)
parser.add_argument('--dff', type=int, default=256)
parser.add_argument('--d_model', type=int, default=128)

# Training arguments
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epoch', type=int, default=150)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--scheduler', type=str, default='cosine',
                    choices=['cosine', 'onecycle', 'step'], help='Learning rate scheduler')
parser.add_argument('--warmup_pct', type=float, default=0.1,
                    help='Percentage of epochs for warmup')
parser.add_argument('--normalize_per_channel', type=bool, default=True,
                    help='Normalize each channel independently')

# WandB arguments
parser.add_argument('--wandb_project', type=str, default='har-masking-pretrain')
parser.add_argument('--wandb_entity', type=str, default=None)
parser.add_argument('--no_wandb', action='store_true')
parser.add_argument('--force_retrain', action='store_true', help='Force retraining even if model exists')


def set_random_seeds(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_existing_model(dataset, masking_type, time_mask, channel_mask, alpha):
    """Check if pretrained model already exists"""
    model_dir = f"model/{dataset}/"
    divide = 100 if dataset != 'uschad' else None

    if masking_type == 'spantime_channel':
        if divide:
            model_file = f"spantime{time_mask}_channel{channel_mask}_divide{divide}_alpha{alpha}"
        else:
            model_file = f"spantime{time_mask}_channel{channel_mask}_alpha{alpha}"
    elif masking_type == 'time_channel':
        if divide:
            model_file = f"time{time_mask}_channel{channel_mask}_divide{divide}_alpha{alpha}"
        else:
            model_file = f"time{time_mask}_channel{channel_mask}_alpha{alpha}"
    elif masking_type == 'channel':
        if divide:
            model_file = f"channel{channel_mask}_divide{divide}"
        else:
            model_file = f"channel{channel_mask}"
    elif masking_type == 'spantime':
        if divide:
            model_file = f"spantime{time_mask}_divide{divide}"
        else:
            model_file = f"spantime{time_mask}"
    elif masking_type == 'time':
        if divide:
            model_file = f"time{time_mask}_divide{divide}"
        else:
            model_file = f"time{time_mask}"

    model_path = os.path.join(model_dir, model_file)
    return os.path.exists(model_path), model_path


def train_step(model, my_type, optimizer, loss_func, x, y_time, time_index,
               y_channel=None, channel_index=None, alpha=None):
    optimizer.zero_grad()
    out = model(x)

    if my_type in ['time', 'spantime', 'spantime_channel', 'time_channel']:
        y_t = out[:, time_index, :]
    if my_type in ['channel', 'time_channel', 'spantime_channel']:
        y_c = out[:, :, channel_index]

    if my_type in ['time', 'spantime']:
        loss = loss_func(y_t, y_time)
        time_loss = loss.item()
        channel_loss = 0
    elif my_type == 'channel':
        loss = loss_func(y_c, y_channel)
        time_loss = 0
        channel_loss = loss.item()
    elif my_type in ['spantime_channel', 'time_channel']:
        time_loss_val = loss_func(y_t, y_time)
        channel_loss_val = loss_func(y_c, y_channel)
        loss = alpha * time_loss_val + (1 - alpha) * channel_loss_val
        time_loss = time_loss_val.item()
        channel_loss = channel_loss_val.item()

    loss.backward()
    optimizer.step()
    return loss, time_loss, channel_loss


def get_scheduler(optimizer, args, total_steps):
    """Get learning rate scheduler"""
    if args.scheduler == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epoch, eta_min=1e-5
        )
    elif args.scheduler == 'onecycle':
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            total_steps=total_steps,
            pct_start=args.warmup_pct,
            anneal_strategy='cos'
        )
    elif args.scheduler == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=50, gamma=0.5
        )
    else:
        return None


def pretrain_with_wandb(model, data_name, x_train, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = torch.nn.MSELoss()

    x_train = torch.from_numpy(x_train).float()
    dataset = torch.utils.data.TensorDataset(x_train)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Setup scheduler
    total_steps = len(train_loader) * args.epoch
    scheduler = get_scheduler(optimizer, args, total_steps)

    n_timesteps = x_train.shape[1]
    n_features = x_train.shape[2]

    best_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    patience = 20  # Early stopping patience

    print(f"Training on {device}")
    print(f"Total batches per epoch: {len(train_loader)}")
    print(f"Scheduler: {args.scheduler}")

    for epoch in range(args.epoch):
        epoch_losses = []
        epoch_time_losses = []
        epoch_channel_losses = []
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            x = batch[0].to(device)
            x_np = x.cpu().numpy().copy()

            # Apply masking
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

            loss, time_loss, channel_loss = train_step(
                model, args.type, optimizer, loss_func, x_mask,
                y_time, time_index, y_channel, channel_index, args.alpha
            )

            # Step scheduler if OneCycle (per batch)
            if args.scheduler == 'onecycle':
                scheduler.step()

            epoch_losses.append(loss.item())
            epoch_time_losses.append(time_loss)
            epoch_channel_losses.append(channel_loss)

            if not args.no_wandb and batch_idx % 10 == 0:
                wandb.log({
                    'batch_loss': loss.item(),
                    'batch_time_loss': time_loss,
                    'batch_channel_loss': channel_loss,
                    'batch': epoch * len(train_loader) + batch_idx,
                    'learning_rate': optimizer.param_groups[0]['lr']
                })

        # Step scheduler if not OneCycle (per epoch)
        if scheduler and args.scheduler != 'onecycle':
            scheduler.step()

        epoch_time = time.time() - epoch_start
        avg_loss = np.mean(epoch_losses)
        avg_time_loss = np.mean(epoch_time_losses)
        avg_channel_loss = np.mean(epoch_channel_losses)

        current_lr = optimizer.param_groups[0]['lr']

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{args.epoch}] - '
                  f'Loss: {avg_loss:.4f} - '
                  f'Time Loss: {avg_time_loss:.4f} - '
                  f'Channel Loss: {avg_channel_loss:.4f} - '
                  f'LR: {current_lr:.6f} - '
                  f'Time: {epoch_time:.1f}s')

        if not args.no_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'epoch_loss': avg_loss,
                'epoch_time_loss': avg_time_loss,
                'epoch_channel_loss': avg_channel_loss,
                'epoch_time': epoch_time,
                'learning_rate': current_lr
            })

        # Save best model (after warmup period)
        if epoch > int(args.epoch * 0.5) and avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1
            patience_counter = 0

            divide = None if data_name == 'uschad' else 100
            model_dir = save_model(data_name, args.type, args.time_mask, args.channel_mask,
                                   args.alpha, divide, model, args.epoch)
            print(f"  -> Best model saved at epoch {best_epoch} (loss: {best_loss:.4f})")

            if not args.no_wandb:
                wandb.log({'best_loss': best_loss, 'best_epoch': best_epoch})
        elif epoch > int(args.epoch * 0.5):
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    return best_loss, best_epoch


if __name__ == '__main__':
    args = parser.parse_args()

    # Check if model already exists
    exists, model_path = check_existing_model(args.dataset, args.type,
                                              args.time_mask, args.channel_mask, args.alpha)

    if exists and not args.force_retrain:
        print("=" * 60)
        print(f"Pretrained model already exists at: {model_path}")
        print("Use --force_retrain to retrain anyway")
        print("=" * 60)
        exit(0)

    # Set random seeds for reproducibility
    set_random_seeds(42)

    # Initialize WandB
    wandb_run_name = f"pretrain_{args.dataset}_{args.type}_tm{args.time_mask}_cm{args.channel_mask}_a{args.alpha}_{args.scheduler}"

    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=wandb_run_name,
            config=vars(args),
            tags=[args.dataset, args.type, "pretrain", args.scheduler]
        )

    print("=" * 60)
    print("Self-supervised Pretraining for HAR")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Masking: {args.type}")
    print(f"Parameters: time_mask={args.time_mask}%, channel_mask={args.channel_mask}, alpha={args.alpha}")

    # Load dataset with new normalization
    print(f"\nLoading {args.dataset} dataset...")
    x_train, y_train, _, _ = get_data(
        args.dir, args.dataset,
        transformer=True,
        normalize_per_channel=args.normalize_per_channel
    )

    n_samples, n_timesteps, n_features = x_train.shape
    print(f"Data shape: {x_train.shape}")

    # Create model
    print(f"\nInitializing model...")
    model = get_pretrain_model(args.num_layers, args.d_model, args.num_heads, args.dff,
                               maximum_position_encoding=n_timesteps, n_features=n_features)

    # Start pretraining
    print(f"\nStarting pretraining...")
    best_loss, best_epoch = pretrain_with_wandb(model, args.dataset, x_train, args)

    print("\n" + "=" * 60)
    print(f"Pretraining completed!")
    print(f"Best loss: {best_loss:.4f} at epoch {best_epoch}")
    print("=" * 60)

    if not args.no_wandb:
        wandb.finish()