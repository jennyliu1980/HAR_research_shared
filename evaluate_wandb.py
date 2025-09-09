import torch
import torch.nn as nn
import numpy as np
import wandb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import argparse
from dataset import get_data
from module import get_base, get_evaluate, get_evaluate_simple
import time
import os
import json
import random

parser = argparse.ArgumentParser(description='Fine-tuning for HAR with WandB')

# Dataset parameters
parser.add_argument('--dir', type=str, default='datasets/sub', help='dataset path')
parser.add_argument('--dataset', type=str, default='ucihar',
                    choices=['ucihar', 'motion', 'uschad'], help='dataset')

# Model parameters
parser.add_argument('--model_dir', type=str, default='model', help='pretrained model directory')
parser.add_argument('--type', type=str, default='channel',
                    choices=['time', 'spantime', 'spantime_channel', 'time_channel', 'channel'])
parser.add_argument('--channel_mask', type=int, default=3)
parser.add_argument('--time_mask', type=int, default=15)
parser.add_argument('--alpha', type=float, default=0.5)

# Training parameters
parser.add_argument('--batch_size', type=int, default=1024, help='batch size for fine-tuning')
parser.add_argument('--ft_epoch', type=int, default=100, help='number of fine-tuning epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate for fine-tuning')

# New parameters for experimentation
parser.add_argument('--eval_head', type=str, default='complex',
                    choices=['simple', 'complex'], help='Evaluation head type')
parser.add_argument('--normalize_per_channel', type=bool, default=True,
                    help='Normalize each channel independently')
parser.add_argument('--optimizer', type=str, default='adam',
                    choices=['adam', 'adamw', 'sgd'], help='Optimizer type')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for optimizer')
parser.add_argument('--scheduler', type=str, default='none',
                    choices=['none', 'cosine', 'step', 'onecycle'], help='Learning rate scheduler')
parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate in evaluation head')
parser.add_argument('--warmup_epochs', type=int, default=10, help='Number of warmup epochs')

# WandB parameters
parser.add_argument('--wandb_project', type=str, default='har-masking-final',
                    help='WandB project name')
parser.add_argument('--wandb_entity', type=str, default=None,
                    help='WandB entity/team name')
parser.add_argument('--no_wandb', action='store_true',
                    help='Disable WandB logging')
parser.add_argument('--exp_suffix', type=str, default='',
                    help='Suffix for experiment name')


def set_random_seeds(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_evaluate_complex(base, n_outputs, dropout_rate=0.1):
    """Complex evaluation head with configurable dropout"""
    base_encoder = base.encoder
    base_encoder.requires_grad_(False)

    class ComplexClassifier(nn.Module):
        def __init__(self, encoder, n_outputs, dropout_rate):
            super().__init__()
            self.encoder = encoder
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.flatten = nn.Flatten()

            # Complex head with BatchNorm and Dropout
            self.classifier = nn.Sequential(
                nn.Linear(encoder.d_model, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(128, n_outputs)
            )

        def forward(self, x):
            x = self.encoder(x)  # (batch, seq_len, d_model)
            x = x.transpose(1, 2)  # (batch, d_model, seq_len)
            x = self.pool(x)  # (batch, d_model, 1)
            x = self.flatten(x)  # (batch, d_model)
            x = self.classifier(x)  # (batch, n_outputs)
            return x

    return ComplexClassifier(base_encoder, n_outputs, dropout_rate)


def evaluate_model(model, data_loader, device, num_classes):
    """Evaluate model and return comprehensive metrics"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            preds = torch.argmax(outputs, dim=1)
            labels = torch.argmax(batch_y, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_per_class = f1_score(all_labels, all_preds, average=None)
    cm = confusion_matrix(all_labels, all_preds)

    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision': precision,
        'recall': recall,
        'f1_per_class': f1_per_class,
        'confusion_matrix': cm
    }


def get_optimizer(model, args):
    """Get optimizer based on arguments"""
    params = filter(lambda p: p.requires_grad, model.parameters())

    if args.optimizer == 'adam':
        return torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        return torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")


def get_scheduler(optimizer, args, total_steps):
    """Get learning rate scheduler based on arguments"""
    if args.scheduler == 'none':
        return None
    elif args.scheduler == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.ft_epoch, eta_min=1e-5
        )
    elif args.scheduler == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.1
        )
    elif args.scheduler == 'onecycle':
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr, total_steps=total_steps,
            pct_start=args.warmup_epochs / args.ft_epoch
        )
    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler}")


def fine_tune_with_wandb(model, x_train, y_train, x_test, y_test, args):
    """Fine-tune model without validation set"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Freeze encoder (first component of the model)
    if hasattr(model, 'encoder'):
        for param in model.encoder.parameters():
            param.requires_grad = False
        print("Encoder weights frozen")

    # Convert to tensors
    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).float()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).float()

    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, args)

    # Scheduler
    total_steps = len(train_loader) * args.ft_epoch
    scheduler = get_scheduler(optimizer, args, total_steps)

    num_classes = y_train.shape[1]
    best_test_f1 = 0
    best_model_state = None
    best_epoch = 0

    print(f"Training on {device}")
    print(f"Train samples: {len(x_train)}, Test samples: {len(x_test)}")
    print(f"Optimizer: {args.optimizer}, LR: {args.lr}, WD: {args.weight_decay}")
    print(f"Scheduler: {args.scheduler}, Eval Head: {args.eval_head}")

    # Training loop
    for epoch in range(args.ft_epoch):
        epoch_start = time.time()

        # Training
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            labels = torch.argmax(batch_y, dim=1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if args.scheduler == 'onecycle':
                scheduler.step()

            train_loss += loss.item()
            train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        # Step scheduler (except for onecycle which steps per batch)
        if scheduler and args.scheduler != 'onecycle':
            scheduler.step()

        # Training metrics
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='macro')
        avg_train_loss = train_loss / len(train_loader)

        # Test evaluation
        test_metrics = evaluate_model(model, test_loader, device, num_classes)

        epoch_time = time.time() - epoch_start

        current_lr = optimizer.param_groups[0]['lr']

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{args.ft_epoch}] ({epoch_time:.1f}s) LR: {current_lr:.6f}')
            print(f'  Train - Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}')
            print(f'  Test  - Acc: {test_metrics["accuracy"]:.4f}, F1: {test_metrics["f1_macro"]:.4f}')

        # Log to WandB
        if not args.no_wandb:
            log_dict = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'train_acc': train_acc,
                'train_f1': train_f1,
                'test_acc': test_metrics['accuracy'],
                'test_f1_macro': test_metrics['f1_macro'],
                'test_f1_weighted': test_metrics['f1_weighted'],
                'test_precision': test_metrics['precision'],
                'test_recall': test_metrics['recall'],
                'learning_rate': current_lr,
                'epoch_time': epoch_time
            }
            wandb.log(log_dict)

        # Save best model
        if test_metrics['f1_macro'] > best_test_f1:
            best_test_f1 = test_metrics['f1_macro']
            best_model_state = model.state_dict().copy()
            best_epoch = epoch + 1
            print(f'  -> New best test F1: {best_test_f1:.4f}')

    # Load best model for final evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    test_metrics = evaluate_model(model, test_loader, device, num_classes)

    print("\n" + "=" * 50)
    print(f"Final Test Results:")
    print(f"  F1 Score: {test_metrics['f1_macro']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  Best F1: {best_test_f1:.4f} at epoch {best_epoch}")
    print("=" * 50)

    # Log final results to WandB
    if not args.no_wandb:
        wandb.summary['final_test_accuracy'] = test_metrics['accuracy']
        wandb.summary['final_test_f1_macro'] = test_metrics['f1_macro']
        wandb.summary['final_test_f1_weighted'] = test_metrics['f1_weighted']
        wandb.summary['final_test_precision'] = test_metrics['precision']
        wandb.summary['final_test_recall'] = test_metrics['recall']
        wandb.summary['best_epoch'] = best_epoch
        wandb.summary['eval_head'] = args.eval_head
        wandb.summary['optimizer'] = args.optimizer
        wandb.summary['scheduler'] = args.scheduler

    return test_metrics


def save_experiment_report(args, test_metrics, wandb_run_name, wandb_url):
    """Save detailed experiment report to file"""

    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'experiment_name': f"{args.type}_tm{args.time_mask}_cm{args.channel_mask}_a{args.alpha}_{args.exp_suffix}",
        'wandb_run_name': wandb_run_name,
        'wandb_url': wandb_url,
        'dataset': args.dataset,
        'configuration': {
            'masking_type': args.type,
            'time_mask_percent': args.time_mask,
            'channel_mask_num': args.channel_mask,
            'alpha': args.alpha,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'epochs': args.ft_epoch,
            'eval_head': args.eval_head,
            'optimizer': args.optimizer,
            'scheduler': args.scheduler,
            'weight_decay': args.weight_decay,
            'dropout_rate': args.dropout_rate,
            'normalize_per_channel': args.normalize_per_channel
        },
        'results': {
            'test_f1_macro': float(test_metrics['f1_macro']),
            'test_accuracy': float(test_metrics['accuracy']),
            'test_f1_weighted': float(test_metrics['f1_weighted']),
            'test_precision': float(test_metrics['precision']),
            'test_recall': float(test_metrics['recall']),
            'per_class_f1': test_metrics['f1_per_class'].tolist()
        }
    }

    # Save JSON report
    report_file = f"experiments/{args.dataset}_experiments_detailed.json"
    os.makedirs("experiments", exist_ok=True)

    if os.path.exists(report_file):
        with open(report_file, 'r') as f:
            all_reports = json.load(f)
    else:
        all_reports = []

    all_reports.append(report)

    with open(report_file, 'w') as f:
        json.dump(all_reports, f, indent=2)

    # Save human-readable report
    txt_file = f"experiments/{args.dataset}_results_detailed.txt"
    with open(txt_file, 'a') as f:
        f.write("\n" + "=" * 70 + "\n")
        f.write(f"Experiment: {report['experiment_name']}\n")
        f.write(f"Timestamp: {report['timestamp']}\n")
        f.write(f"WandB: {wandb_run_name} ({wandb_url})\n")
        f.write("-" * 70 + "\n")
        f.write(f"Configuration:\n")
        f.write(f"  Masking Type: {args.type}\n")
        f.write(f"  Time Mask: {args.time_mask}%\n")
        f.write(f"  Channel Mask: {args.channel_mask} channels\n")
        f.write(f"  Alpha: {args.alpha}\n")
        f.write(f"  Eval Head: {args.eval_head}\n")
        f.write(f"  Optimizer: {args.optimizer} (LR: {args.lr}, WD: {args.weight_decay})\n")
        f.write(f"  Scheduler: {args.scheduler}\n")
        f.write("-" * 70 + "\n")
        f.write(f"Results:\n")
        f.write(f"  F1 Score (macro): {test_metrics['f1_macro']:.4f}\n")
        f.write(f"  Accuracy: {test_metrics['accuracy']:.4f}\n")
        f.write(f"  Precision: {test_metrics['precision']:.4f}\n")
        f.write(f"  Recall: {test_metrics['recall']:.4f}\n")
        f.write("=" * 70 + "\n")

    print(f"\nExperiment report saved to {txt_file}")


if __name__ == '__main__':
    args = parser.parse_args()

    # Set random seeds for reproducibility
    set_random_seeds(42)

    # Initialize WandB
    wandb_run_name = f"{args.dataset}_{args.type}_tm{args.time_mask}_cm{args.channel_mask}_a{args.alpha}_{args.eval_head}_lr{args.lr}_{args.exp_suffix}"
    wandb_url = None

    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=wandb_run_name,
            config=vars(args),
            tags=[args.dataset, args.type, "finetune", args.eval_head, args.optimizer]
        )
        wandb_url = wandb.run.get_url() if wandb.run else ""

    print("=" * 60)
    print("Fine-tuning for Human Activity Recognition")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.type}")
    print(f"Parameters: time_mask={args.time_mask}%, channel_mask={args.channel_mask}, alpha={args.alpha}")
    print(f"Eval Head: {args.eval_head}")
    if wandb_url:
        print(f"WandB: {wandb_url}")

    # Load dataset with new normalization option
    print(f"\nLoading {args.dataset} dataset...")
    x_train, y_train, x_test, y_test = get_data(
        args.dir, args.dataset,
        transformer=True,
        normalize_per_channel=args.normalize_per_channel
    )

    n_outputs = y_train.shape[1]
    print(f"Train shape: {x_train.shape}, Test shape: {x_test.shape}")

    # Load pretrained model
    print(f"\nLoading pretrained model...")
    try:
        divide = None if args.dataset == 'uschad' else 100
        pretrained_model = get_base(
            args.model_dir, args.dataset, args.type,
            args.time_mask, args.channel_mask, args.alpha,
            divide=divide
        )
        print("Pretrained model loaded successfully!")
    except Exception as e:
        print(f"Error loading pretrained model: {e}")
        print("Please run pretraining first!")
        if not args.no_wandb:
            wandb.finish()
        exit(1)

    # Create evaluation model based on specified type
    if args.eval_head == 'simple':
        eval_model = get_evaluate_simple(pretrained_model, n_outputs)
    else:  # complex
        eval_model = get_evaluate_complex(pretrained_model, n_outputs, args.dropout_rate)

    # Fine-tune
    print(f"\nStarting fine-tuning...")
    test_metrics = fine_tune_with_wandb(eval_model, x_train, y_train, x_test, y_test, args)

    # Save detailed report
    save_experiment_report(args, test_metrics, wandb_run_name, wandb_url)

    if not args.no_wandb:
        wandb.finish()