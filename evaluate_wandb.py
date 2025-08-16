import torch
import torch.nn as nn
import numpy as np
import wandb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import argparse
from dataset import get_data
from module import get_base, get_evaluate
from utils import evaluate_record
import time
import seaborn as sns
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Fine-tuning and evaluation with WandB')

# Dataset parameters
parser.add_argument('--dir', type=str, default='datasets/sub', help='dataset path')
parser.add_argument('--dataset', type=str, default='ucihar',
                    choices=['ucihar', 'motion', 'uschad'], help='dataset')
parser.add_argument('--seed', type=int, default=100,
                    help='random seed for dataset division')

# Pretrained model parameters
parser.add_argument('--model_dir', type=str, default='model',
                    help='pretrained model directory')
parser.add_argument('--type', type=str, default='channel',
                    choices=['time', 'spantime', 'spantime_channel', 'time_channel', 'channel'],
                    help='masking strategies')
parser.add_argument('--channel_mask', type=int, default=3,
                    help='number of channel masks')
parser.add_argument('--time_mask', type=int, default=3,
                    help='time mask ratio')
parser.add_argument('--alpha', type=float, default=0.5,
                    help='the hyperparameter alpha')
parser.add_argument('--pretrain_epoch', type=int, default=150,
                    help='pretrain epochs used')

# Fine-tuning parameters
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size for fine-tuning')
parser.add_argument('--ft_epoch', type=int, default=100,
                    help='number of fine-tuning epochs')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate for fine-tuning')
parser.add_argument('--freeze_encoder', action='store_true',
                    help='Freeze encoder weights during fine-tuning')

# WandB parameters
parser.add_argument('--wandb_project', type=str, default='har-masking-finetune',
                    help='WandB project name')
parser.add_argument('--wandb_entity', type=str, default=None,
                    help='WandB entity/team name')
parser.add_argument('--wandb_name', type=str, default=None,
                    help='WandB run name')
parser.add_argument('--no_wandb', action='store_true',
                    help='Disable WandB logging')


def evaluate_model(model, data_loader, device, num_classes):
    """Evaluate model and return metrics"""
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
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')

    # Per-class metrics
    f1_per_class = f1_score(all_labels, all_preds, average=None)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision': precision,
        'recall': recall,
        'f1_per_class': f1_per_class,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels
    }


def plot_confusion_matrix(cm, class_names=None):
    """Create confusion matrix plot"""
    fig, ax = plt.subplots(figsize=(10, 8))

    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(cm))]

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')

    return fig


def fine_tune_with_wandb(model, x_train, y_train, x_val, y_val, x_test, y_test, args):
    """Fine-tune model with WandB logging"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Optionally freeze encoder
    if args.freeze_encoder:
        for param in model[0].parameters():  # model[0] is the encoder
            param.requires_grad = False
        print("Encoder weights frozen")

    # Convert to tensors
    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).float()
    x_val = torch.from_numpy(x_val).float()
    y_val = torch.from_numpy(y_val).float()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).float()

    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = torch.utils.data.TensorDataset(x_val, y_val)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)

    num_classes = y_train.shape[1]
    best_val_f1 = 0
    best_model_state = None
    best_epoch = 0

    print(f"Training on {device}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Training loop
    for epoch in range(args.ft_epoch):
        epoch_start = time.time()

        # Training phase
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

            train_loss += loss.item()
            train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        # Training metrics
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='macro')
        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        val_metrics = evaluate_model(model, val_loader, device, num_classes)

        epoch_time = time.time() - epoch_start

        # Update learning rate
        scheduler.step(val_metrics['f1_macro'])

        print(f'Epoch [{epoch + 1}/{args.ft_epoch}] ({epoch_time:.1f}s)')
        print(f'  Train - Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}')
        print(f'  Val   - Acc: {val_metrics["accuracy"]:.4f}, F1: {val_metrics["f1_macro"]:.4f}')

        # Log to WandB
        if not args.no_wandb:
            log_dict = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'train_acc': train_acc,
                'train_f1': train_f1,
                'val_acc': val_metrics['accuracy'],
                'val_f1_macro': val_metrics['f1_macro'],
                'val_f1_weighted': val_metrics['f1_weighted'],
                'val_precision': val_metrics['precision'],
                'val_recall': val_metrics['recall'],
                'learning_rate': optimizer.param_groups[0]['lr'],
                'epoch_time': epoch_time
            }

            # Log per-class F1 scores
            for i, f1 in enumerate(val_metrics['f1_per_class']):
                log_dict[f'val_f1_class_{i}'] = f1

            wandb.log(log_dict)

        # Save best model
        if val_metrics['f1_macro'] > best_val_f1:
            best_val_f1 = val_metrics['f1_macro']
            best_model_state = model.state_dict().copy()
            best_epoch = epoch + 1
            print(f'  -> New best val F1: {best_val_f1:.4f}')

            if not args.no_wandb:
                wandb.log({
                    'best_val_f1': best_val_f1,
                    'best_epoch': best_epoch
                })

    # Load best model for final testing
    print("\n" + "=" * 50)
    print(f"Loading best model from epoch {best_epoch}")
    model.load_state_dict(best_model_state)

    # Final test evaluation
    test_metrics = evaluate_model(model, test_loader, device, num_classes)

    print("\nFinal Test Results:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  F1 (macro): {test_metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted): {test_metrics['f1_weighted']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")

    # Log test results to WandB
    if not args.no_wandb:
        # Log final test metrics
        wandb.log({
            'test_accuracy': test_metrics['accuracy'],
            'test_f1_macro': test_metrics['f1_macro'],
            'test_f1_weighted': test_metrics['f1_weighted'],
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall']
        })

        # Log per-class test F1
        for i, f1 in enumerate(test_metrics['f1_per_class']):
            wandb.log({f'test_f1_class_{i}': f1})

        # Create and log confusion matrix
        class_names = None
        if args.dataset == 'ucihar':
            class_names = ['Walking', 'Upstairs', 'Downstairs', 'Sitting', 'Standing', 'Laying']

        fig = plot_confusion_matrix(test_metrics['confusion_matrix'], class_names)
        wandb.log({"confusion_matrix": wandb.Image(fig)})
        plt.close(fig)

        # Create summary table
        summary_table = wandb.Table(
            columns=["Metric", "Value"],
            data=[
                ["Test Accuracy", test_metrics['accuracy']],
                ["Test F1 (macro)", test_metrics['f1_macro']],
                ["Test F1 (weighted)", test_metrics['f1_weighted']],
                ["Test Precision", test_metrics['precision']],
                ["Test Recall", test_metrics['recall']],
                ["Best Val F1", best_val_f1],
                ["Best Epoch", best_epoch]
            ]
        )
        wandb.log({"summary_table": summary_table})

    return test_metrics


if __name__ == '__main__':
    args = parser.parse_args()

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Initialize WandB
    if not args.no_wandb:
        if args.wandb_name is None:
            args.wandb_name = f"ft_{args.dataset}_{args.type}_tm{args.time_mask}_cm{args.channel_mask}_a{args.alpha}_s{args.seed}"

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config=vars(args),
            tags=[args.dataset, args.type, "finetune", f"seed{args.seed}"]
        )

    print("=" * 60)
    print("Fine-tuning for Human Activity Recognition")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Pretrained model: {args.type}")
    if args.type in ['time', 'spantime', 'time_channel', 'spantime_channel']:
        print(f"  Time mask: {args.time_mask}%")
    if args.type in ['channel', 'time_channel', 'spantime_channel']:
        print(f"  Channel mask: {args.channel_mask}")
    if args.type in ['time_channel', 'spantime_channel']:
        print(f"  Alpha: {args.alpha}")
    print("=" * 60)

    # Load dataset
    print(f"\nLoading {args.dataset} dataset...")
    x_train, y_train, x_val, y_val, x_test, y_test = get_data(
        args.dir, args.dataset, transformer=True, divide_seed=args.seed
    )

    n_timesteps = x_train.shape[1]
    n_features = x_train.shape[2]
    n_outputs = y_train.shape[1]

    print(f"Dataset shapes:")
    print(f"  Train: {x_train.shape}, {y_train.shape}")
    print(f"  Val:   {x_val.shape}, {y_val.shape}")
    print(f"  Test:  {x_test.shape}, {y_test.shape}")

    # Load pretrained model
    print(f"\nLoading pretrained model...")
    try:
        pretrained_model = get_base(
            args.model_dir, args.dataset, args.type,
            args.time_mask, args.channel_mask, args.alpha,
            divide=args.seed if args.dataset != 'uschad' else None,
            epoch=args.pretrain_epoch if args.pretrain_epoch != 150 else None
        )
        print("Pretrained model loaded successfully!")
    except Exception as e:
        print(f"Error loading pretrained model: {e}")
        print("Make sure you have run the pretraining script first.")
        if not args.no_wandb:
            wandb.finish()
        exit(1)

    # Create evaluation model
    eval_model = get_evaluate(pretrained_model, n_outputs)

    # Count parameters
    total_params = sum(p.numel() for p in eval_model.parameters())
    trainable_params = sum(p.numel() for p in eval_model.parameters() if p.requires_grad)
    print(f"Model parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    # Start fine-tuning
    print(f"\nStarting fine-tuning...")
    print(f"  Epochs: {args.ft_epoch}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Freeze encoder: {args.freeze_encoder}")
    print("=" * 60)

    test_metrics = fine_tune_with_wandb(
        eval_model, x_train, y_train, x_val, y_val, x_test, y_test, args
    )

    # Record results
    evaluate_record(
        args.dataset, args.type, args.time_mask, args.channel_mask,
        args.alpha, None, test_metrics['f1_macro'],
        divide=args.seed if args.dataset != 'uschad' else None,
        epoch=args.pretrain_epoch if args.pretrain_epoch != 150 else None
    )

    print(f"\nResults saved to {args.dataset}_record.txt")
    print("=" * 60)

    if not args.no_wandb:
        wandb.finish()