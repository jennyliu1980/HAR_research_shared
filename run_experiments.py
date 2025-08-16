#!/usr/bin/env python3
"""
Complete experiment pipeline for reproducing the paper:
"An Improved Masking Strategy for Self-supervised Masked Reconstruction in HAR"

This script runs all experiments with different masking strategies and records results.
"""

import subprocess
import argparse
import json
import time
from pathlib import Path
import pandas as pd


def run_single_experiment(dataset, masking_type, time_mask, channel_mask, alpha, seed,
                          pretrain_epochs=150, finetune_epochs=100, use_wandb=True):
    """Run a single experiment (pretrain + finetune)"""

    experiment_name = f"{dataset}_{masking_type}_tm{time_mask}_cm{channel_mask}_a{alpha}_s{seed}"
    print(f"\n{'=' * 60}")
    print(f"Running experiment: {experiment_name}")
    print(f"{'=' * 60}")

    # Pretraining command
    pretrain_cmd = [
        "python", "main_wandb.py",
        "--dataset", dataset,
        "--type", masking_type,
        "--time_mask", str(time_mask),
        "--channel_mask", str(channel_mask),
        "--alpha", str(alpha),
        "--seed", str(seed),
        "--epoch", str(pretrain_epochs)
    ]

    if not use_wandb:
        pretrain_cmd.append("--no_wandb")

    print(f"Step 1: Pretraining...")
    print(f"Command: {' '.join(pretrain_cmd)}")

    try:
        result = subprocess.run(pretrain_cmd, capture_output=True, text=True, check=True)
        print("Pretraining completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Pretraining failed with error: {e}")
        print(f"Error output: {e.stderr}")
        return None

    # Fine-tuning command
    finetune_cmd = [
        "python", "evaluate_wandb.py",
        "--dataset", dataset,
        "--type", masking_type,
        "--time_mask", str(time_mask),
        "--channel_mask", str(channel_mask),
        "--alpha", str(alpha),
        "--seed", str(seed),
        "--ft_epoch", str(finetune_epochs)
    ]

    if not use_wandb:
        finetune_cmd.append("--no_wandb")

    print(f"\nStep 2: Fine-tuning...")
    print(f"Command: {' '.join(finetune_cmd)}")

    try:
        result = subprocess.run(finetune_cmd, capture_output=True, text=True, check=True)
        print("Fine-tuning completed successfully!")

        # Parse results from output
        output_lines = result.stdout.split('\n')
        for line in output_lines:
            if "Test Results:" in line:
                idx = output_lines.index(line)
                # Extract metrics from the following lines
                metrics = {}
                for i in range(1, 6):
                    if idx + i < len(output_lines):
                        metric_line = output_lines[idx + i]
                        if "Accuracy:" in metric_line:
                            metrics['accuracy'] = float(metric_line.split(':')[1].strip())
                        elif "F1 (macro):" in metric_line:
                            metrics['f1_macro'] = float(metric_line.split(':')[1].strip())
                        elif "F1 (weighted):" in metric_line:
                            metrics['f1_weighted'] = float(metric_line.split(':')[1].strip())

                return metrics

    except subprocess.CalledProcessError as e:
        print(f"Fine-tuning failed with error: {e}")
        print(f"Error output: {e.stderr}")
        return None

    return None


def run_all_experiments(dataset='ucihar', seeds=[100, 200, 300], use_wandb=True):
    """Run all experiments from the paper"""

    # Define all experiment configurations based on the paper
    experiments = [
        # Baseline: No masking (train from scratch)
        # This would need a separate script without pretraining

        # Time masking only
        {'type': 'time', 'time_mask': 10, 'channel_mask': 0, 'alpha': 0},
        {'type': 'time', 'time_mask': 20, 'channel_mask': 0, 'alpha': 0},
        {'type': 'time', 'time_mask': 30, 'channel_mask': 0, 'alpha': 0},
        {'type': 'time', 'time_mask': 40, 'channel_mask': 0, 'alpha': 0},

        # Span-time masking only
        {'type': 'spantime', 'time_mask': 10, 'channel_mask': 0, 'alpha': 0},
        {'type': 'spantime', 'time_mask': 20, 'channel_mask': 0, 'alpha': 0},
        {'type': 'spantime', 'time_mask': 30, 'channel_mask': 0, 'alpha': 0},
        {'type': 'spantime', 'time_mask': 40, 'channel_mask': 0, 'alpha': 0},

        # Channel masking only
        {'type': 'channel', 'time_mask': 0, 'channel_mask': 1, 'alpha': 0},
        {'type': 'channel', 'time_mask': 0, 'channel_mask': 2, 'alpha': 0},
        {'type': 'channel', 'time_mask': 0, 'channel_mask': 3, 'alpha': 0},
        {'type': 'channel', 'time_mask': 0, 'channel_mask': 4, 'alpha': 0},

        # Combined masking (time + channel)
        {'type': 'time_channel', 'time_mask': 30, 'channel_mask': 3, 'alpha': 0.3},
        {'type': 'time_channel', 'time_mask': 30, 'channel_mask': 3, 'alpha': 0.5},
        {'type': 'time_channel', 'time_mask': 30, 'channel_mask': 3, 'alpha': 0.7},

        # Combined masking (spantime + channel) - Paper's best
        {'type': 'spantime_channel', 'time_mask': 30, 'channel_mask': 3, 'alpha': 0.3},
        {'type': 'spantime_channel', 'time_mask': 30, 'channel_mask': 3, 'alpha': 0.5},
        {'type': 'spantime_channel', 'time_mask': 30, 'channel_mask': 3, 'alpha': 0.7},
    ]

    results = []

    for exp in experiments:
        for seed in seeds:
            print(f"\n{'=' * 80}")
            print(f"Experiment: {exp['type']} | Time: {exp['time_mask']}% | "
                  f"Channel: {exp['channel_mask']} | Alpha: {exp['alpha']} | Seed: {seed}")
            print(f"{'=' * 80}")

            metrics = run_single_experiment(
                dataset=dataset,
                masking_type=exp['type'],
                time_mask=exp['time_mask'],
                channel_mask=exp['channel_mask'],
                alpha=exp['alpha'],
                seed=seed,
                use_wandb=use_wandb
            )

            if metrics:
                result = {
                    'dataset': dataset,
                    'masking_type': exp['type'],
                    'time_mask': exp['time_mask'],
                    'channel_mask': exp['channel_mask'],
                    'alpha': exp['alpha'],
                    'seed': seed,
                    **metrics
                }
                results.append(result)

                # Save intermediate results
                df = pd.DataFrame(results)
                df.to_csv(f'results_{dataset}_intermediate.csv', index=False)

            # Small delay between experiments
            time.sleep(5)

    return results


def analyze_results(results_df):
    """Analyze and summarize experiment results"""

    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)

    # Group by masking type and compute statistics
    grouped = results_df.groupby(['masking_type', 'time_mask', 'channel_mask', 'alpha'])

    summary = grouped.agg({
        'f1_macro': ['mean', 'std'],
        'accuracy': ['mean', 'std']
    }).round(4)

    print("\nResults by Masking Strategy:")
    print(summary)

    # Find best configuration
    best_idx = results_df.groupby(['masking_type', 'time_mask', 'channel_mask', 'alpha'])['f1_macro'].mean().idxmax()
    best_config = results_df[
        (results_df['masking_type'] == best_idx[0]) &
        (results_df['time_mask'] == best_idx[1]) &
        (results_df['channel_mask'] == best_idx[2]) &
        (results_df['alpha'] == best_idx[3])
        ]

    print("\n" + "=" * 80)
    print("BEST CONFIGURATION:")
    print(f"  Type: {best_idx[0]}")
    print(f"  Time Mask: {best_idx[1]}%")
    print(f"  Channel Mask: {best_idx[2]}")
    print(f"  Alpha: {best_idx[3]}")
    print(f"  Mean F1 Score: {best_config['f1_macro'].mean():.4f} ± {best_config['f1_macro'].std():.4f}")
    print(f"  Mean Accuracy: {best_config['accuracy'].mean():.4f} ± {best_config['accuracy'].std():.4f}")
    print("=" * 80)

    return summary


def compare_with_paper():
    """Compare results with paper's reported performance"""

    paper_results = {
        'UCI-HAR': {
            'No pretraining': 0.892,
            'Time (30%)': 0.914,
            'Span-time (30%)': 0.920,
            'Channel (3)': 0.908,
            'Time+Channel (30%, 3, α=0.5)': 0.924,
            'Spantime+Channel (30%, 3, α=0.5)': 0.931  # Best
        },
        'MotionSense': {
            'No pretraining': 0.876,
            'Spantime+Channel (30%, 3, α=0.5)': 0.918
        },
        'USC-HAD': {
            'No pretraining': 0.823,
            'Spantime+Channel (30%, 3, α=0.5)': 0.871
        }
    }

    print("\n" + "=" * 80)
    print("COMPARISON WITH PAPER RESULTS")
    print("=" * 80)
    print("\nPaper's reported F1 scores (macro):")

    for dataset, results in paper_results.items():
        print(f"\n{dataset}:")
        for method, score in results.items():
            print(f"  {method}: {score:.3f}")

    print("\n" + "=" * 80)
    print("Your results should be within ±0.02 of these values")
    print("Small differences are expected due to:")
    print("  - Random initialization")
    print("  - Hardware differences")
    print("  - Minor implementation details")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run all HAR experiments')
    parser.add_argument('--dataset', type=str, default='ucihar',
                        choices=['ucihar', 'motion', 'uschad'],
                        help='Dataset to use')
    parser.add_argument('--seeds', type=int, nargs='+', default=[100, 200, 300],
                        help='Random seeds for experiments')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick test with fewer experiments')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable WandB logging')
    parser.add_argument('--analyze_only', type=str, default=None,
                        help='Path to existing results CSV to analyze')

    args = parser.parse_args()

    if args.analyze_only:
        # Just analyze existing results
        df = pd.read_csv(args.analyze_only)
        analyze_results(df)
        compare_with_paper()
    else:
        # Run experiments
        if args.quick:
            # Quick test: only run best configuration
            experiments = [
                {'type': 'spantime_channel', 'time_mask': 30, 'channel_mask': 3, 'alpha': 0.5}
            ]
            seeds = [100]
        else:
            seeds = args.seeds

        results = run_all_experiments(
            dataset=args.dataset,
            seeds=seeds,
            use_wandb=not args.no_wandb
        )

        if results:
            # Save final results
            df = pd.DataFrame(results)
            filename = f'results_{args.dataset}_final.csv'
            df.to_csv(filename, index=False)
            print(f"\nResults saved to {filename}")

            # Analyze results
            analyze_results(df)
            compare_with_paper()
        else:
            print("\nNo results to analyze")