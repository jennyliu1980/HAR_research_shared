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
import os


def run_single_experiment(dataset, masking_type, time_mask, channel_mask, alpha, seed,
                          pretrain_epochs=300, finetune_epochs=200, use_wandb=True, skip_pretrain=False):
    """Run a single experiment (pretrain + finetune)"""

    experiment_name = f"{dataset}_{masking_type}_tm{time_mask}_cm{channel_mask}_a{alpha}_s{seed}"
    print(f"\n{'=' * 60}")
    print(f"Running experiment: {experiment_name}")
    print(f"{'=' * 60}")

    # Check if pretrained model already exists
    model_dir = f"model/{dataset}/"
    if masking_type == 'spantime_channel':
        model_file = f"spantime{time_mask}_channel{channel_mask}_divide{seed}_alpha{alpha}"
    elif masking_type == 'time_channel':
        model_file = f"time{time_mask}_channel{channel_mask}_divide{seed}_alpha{alpha}"
    elif masking_type == 'spantime':
        model_file = f"spantime{time_mask}_divide{seed}"
    elif masking_type == 'time':
        model_file = f"time{time_mask}_divide{seed}"
    elif masking_type == 'channel':
        model_file = f"channel{channel_mask}_divide{seed}"
    else:
        model_file = f"{masking_type}_{time_mask}_{channel_mask}_{alpha}_{seed}"

    model_path = os.path.join(model_dir, model_file)

    # Step 1: Pretraining (skip if model exists or skip_pretrain is True)
    model_exists = os.path.exists(model_path)

    if model_exists:
        print(f"✅ Pretrained model found at: {model_path}")
        print("  → Skipping pretraining, going directly to fine-tuning...")
    elif skip_pretrain:
        print(f"⚠️ Model not found but skip_pretrain=True")
        print(f"  Expected model path: {model_path}")
        return None
    else:
        print(f"⚠️ Model not found at: {model_path}")
        print("  → Starting pretraining...")

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

        print(f"Command: {' '.join(pretrain_cmd)}")

        try:
            result = subprocess.run(pretrain_cmd, capture_output=True, text=True, check=True)
            print("✅ Pretraining completed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Pretraining failed with error: {e}")
            print(f"Error output: {e.stderr[-2000:]}")  # Last 2000 chars
            return None

    # Step 2: Fine-tuning
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
        print("✅ Fine-tuning completed successfully!")

        # Parse results from output
        output_lines = result.stdout.split('\n')
        metrics = {}

        # Look for test results
        for i, line in enumerate(output_lines):
            if "Final Test Results:" in line or "Test Results:" in line:
                # Extract metrics from the following lines
                for j in range(1, 10):  # Check next 10 lines
                    if i + j < len(output_lines):
                        metric_line = output_lines[i + j]
                        if "Accuracy:" in metric_line:
                            try:
                                metrics['accuracy'] = float(metric_line.split(':')[-1].strip())
                            except:
                                pass
                        elif "F1 (macro):" in metric_line or "F1 Score:" in metric_line:
                            try:
                                metrics['f1_macro'] = float(metric_line.split(':')[-1].strip())
                            except:
                                pass
                        elif "F1 (weighted):" in metric_line:
                            try:
                                metrics['f1_weighted'] = float(metric_line.split(':')[-1].strip())
                            except:
                                pass

        if metrics:
            print(f"\n📊 Results: F1={metrics.get('f1_macro', 'N/A'):.4f}, Acc={metrics.get('accuracy', 'N/A'):.4f}")
        else:
            print("⚠️ Warning: Could not parse metrics from output")

        return metrics

    except subprocess.CalledProcessError as e:
        print(f"❌ Fine-tuning failed with error: {e}")
        print(f"Error output: {e.stderr[-1000:]}")  # Last 1000 chars
        return None

    return None


def run_existing_models_evaluation(dataset='ucihar', seed=100, use_wandb=True):
    """Evaluate all existing pretrained models without retraining"""

    print("\n" + "=" * 80)
    print("EVALUATING EXISTING MODELS")
    print("=" * 80)

    # Check what models exist
    model_dir = f"model/{dataset}/"
    if not os.path.exists(model_dir):
        print(f"❌ Model directory {model_dir} does not exist")
        return []

    existing_models = os.listdir(model_dir)
    print(f"Found {len(existing_models)} pretrained models:")
    for model in existing_models:
        print(f"  - {model}")

    results = []

    # Define configurations based on existing models
    configs_to_test = []

    # Parse existing model names to determine configurations
    for model_name in existing_models:
        config = None

        # Parse different model name patterns
        if model_name.startswith('spantime') and 'channel' in model_name:
            # spantime30_channel3_divide100_alpha0.5
            parts = model_name.split('_')
            time_mask = int(parts[0].replace('spantime', ''))
            channel_mask = int(parts[1].replace('channel', ''))
            alpha = float(parts[3].replace('alpha', ''))
            config = {'type': 'spantime_channel', 'time_mask': time_mask,
                      'channel_mask': channel_mask, 'alpha': alpha}

        elif model_name.startswith('spantime'):
            # spantime30_divide100
            time_mask = int(model_name.split('_')[0].replace('spantime', ''))
            config = {'type': 'spantime', 'time_mask': time_mask,
                      'channel_mask': 0, 'alpha': 0}

        elif model_name.startswith('time') and 'channel' in model_name:
            # time30_channel3_divide100_alpha0.5
            parts = model_name.split('_')
            time_mask = int(parts[0].replace('time', ''))
            channel_mask = int(parts[1].replace('channel', ''))
            alpha = float(parts[3].replace('alpha', ''))
            config = {'type': 'time_channel', 'time_mask': time_mask,
                      'channel_mask': channel_mask, 'alpha': alpha}

        elif model_name.startswith('time'):
            # time30_divide100
            time_mask = int(model_name.split('_')[0].replace('time', ''))
            config = {'type': 'time', 'time_mask': time_mask,
                      'channel_mask': 0, 'alpha': 0}

        elif model_name.startswith('channel'):
            # channel3_divide100
            channel_mask = int(model_name.split('_')[0].replace('channel', ''))
            config = {'type': 'channel', 'time_mask': 0,
                      'channel_mask': channel_mask, 'alpha': 0}

        if config and config not in configs_to_test:
            configs_to_test.append(config)

    print(f"\nWill evaluate {len(configs_to_test)} configurations:")
    for config in configs_to_test:
        print(
            f"  - {config['type']}: time={config['time_mask']}%, channel={config['channel_mask']}, alpha={config['alpha']}")

    # Run evaluation for each configuration
    for config in configs_to_test:
        print(f"\n{'=' * 60}")
        print(
            f"Evaluating: {config['type']} (time={config['time_mask']}, channel={config['channel_mask']}, alpha={config['alpha']})")

        metrics = run_single_experiment(
            dataset=dataset,
            masking_type=config['type'],
            time_mask=config['time_mask'],
            channel_mask=config['channel_mask'],
            alpha=config['alpha'],
            seed=seed,
            use_wandb=use_wandb,
            skip_pretrain=True  # Skip pretraining since model exists
        )

        if metrics:
            result = {
                'dataset': dataset,
                'masking_type': config['type'],
                'time_mask': config['time_mask'],
                'channel_mask': config['channel_mask'],
                'alpha': config['alpha'],
                'seed': seed,
                **metrics
            }
            results.append(result)

            # Save intermediate results
            df = pd.DataFrame(results)
            df.to_csv(f'results_{dataset}_existing_models.csv', index=False)
            print(f"✅ Saved to results_{dataset}_existing_models.csv")

    return results


def run_all_experiments(dataset='ucihar', seeds=[100], use_wandb=True):
    """Run ALL experiments from the paper"""

    # Define all experiment configurations based on the paper
    experiments = [
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
        {'type': 'spantime_channel', 'time_mask': 30, 'channel_mask': 3, 'alpha': 0.5},  # BEST
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


def run_quick_experiment(dataset='ucihar', seed=100, use_wandb=True):
    """Run ONLY the best configuration for quick testing"""

    print("\n" + "=" * 80)
    print("QUICK TEST MODE - Running best configuration only")
    print("=" * 80)
    print("Configuration: spantime_channel, time_mask=30%, channel_mask=3, alpha=0.5")
    print("Expected F1 (paper): 0.931")
    print("=" * 80)

    # Run only the best configuration
    metrics = run_single_experiment(
        dataset=dataset,
        masking_type='spantime_channel',
        time_mask=30,
        channel_mask=3,
        alpha=0.5,
        seed=seed,
        use_wandb=use_wandb
    )

    if metrics:
        results = [{
            'dataset': dataset,
            'masking_type': 'spantime_channel',
            'time_mask': 30,
            'channel_mask': 3,
            'alpha': 0.5,
            'seed': seed,
            **metrics
        }]

        # Save results
        df = pd.DataFrame(results)
        df.to_csv(f'results_{dataset}_quick.csv', index=False)

        # Print summary
        print("\n" + "=" * 80)
        print("QUICK TEST RESULTS:")
        print("=" * 80)
        print(f"F1 Score (macro): {metrics.get('f1_macro', 'N/A'):.4f}")
        print(f"Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
        print(f"Expected F1: 0.931 ± 0.02")

        f1 = metrics.get('f1_macro', 0)
        if abs(f1 - 0.931) < 0.02:
            print("✅ SUCCESS: Result matches paper!")
        else:
            print(f"⚠️ Difference from paper: {f1 - 0.931:+.4f}")
        print("=" * 80)

        return results
    else:
        print("❌ Quick test failed")
        return []


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
    parser = argparse.ArgumentParser(description='Run HAR experiments')
    parser.add_argument('--dataset', type=str, default='ucihar',
                        choices=['ucihar', 'motion', 'uschad'],
                        help='Dataset to use')
    parser.add_argument('--seeds', type=int, nargs='+', default=[100],
                        help='Random seeds for experiments (default: just 100)')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick test with ONLY the best configuration')
    parser.add_argument('--eval_existing', action='store_true',
                        help='Only evaluate existing models without pretraining')
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
    elif args.eval_existing:
        # Evaluate only existing models
        print("Mode: Evaluate existing models only")
        results = run_existing_models_evaluation(
            dataset=args.dataset,
            seed=args.seeds[0],  # Use first seed only
            use_wandb=not args.no_wandb
        )

        if results:
            df = pd.DataFrame(results)
            filename = f'results_{args.dataset}_existing.csv'
            df.to_csv(filename, index=False)
            print(f"\n✅ Final results saved to {filename}")

            # Show summary
            print("\n" + "=" * 80)
            print("EVALUATION SUMMARY")
            print("=" * 80)
            for _, row in df.iterrows():
                print(
                    f"{row['masking_type']:<20} (tm={row['time_mask']:>2}, cm={row['channel_mask']}, α={row['alpha']:.1f}): "
                    f"F1={row.get('f1_macro', 0):.4f}, Acc={row.get('accuracy', 0):.4f}")

            # Compare with paper
            compare_with_paper()
    elif args.quick:
        # Quick mode: only run best configuration
        results = run_quick_experiment(
            dataset=args.dataset,
            seed=args.seeds[0],
            use_wandb=not args.no_wandb
        )
    else:
        # Run all experiments (with multiple seeds if specified)
        if len(args.seeds) > 1:
            print(f"⚠️ Running with {len(args.seeds)} seeds will take ~{18 * len(args.seeds)} hours!")
            response = input("Continue? (y/n): ")
            if response.lower() != 'y':
                print("Aborted.")
                exit(0)

        results = run_all_experiments(
            dataset=args.dataset,
            seeds=args.seeds,
            use_wandb=not args.no_wandb
        )

        if results:
            df = pd.DataFrame(results)
            filename = f'results_{args.dataset}_final.csv'
            df.to_csv(filename, index=False)
            print(f"\nResults saved to {filename}")
            analyze_results(df)
            compare_with_paper()
        else:
            print("\nNo results to analyze")