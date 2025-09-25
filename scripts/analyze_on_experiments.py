#!/usr/bin/env python3
"""
Script to analyze results from ON Q5 experiments.
Calculates mean and standard deviation for each experiment configuration.
"""

import os
import re
import glob
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def extract_metrics_from_tensorboard(log_dir: str, run_name: str) -> Dict[str, List[float]]:
    """
    Extract metrics from TensorBoard log files for a specific run.
    
    Args:
        log_dir: Directory containing TensorBoard logs
        run_name: Name of the run to extract metrics from
        
    Returns:
        Dictionary mapping metric names to lists of values
    """
    run_path = os.path.join(log_dir, run_name)
    if not os.path.exists(run_path):
        print(f"Warning: Run directory not found: {run_path}")
        return {}
    
    # Find all version directories
    version_dirs = glob.glob(os.path.join(run_path, "version_*"))
    # print(version_dirs)
    if not version_dirs:
        print(f"Warning: No version directories found in {run_path}")
        return {}
    
    metrics = {}
    
    for version_dir in version_dirs:
        event_files = glob.glob(os.path.join(version_dir, "events.out.tfevents.*"))
        if not event_files:
            continue
            
        # Use the first event file found
        # event_file = event_files[-1]
        
        for event_file in event_files:
            try:
                ea = EventAccumulator(event_file)
                ea.Reload()
                
                # Get all scalar tags
                scalar_tags = ea.Tags()['scalars']
                
                for tag in scalar_tags:
                    scalar_events = ea.Scalars(tag)
                    values = [event.value for event in scalar_events]
                    
                    if tag not in metrics:
                        metrics[tag] = []
                    metrics[tag].extend(values)
                    
            except Exception as e:
                print(f"Error reading {event_file}: {e}")
                continue
    
    return metrics


def parse_run_name(run_name: str) -> Dict[str, Any]:
    """
    Parse run name to extract experiment parameters.
    
    Expected format: on_regression_on_glg_nsm{num_samples}_nst{max_steps}_bs{batch_size}
    """
    pattern = r'on_regression_on_glg_nsm(\d+)_nst(\d+)_bs(\d+)'
    match = re.match(pattern, run_name)
    
    if match:
        return {
            'num_samples': int(match.group(1)),
            'max_steps': int(match.group(2)),
            'batch_size': int(match.group(3))
        }
    return {}


def find_experiment_runs(log_dir: str) -> Dict[str, List[str]]:
    """
    Find all runs for each experiment configuration.
    
    Returns:
        Dictionary mapping experiment config to list of run names
    """
    experiments = {}
    
    # Look for all run directories
    all_runs = glob.glob(os.path.join(log_dir, "on_regression_on_glg_*"))
    print(all_runs)
    
    for run_path in all_runs:
        run_name = os.path.basename(run_path)
        # exp_runs = glob.glob(os.path.join(run_path, "version_*"))
        params = parse_run_name(run_name)
        
        if not params:
            continue
            
        # Create experiment key
        exp_key = f"nsm{params['num_samples']}_nst{params['max_steps']}_bs{params['batch_size']}"
        
        if exp_key not in experiments:
            experiments[exp_key] = []
        experiments[exp_key].append(run_name)

        # experiments[exp_key] = exp_runs

    return experiments


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """Calculate mean, std, min, max for a list of values."""
    if not values:
        return {'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan, 'count': 0}
    
    values_array = np.array(values)
    return {
        'mean': np.mean(values_array),
        'std': np.std(values_array, ddof=1),  # Sample standard deviation
        'min': np.min(values_array),
        'max': np.max(values_array),
        'count': len(values_array)
    }


def analyze_experiments(log_dir: str = "lightning_logs", output_file: str = None):
    """
    Analyze all ON Q5 experiments and calculate statistics.
    
    Args:
        log_dir: Directory containing TensorBoard logs
        output_file: Optional file to save results as CSV
    """
    print("Analyzing ON Q5 experiments...")
    print(f"Looking for logs in: {log_dir}")
    
    # Find all experiment runs
    experiments = find_experiment_runs(log_dir)

    # print(experiments)
    
    if not experiments:
        print("No experiment runs found!")
        return
    
    print(f"Found {len(experiments)} experiment configurations:")
    for exp_key, runs in experiments.items():
        print(f"  {exp_key}: {len(runs)} runs")
    
    # Define the experiments from the shell script
    expected_experiments = [
        ("nsm30_nst1001_bs15", "Experiment 1: 30 samples, 1001 steps, batch_size 15"),
        ("nsm300_nst1001_bs15", "Experiment 2: 300 samples, 1001 steps, batch_size 15"),
        ("nsm3000_nst301_bs15", "Experiment 3: 3000 samples, 301 steps, batch_size 15"),
        ("nsm30000_nst3001_bs15", "Experiment 4: 30000 samples, 3001 steps, batch_size 15"),
    ]
    
    results = []
    
    for exp_key, exp_description in expected_experiments:
        print(f"\n{exp_description}")
        print("=" * 60)
        
        if exp_key not in experiments:
            print(f"Warning: No runs found for {exp_key}")
            continue
        
        runs = experiments[exp_key]
        print(f"Found {len(runs)} runs for this experiment")
        
        # Collect all metrics from all runs
        all_metrics = {}
        
        for run_name in runs:
            print(f"  Processing {run_name}...")
            run_metrics = extract_metrics_from_tensorboard(log_dir, run_name)
            
            for metric_name, values in run_metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].extend(values)

        # print(all_metrics)
        
        # Calculate statistics for each metric
        for metric_name, values in all_metrics.items():
            if not values:
                continue
                
            stats = calculate_statistics(values)
            
            # Get final values (last epoch/step) for each run
            final_values = []
            for run_name in runs:
                # print(log_dir)
                run_metrics = extract_metrics_from_tensorboard(log_dir, run_name)
                # print(run_metrics)
                if metric_name in run_metrics and run_metrics[metric_name]:
                    final_values.append(run_metrics[metric_name][-1])  # Last value
            
            # print(final_values)
            final_stats = calculate_statistics(final_values)
            
            result = {
                'experiment': exp_key,
                'description': exp_description,
                'metric': metric_name,
                'all_values_mean': stats['mean'],
                'all_values_std': stats['std'],
                'all_values_min': stats['min'],
                'all_values_max': stats['max'],
                'all_values_count': stats['count'],
                'final_values_mean': final_stats['mean'],
                'final_values_std': final_stats['std'],
                'final_values_min': final_stats['min'],
                'final_values_max': final_stats['max'],
                'final_values_count': final_stats['count']
            }
            results.append(result)
            
            # Print key metrics
            if 'test/loss' in metric_name.lower() or 'test/accuracy' in metric_name.lower():
                print(f"    {metric_name}:")
                print(f"      All values - Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}")
                # print(f"      Final values - Mean: {final_stats['mean']:.6f}, Std: {final_stats['std']:.6f}")
    
    # Convert to DataFrame and save
    # if results:
    #     df = pd.DataFrame(results)
        
    #     if output_file:
    #         df.to_csv(output_file, index=False)
    #         print(f"\nResults saved to: {output_file}")
        
    #     # Print summary table for key metrics
    #     print("\n" + "="*80)
    #     print("SUMMARY - Final Values (Mean ± Std)")
    #     print("="*80)
        
    #     key_metrics = ['val/loss', 'val/accuracy', 'train/loss', 'test/loss', 'test/accuracy']
        
    #     for metric in key_metrics:
    #         metric_results = df[df['metric'] == metric]
    #         if not metric_results.empty:
    #             print(f"\n{metric}:")
    #             for _, row in metric_results.iterrows():
    #                 print(f"  {row['description']}")
    #                 print(f"    Mean: {row['final_values_mean']:.6f} ± {row['final_values_std']:.6f}")
    # else:
    #     print("No results found!")


def main():
    parser = argparse.ArgumentParser(description="Analyze ON Q5 experiment results")
    parser.add_argument("--log_dir", type=str, default="lightning_logs",
                       help="Directory containing TensorBoard logs")
    parser.add_argument("--output", type=str, default="on_q5_results.csv",
                       help="Output CSV file for results")
    
    args = parser.parse_args()
    
    analyze_experiments(args.log_dir, args.output)


if __name__ == "__main__":
    main()
