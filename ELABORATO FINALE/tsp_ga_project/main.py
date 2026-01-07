#!/usr/bin/env python3
"""
Main Script for TSP Genetic Algorithm Thesis Experiments
Generates all data and figures required for Chapter 5

Usage:
    python main.py --all              # Run all experiments
    python main.py --standard         # Run standard experiments only
    python main.py --sensitivity      # Run sensitivity analyses only
    python main.py --visualize        # Generate figures only
    python main.py --download         # Download TSPLIB instances
"""

import argparse
import sys
import os
import time
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tsp_loader import download_tsplib_instances, load_all_instances
from experiment_runner import BatchExperimentRunner, ExperimentRunner
from visualization import ThesisVisualizer, generate_tables


def print_header():
    """Print program header"""
    print("\n" + "="*70)
    print(" TSP GENETIC ALGORITHM - THESIS EXPERIMENTS ")
    print(" Elaborato Finale - Algoritmi Genetici per il TSP ")
    print(f" Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")


def download_instances():
    """Download required TSPLIB instances"""
    print("\n" + "-"*60)
    print("DOWNLOADING TSPLIB INSTANCES")
    print("-"*60)
    download_tsplib_instances()
    print("\nDownload complete!")


def run_standard_experiments(instances):
    """Run standard GA experiments on all instances"""
    print("\n" + "="*70)
    print("RUNNING STANDARD EXPERIMENTS")
    print("="*70)
    print("Configuration:")
    print("  - 30 independent runs per instance")
    print("  - Population: 100, Generations: 500")
    print("  - Crossover: 0.8, Mutation: 0.1")
    print("  - Tournament size: 3, Elite size: 2")
    print("-"*70)

    batch_runner = BatchExperimentRunner(instances, n_runs=30)
    results = batch_runner.run_all_standard_experiments()

    return results


def run_sensitivity_analyses(instances):
    """Run all sensitivity analyses"""
    print("\n" + "="*70)
    print("RUNNING SENSITIVITY ANALYSES")
    print("="*70)
    print("Planned analyses:")
    print("  1. Population size (berlin52): [50, 100, 150, 200]")
    print("  2. Generations (berlin52): [250, 500, 750, 1000]")
    print("  3. Mutation rate (kroA100): [0.05, 0.10, 0.15, 0.20]")
    print("-"*70)

    batch_runner = BatchExperimentRunner(instances, n_runs=30)
    results = batch_runner.run_all_sensitivity_analyses()

    return results


def generate_visualizations(instances):
    """Generate all thesis figures"""
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)

    visualizer = ThesisVisualizer()
    visualizer.generate_all_figures(instances)

    # Generate tables
    print("\nGenerating tables...")
    generate_tables()


def run_quick_test(instances):
    """Run a quick test with reduced parameters"""
    print("\n" + "="*70)
    print("RUNNING QUICK TEST")
    print("="*70)
    print("Reduced parameters for testing:")
    print("  - 3 runs per experiment")
    print("  - 100 generations")
    print("  - Only berlin52")
    print("-"*70)

    if 'berlin52' not in instances:
        print("Error: berlin52 not available for testing")
        return

    instance = instances['berlin52']
    runner = ExperimentRunner(instance, n_runs=3)

    # Quick standard run
    test_params = {
        'pop_size': 50,
        'n_generations': 100,
        'crossover_rate': 0.8,
        'mutation_rate': 0.1,
        'tournament_size': 3,
        'elite_size': 2
    }

    results = runner.run_standard_experiment(test_params)
    runner.save_results(results, "test_run.json")

    # Quick baseline
    baseline = runner.run_baseline_experiments()
    runner.save_results(baseline, "test_baseline.json")

    print("\nQuick test complete!")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Run TSP GA experiments for thesis'
    )
    parser.add_argument('--all', action='store_true',
                       help='Run all experiments and generate figures')
    parser.add_argument('--standard', action='store_true',
                       help='Run standard experiments only')
    parser.add_argument('--sensitivity', action='store_true',
                       help='Run sensitivity analyses only')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate figures from existing results')
    parser.add_argument('--download', action='store_true',
                       help='Download TSPLIB instances')
    parser.add_argument('--test', action='store_true',
                       help='Run quick test with reduced parameters')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Verbose output')

    args = parser.parse_args()

    # If no specific action, show help
    if not any([args.all, args.standard, args.sensitivity,
                args.visualize, args.download, args.test]):
        parser.print_help()
        return

    print_header()
    total_start = time.time()

    try:
        # Download instances if requested
        if args.download or args.all:
            download_instances()

        # Load TSP instances
        print("\nLoading TSP instances...")
        instances = load_all_instances()

        if not instances:
            print("Error: No TSP instances found. Run with --download first.")
            return

        print(f"Loaded {len(instances)} instances: {list(instances.keys())}")

        # Run experiments based on arguments
        if args.test:
            run_quick_test(instances)

        elif args.all:
            # Run everything
            print("\n" + "="*70)
            print("RUNNING COMPLETE EXPERIMENT SUITE")
            print("="*70)
            print("This will take several hours...")
            print("Estimated experiments: 516 total")
            print("  - Standard: 90 GA runs + 3 baselines")
            print("  - Sensitivity: 420 GA runs")
            print("-"*70)

            response = input("\nContinue? (y/n): ")
            if response.lower() != 'y':
                print("Aborted by user")
                return

            # Standard experiments
            run_standard_experiments(instances)

            # Sensitivity analyses
            run_sensitivity_analyses(instances)

            # Generate figures
            generate_visualizations(instances)

        else:
            if args.standard:
                run_standard_experiments(instances)

            if args.sensitivity:
                run_sensitivity_analyses(instances)

            if args.visualize:
                generate_visualizations(instances)

        # Summary
        total_time = time.time() - total_start
        print("\n" + "="*70)
        print("EXECUTION COMPLETE")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70 + "\n")

    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user")
    except Exception as e:
        print(f"\n\nError during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()