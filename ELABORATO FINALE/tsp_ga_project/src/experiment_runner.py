"""
Experiment Runner Module
Manages batch execution of experiments and collection of results
"""

import json
import os
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
from tsp_loader import TSPInstance
from ga_engine import GeneticAlgorithm
from baseline import nearest_neighbor


class ExperimentRunner:
    """Class to run and manage experiments"""

    def __init__(self, tsp_instance: TSPInstance, n_runs: int = 30,
                 results_dir: str = "results"):
        """
        Initialize experiment runner

        Args:
            tsp_instance: TSP instance to solve
            n_runs: Number of independent runs
            results_dir: Directory to save results
        """
        self.instance = tsp_instance
        self.n_runs = n_runs
        self.results_dir = results_dir

        # Create results directory
        self.instance_dir = os.path.join(results_dir, tsp_instance.name)
        os.makedirs(self.instance_dir, exist_ok=True)

    def run_standard_experiment(self, ga_params: Optional[Dict] = None,
                               verbose: bool = True) -> Dict:
        """
        Run standard experiment with default parameters

        Args:
            ga_params: GA parameters (uses defaults if None)
            verbose: Print progress

        Returns:
            Dictionary with all results
        """
        if ga_params is None:
            ga_params = {
                'pop_size': 100,
                'n_generations': 500,
                'crossover_rate': 0.8,
                'mutation_rate': 0.1,
                'tournament_size': 3,
                'elite_size': 2
            }

        results = {
            'instance': self.instance.name,
            'config': ga_params,
            'optimal': self.instance.optimal,
            'runs': []
        }

        if verbose:
            print(f"\nRunning {self.n_runs} experiments on {self.instance.name}")
            print(f"Parameters: {ga_params}")
            print("-" * 60)

        for run_id in range(self.n_runs):
            if verbose and (run_id + 1) % 5 == 0:
                print(f"Completed {run_id + 1}/{self.n_runs} runs")

            # Run GA with specific seed
            ga = GeneticAlgorithm(
                self.instance,
                random_seed=run_id,
                **ga_params
            )

            start_time = time.time()
            best_tour, best_fitness = ga.evolve()
            elapsed_time = time.time() - start_time

            # Collect run data
            run_data = {
                'run_id': run_id,
                'seed': run_id,
                'best_fitness': best_fitness,
                'gap_percent': self.instance.gap_from_optimal(best_fitness),
                'time_seconds': elapsed_time,
                'convergence_generation': ga.convergence_generation,
                'best_fitness_history': ga.best_fitness_history,
                'mean_fitness_history': ga.mean_fitness_history,
                'diversity_history': ga.diversity_history,
                'best_tour': best_tour
            }

            results['runs'].append(run_data)

        # Calculate statistics
        results['statistics'] = self._calculate_statistics(results['runs'])

        if verbose:
            self._print_summary(results)

        return results

    def run_sensitivity_analysis(self, parameter_name: str,
                                 parameter_values: List[Any],
                                 base_params: Optional[Dict] = None,
                                 verbose: bool = True) -> Dict:
        """
        Run sensitivity analysis for a specific parameter

        Args:
            parameter_name: Name of parameter to vary
            parameter_values: List of values to test
            base_params: Base GA parameters
            verbose: Print progress

        Returns:
            Dictionary with sensitivity analysis results
        """
        if base_params is None:
            base_params = {
                'pop_size': 100,
                'n_generations': 500,
                'crossover_rate': 0.8,
                'mutation_rate': 0.1,
                'tournament_size': 3,
                'elite_size': 2
            }

        results = {
            'instance': self.instance.name,
            'parameter_name': parameter_name,
            'parameter_values': parameter_values,
            'base_params': base_params,
            'experiments': []
        }

        if verbose:
            print(f"\nSensitivity Analysis: {parameter_name}")
            print(f"Testing values: {parameter_values}")
            print(f"Instance: {self.instance.name}")
            print("-" * 60)

        for value in parameter_values:
            if verbose:
                print(f"\nTesting {parameter_name} = {value}")

            # Update parameters
            test_params = base_params.copy()
            test_params[parameter_name] = value

            # Run experiment
            exp_results = self.run_standard_experiment(test_params, verbose=False)

            # Store results
            results['experiments'].append({
                'parameter_value': value,
                'results': exp_results
            })

            if verbose:
                stats = exp_results['statistics']
                print(f"  Gap: {stats['gap_percent']['mean']:.2f}% "
                      f"(±{stats['gap_percent']['std']:.2f}%)")
                print(f"  Time: {stats['time_seconds']['mean']:.1f}s")

        return results

    def run_baseline_experiments(self, verbose: bool = True) -> Dict:
        """
        Run baseline algorithms for comparison

        Args:
            verbose: Print progress

        Returns:
            Dictionary with baseline results
        """
        results = {
            'instance': self.instance.name,
            'optimal': self.instance.optimal,
            'algorithms': {}
        }

        if verbose:
            print(f"\nRunning baseline algorithms on {self.instance.name}")
            print("-" * 60)

        # Nearest Neighbor
        start_time = time.time()
        nn_tour, nn_length = nearest_neighbor(self.instance.distance_matrix, start=0)
        nn_time = time.time() - start_time

        results['algorithms']['nearest_neighbor'] = {
            'tour': nn_tour,
            'fitness': nn_length,
            'gap_percent': self.instance.gap_from_optimal(nn_length),
            'time_seconds': nn_time
        }

        if verbose:
            print(f"Nearest Neighbor: {nn_length:.0f} "
                  f"(gap: {results['algorithms']['nearest_neighbor']['gap_percent']:.1f}%), "
                  f"time: {nn_time:.4f}s")

        return results

    def _calculate_statistics(self, runs: List[Dict]) -> Dict:
        """Calculate statistics from runs"""
        fitness_values = [r['best_fitness'] for r in runs]
        gap_values = [r['gap_percent'] for r in runs if r['gap_percent'] is not None]
        time_values = [r['time_seconds'] for r in runs]
        convergence_values = [r['convergence_generation'] for r in runs]

        stats = {
            'best_fitness': {
                'min': np.min(fitness_values),
                'mean': np.mean(fitness_values),
                'max': np.max(fitness_values),
                'std': np.std(fitness_values)
            },
            'time_seconds': {
                'mean': np.mean(time_values),
                'std': np.std(time_values)
            },
            'convergence_generation': {
                'mean': np.mean(convergence_values),
                'std': np.std(convergence_values)
            }
        }

        if gap_values:
            stats['gap_percent'] = {
                'min': np.min(gap_values),
                'mean': np.mean(gap_values),
                'max': np.max(gap_values),
                'std': np.std(gap_values)
            }

        return stats

    def _print_summary(self, results: Dict):
        """Print summary of results"""
        stats = results['statistics']
        print("\n" + "=" * 60)
        print(f"SUMMARY - {results['instance']}")
        print("=" * 60)
        print(f"Runs: {len(results['runs'])}")
        print(f"Optimal: {results['optimal']}")
        print(f"Best found: {stats['best_fitness']['min']:.0f}")
        print(f"Mean: {stats['best_fitness']['mean']:.0f} "
              f"(±{stats['best_fitness']['std']:.0f})")
        if 'gap_percent' in stats:
            print(f"Gap: {stats['gap_percent']['mean']:.2f}% "
                  f"(±{stats['gap_percent']['std']:.2f}%)")
        print(f"Time: {stats['time_seconds']['mean']:.1f}s "
              f"(±{stats['time_seconds']['std']:.1f}s)")
        print(f"Convergence: gen {stats['convergence_generation']['mean']:.0f} "
              f"(±{stats['convergence_generation']['std']:.0f})")

    def save_results(self, results: Dict, filename: str):
        """Save results to JSON file"""
        filepath = os.path.join(self.instance_dir, filename)

        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj

        serializable_results = convert_to_serializable(results)

        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"Results saved to {filepath}")

    def load_results(self, filename: str) -> Dict:
        """Load results from JSON file"""
        filepath = os.path.join(self.instance_dir, filename)
        with open(filepath, 'r') as f:
            return json.load(f)

    def export_to_csv(self, results: Dict, filename: str):
        """Export summary statistics to CSV"""
        filepath = os.path.join(self.instance_dir, filename)

        # Create DataFrame from runs
        runs_data = []
        for run in results['runs']:
            runs_data.append({
                'run_id': run['run_id'],
                'best_fitness': run['best_fitness'],
                'gap_percent': run['gap_percent'],
                'time_seconds': run['time_seconds'],
                'convergence_generation': run['convergence_generation']
            })

        df = pd.DataFrame(runs_data)
        df.to_csv(filepath, index=False)
        print(f"CSV exported to {filepath}")

        return df


class BatchExperimentRunner:
    """Run experiments across multiple instances"""

    def __init__(self, instances: Dict[str, TSPInstance],
                 n_runs: int = 30, results_dir: str = "results"):
        """
        Initialize batch runner

        Args:
            instances: Dictionary of TSP instances
            n_runs: Number of runs per instance
            results_dir: Directory to save results
        """
        self.instances = instances
        self.n_runs = n_runs
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def run_all_standard_experiments(self, ga_params: Optional[Dict] = None) -> Dict:
        """Run standard experiments on all instances"""
        all_results = {}

        for name, instance in self.instances.items():
            print(f"\n{'='*70}")
            print(f"Instance: {name}")
            print(f"{'='*70}")

            runner = ExperimentRunner(instance, self.n_runs, self.results_dir)

            # Run standard experiment
            results = runner.run_standard_experiment(ga_params)
            runner.save_results(results, "standard_runs.json")
            runner.export_to_csv(results, "standard_runs.csv")

            # Run baseline
            baseline_results = runner.run_baseline_experiments()
            runner.save_results(baseline_results, "nearest_neighbor.json")

            all_results[name] = {
                'ga': results,
                'baseline': baseline_results
            }

        return all_results

    def run_all_sensitivity_analyses(self) -> Dict:
        """Run all required sensitivity analyses"""
        results = {}

        # Population size sensitivity on berlin52
        if 'berlin52' in self.instances:
            print("\n" + "="*70)
            print("Population Size Sensitivity (berlin52)")
            print("="*70)

            runner = ExperimentRunner(self.instances['berlin52'],
                                    self.n_runs, self.results_dir)
            pop_results = runner.run_sensitivity_analysis(
                'pop_size', [50, 100, 150, 200]
            )
            runner.save_results(pop_results, "pop_sensitivity.json")
            results['pop_sensitivity'] = pop_results

        # Generation sensitivity on berlin52
        if 'berlin52' in self.instances:
            print("\n" + "="*70)
            print("Generation Sensitivity (berlin52)")
            print("="*70)

            runner = ExperimentRunner(self.instances['berlin52'],
                                    self.n_runs, self.results_dir)
            gen_results = runner.run_sensitivity_analysis(
                'n_generations', [250, 500, 750, 1000]
            )
            runner.save_results(gen_results, "gen_sensitivity.json")
            results['gen_sensitivity'] = gen_results

        # Mutation rate sensitivity on kroA100
        if 'kroA100' in self.instances:
            print("\n" + "="*70)
            print("Mutation Rate Sensitivity (kroA100)")
            print("="*70)

            runner = ExperimentRunner(self.instances['kroA100'],
                                    self.n_runs, self.results_dir)
            mut_results = runner.run_sensitivity_analysis(
                'mutation_rate', [0.05, 0.10, 0.15, 0.20]
            )
            runner.save_results(mut_results, "mut_sensitivity.json")
            results['mut_sensitivity'] = mut_results

        return results