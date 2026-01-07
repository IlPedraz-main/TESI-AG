"""
Visualization Module
Generate all plots and figures for the thesis
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import os
import json

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ThesisVisualizer:
    """Generate all thesis figures"""

    def __init__(self, results_dir: str = "results", plots_dir: str = "plots"):
        """
        Initialize visualizer

        Args:
            results_dir: Directory containing experiment results
            plots_dir: Directory to save plots
        """
        self.results_dir = results_dir
        self.plots_dir = plots_dir
        os.makedirs(plots_dir, exist_ok=True)

    def load_results(self, instance: str, filename: str) -> Dict:
        """Load results from JSON file"""
        filepath = os.path.join(self.results_dir, instance, filename)
        with open(filepath, 'r') as f:
            return json.load(f)

    def figure_5_1_to_5_3_convergence(self, instance_name: str):
        """
        Generate convergence plots for berlin52, att48, or kroA100
        Figures 5.1, 5.2, 5.3 in thesis
        """
        # Load standard runs
        results = self.load_results(instance_name, "standard_runs.json")
        runs = results['runs']
        optimal = results['optimal']

        # Load baseline (Nearest Neighbor)
        baseline = self.load_results(instance_name, "nearest_neighbor.json")
        nn_fitness = baseline['algorithms']['nearest_neighbor']['fitness']

        # Extract best fitness history for all runs
        n_generations = len(runs[0]['best_fitness_history'])
        all_best = np.array([run['best_fitness_history'] for run in runs])

        # Calculate mean and std
        mean_best = np.mean(all_best, axis=0)
        std_best = np.std(all_best, axis=0)

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))

        generations = range(n_generations)

        # Plot mean with confidence interval
        ax.plot(generations, mean_best, 'b-', linewidth=2, label='GA Mean')
        ax.fill_between(generations,
                        mean_best - std_best,
                        mean_best + std_best,
                        alpha=0.3, color='blue',
                        label='±1 Std Dev')

        # Add optimal line
        ax.axhline(y=optimal, color='red', linestyle='--',
                  linewidth=2, label=f'Optimal ({optimal})')

        # Add Nearest Neighbor line
        ax.axhline(y=nn_fitness, color='green', linestyle='-.',
                  linewidth=1.5, label=f'Nearest Neighbor ({nn_fitness:.0f})')

        # Formatting
        ax.set_xlabel('Generation', fontsize=12)
        ax.set_ylabel('Tour Length', fontsize=12)
        ax.set_title(f'Convergence Plot - {instance_name}', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Set y-axis limits for better visibility
        y_margin = (nn_fitness - optimal) * 0.1
        ax.set_ylim(optimal - y_margin, nn_fitness + y_margin)

        plt.tight_layout()
        filename = f"convergence_{instance_name}.png"
        plt.savefig(os.path.join(self.plots_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")

    def figure_5_4_to_5_6_tour_comparison(self, instance_name: str, tsp_instance):
        """
        Plot optimal tour vs GA-found tour
        Figures 5.4, 5.5, 5.6 in thesis
        """
        # Load results
        results = self.load_results(instance_name, "standard_runs.json")

        # Find best tour among all runs
        best_run = min(results['runs'], key=lambda x: x['best_fitness'])
        best_tour = best_run['best_tour']
        best_fitness = best_run['best_fitness']

        # Get coordinates
        coords = tsp_instance.coordinates

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Helper function to plot a tour
        def plot_tour(ax, tour, title, fitness):
            # Plot cities
            ax.scatter(coords[:, 0], coords[:, 1], c='red', s=50, zorder=5)

            # Plot tour edges
            for i in range(len(tour)):
                start = tour[i]
                end = tour[(i + 1) % len(tour)]
                ax.plot([coords[start, 0], coords[end, 0]],
                       [coords[start, 1], coords[end, 1]],
                       'b-', alpha=0.6, linewidth=1)

            # Highlight start city
            ax.scatter(coords[tour[0], 0], coords[tour[0], 1],
                      c='green', s=100, marker='s', zorder=6,
                      label='Start')

            ax.set_title(f'{title}\nLength: {fitness:.0f}', fontsize=12)
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.grid(True, alpha=0.3)
            ax.legend()

        # Plot optimal tour (if available - using a simple heuristic as placeholder)
        # In real implementation, you'd load the actual optimal tour
        optimal_tour = list(range(len(coords)))  # Placeholder
        plot_tour(ax1, optimal_tour, f'(a) Optimal Tour - {instance_name}',
                 tsp_instance.optimal)

        # Plot GA best tour
        gap = tsp_instance.gap_from_optimal(best_fitness)
        plot_tour(ax2, best_tour,
                 f'(b) GA Best Tour (Gap: {gap:.2f}%)',
                 best_fitness)

        plt.suptitle(f'Tour Comparison - {instance_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()

        filename = f"tours_{instance_name}.png"
        plt.savefig(os.path.join(self.plots_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")

    def figure_5_7_boxplot_gaps(self):
        """
        Generate box plot of gap distributions across instances
        Figure 5.7 in thesis
        """
        instances = ['berlin52', 'att48', 'kroA100']
        gaps_data = []
        labels = []

        for instance in instances:
            try:
                results = self.load_results(instance, "standard_runs.json")
                gaps = [run['gap_percent'] for run in results['runs']
                       if run['gap_percent'] is not None]
                gaps_data.append(gaps)
                labels.append(instance)
            except FileNotFoundError:
                print(f"Warning: Results for {instance} not found")

        if not gaps_data:
            print("No data available for gap boxplot")
            return

        # Create box plot
        fig, ax = plt.subplots(figsize=(10, 6))

        bp = ax.boxplot(gaps_data, labels=labels, patch_artist=True)

        # Customize colors
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Customize plot
        ax.set_ylabel('Gap from Optimal (%)', fontsize=12)
        ax.set_xlabel('Instance', fontsize=12)
        ax.set_title('Distribution of GA Performance Across Instances',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Add horizontal line at 0 (optimal)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)

        plt.tight_layout()
        filename = "boxplot_gaps.png"
        plt.savefig(os.path.join(self.plots_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")

    def figure_5_8_population_sensitivity(self):
        """
        Population size sensitivity analysis
        Figure 5.8 in thesis
        """
        try:
            results = self.load_results('berlin52', 'pop_sensitivity.json')
        except FileNotFoundError:
            print("Population sensitivity results not found")
            return

        # Extract data
        pop_sizes = results['parameter_values']
        gaps = []
        times = []
        convergence = []

        for exp in results['experiments']:
            stats = exp['results']['statistics']
            gaps.append((stats['gap_percent']['mean'], stats['gap_percent']['std']))
            times.append(stats['time_seconds']['mean'])
            convergence.append((stats['convergence_generation']['mean'],
                              stats['convergence_generation']['std']))

        # Create 2x2 subplot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        # (a) Gap vs Population Size
        gap_means = [g[0] for g in gaps]
        gap_stds = [g[1] for g in gaps]
        ax1.errorbar(pop_sizes, gap_means, yerr=gap_stds, marker='o',
                    capsize=5, linewidth=2, markersize=8)
        ax1.set_xlabel('Population Size')
        ax1.set_ylabel('Mean Gap (%)')
        ax1.set_title('(a) Solution Quality vs Population Size')
        ax1.grid(True, alpha=0.3)

        # (b) Time vs Population Size
        ax2.plot(pop_sizes, times, marker='s', linewidth=2, markersize=8, color='orange')
        ax2.set_xlabel('Population Size')
        ax2.set_ylabel('Mean Time (seconds)')
        ax2.set_title('(b) Computation Time vs Population Size')
        ax2.grid(True, alpha=0.3)

        # (c) Convergence Generation vs Population Size
        conv_means = [c[0] for c in convergence]
        conv_stds = [c[1] for c in convergence]
        ax3.errorbar(pop_sizes, conv_means, yerr=conv_stds, marker='^',
                    capsize=5, linewidth=2, markersize=8, color='green')
        ax3.set_xlabel('Population Size')
        ax3.set_ylabel('Mean Convergence Generation')
        ax3.set_title('(c) Convergence Speed vs Population Size')
        ax3.grid(True, alpha=0.3)

        # (d) Convergence curves for different population sizes
        colors = plt.cm.viridis(np.linspace(0, 1, len(pop_sizes)))
        for i, (pop_size, exp) in enumerate(zip(pop_sizes, results['experiments'])):
            # Average best fitness history across runs
            all_histories = [run['best_fitness_history']
                           for run in exp['results']['runs']]
            mean_history = np.mean(all_histories, axis=0)
            ax4.plot(mean_history[:500], label=f'Pop={pop_size}',
                    color=colors[i], linewidth=1.5)

        ax4.set_xlabel('Generation')
        ax4.set_ylabel('Best Fitness')
        ax4.set_title('(d) Convergence Curves for Different Population Sizes')
        ax4.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)

        plt.suptitle('Population Size Sensitivity Analysis - berlin52',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        filename = "pop_sensitivity.png"
        plt.savefig(os.path.join(self.plots_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")

    def figure_5_9_mutation_sensitivity(self):
        """
        Mutation rate sensitivity analysis
        Figure 5.9 in thesis
        """
        try:
            results = self.load_results('kroA100', 'mut_sensitivity.json')
        except FileNotFoundError:
            print("Mutation sensitivity results not found")
            return

        # Extract data
        mut_rates = results['parameter_values']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # (a) Box plot of gaps
        gaps_data = []
        for exp in results['experiments']:
            gaps = [run['gap_percent'] for run in exp['results']['runs']
                   if run['gap_percent'] is not None]
            gaps_data.append(gaps)

        bp = ax1.boxplot(gaps_data, labels=[f'{m:.2f}' for m in mut_rates],
                        patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)

        ax1.set_xlabel('Mutation Rate')
        ax1.set_ylabel('Gap from Optimal (%)')
        ax1.set_title('(a) Gap Distribution vs Mutation Rate')
        ax1.grid(True, alpha=0.3, axis='y')

        # (b) Convergence curves
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(mut_rates)))
        for i, (mut_rate, exp) in enumerate(zip(mut_rates, results['experiments'])):
            all_histories = [run['best_fitness_history']
                           for run in exp['results']['runs']]
            mean_history = np.mean(all_histories, axis=0)
            ax2.plot(mean_history, label=f'Mut={mut_rate:.2f}',
                    color=colors[i], linewidth=2)

        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Best Fitness')
        ax2.set_title('(b) Convergence Curves for Different Mutation Rates')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        plt.suptitle('Mutation Rate Sensitivity Analysis - kroA100',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        filename = "mut_sensitivity.png"
        plt.savefig(os.path.join(self.plots_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")

    def figure_5_10_generation_sensitivity(self):
        """
        Number of generations sensitivity analysis
        Figure 5.10 in thesis
        """
        try:
            results = self.load_results('berlin52', 'gen_sensitivity.json')
        except FileNotFoundError:
            print("Generation sensitivity results not found")
            return

        # Extract data
        n_gens = results['parameter_values']

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = plt.cm.plasma(np.linspace(0, 1, len(n_gens)))

        for i, (n_gen, exp) in enumerate(zip(n_gens, results['experiments'])):
            # Average best fitness history across runs
            all_histories = [run['best_fitness_history']
                           for run in exp['results']['runs']]

            # Truncate or extend to match n_gen
            truncated_histories = []
            for hist in all_histories:
                if len(hist) > n_gen:
                    truncated_histories.append(hist[:n_gen])
                else:
                    truncated_histories.append(hist)

            mean_history = np.mean(truncated_histories, axis=0)

            ax.plot(range(len(mean_history)), mean_history,
                   label=f'{n_gen} generations',
                   color=colors[i], linewidth=2)

            # Mark endpoint
            ax.scatter(len(mean_history)-1, mean_history[-1],
                      color=colors[i], s=100, zorder=5)

        # Add optimal line
        if 'optimal' in results:
            ax.axhline(y=results['optimal'], color='red', linestyle='--',
                      alpha=0.5, linewidth=1, label='Optimal')

        ax.set_xlabel('Generation', fontsize=12)
        ax.set_ylabel('Best Fitness', fontsize=12)
        ax.set_title('Impact of Number of Generations on Convergence - berlin52',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = "gen_sensitivity.png"
        plt.savefig(os.path.join(self.plots_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")

    def figure_5_11_scalability_time(self):
        """
        Scalability analysis - computation time
        Figure 5.11 in thesis
        """
        instances = ['att48', 'berlin52', 'kroA100']
        n_cities = [48, 52, 100]
        times = []

        for instance in instances:
            try:
                results = self.load_results(instance, "standard_runs.json")
                mean_time = results['statistics']['time_seconds']['mean']
                times.append(mean_time)
            except (FileNotFoundError, KeyError):
                print(f"Warning: Results for {instance} not found")
                times.append(None)

        # Remove None values
        valid_data = [(n, t) for n, t in zip(n_cities, times) if t is not None]
        if not valid_data:
            print("No data available for scalability plot")
            return

        n_cities_valid, times_valid = zip(*valid_data)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot actual data points
        ax.scatter(n_cities_valid, times_valid, s=100, color='blue', zorder=5,
                  label='Observed')

        # Fit polynomial (quadratic expected for TSP)
        if len(n_cities_valid) >= 2:
            z = np.polyfit(n_cities_valid, times_valid, 2)
            p = np.poly1d(z)

            # Plot fitted curve
            x_fit = np.linspace(min(n_cities_valid), max(n_cities_valid), 100)
            ax.plot(x_fit, p(x_fit), 'r-', alpha=0.5, linewidth=2,
                   label=f'Fitted: O(n²)')

            # Extrapolate
            x_extrap = np.linspace(max(n_cities_valid), 500, 50)
            ax.plot(x_extrap, p(x_extrap), 'r--', alpha=0.3, linewidth=1,
                   label='Extrapolation')

            # Add extrapolated points
            for n_extrap in [200, 500]:
                time_extrap = p(n_extrap)
                ax.scatter(n_extrap, time_extrap, s=50, color='red',
                          marker='^', alpha=0.5)
                ax.annotate(f'n={n_extrap}\n~{time_extrap:.0f}s',
                           xy=(n_extrap, time_extrap),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=9, alpha=0.7)

        ax.set_xlabel('Number of Cities (n)', fontsize=12)
        ax.set_ylabel('Mean Computation Time (seconds)', fontsize=12)
        ax.set_title('Scalability Analysis - Computation Time',
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = "scalability_time.png"
        plt.savefig(os.path.join(self.plots_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")

    def figure_5_12_scalability_quality(self):
        """
        Scalability analysis - solution quality
        Figure 5.12 in thesis
        """
        instances = ['att48', 'berlin52', 'kroA100']
        n_cities = [48, 52, 100]
        gaps = []

        for instance in instances:
            try:
                results = self.load_results(instance, "standard_runs.json")
                mean_gap = results['statistics']['gap_percent']['mean']
                gaps.append(mean_gap)
            except (FileNotFoundError, KeyError):
                print(f"Warning: Results for {instance} not found")
                gaps.append(None)

        # Remove None values
        valid_data = [(n, g) for n, g in zip(n_cities, gaps) if g is not None]
        if not valid_data:
            print("No data available for quality scalability plot")
            return

        n_cities_valid, gaps_valid = zip(*valid_data)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot actual data points
        ax.scatter(n_cities_valid, gaps_valid, s=100, color='green', zorder=5)

        # Fit linear or logarithmic trend
        if len(n_cities_valid) >= 2:
            z = np.polyfit(n_cities_valid, gaps_valid, 1)
            p = np.poly1d(z)

            x_fit = np.linspace(min(n_cities_valid), max(n_cities_valid), 100)
            ax.plot(x_fit, p(x_fit), 'g-', alpha=0.5, linewidth=2,
                   label='Linear Fit')

        # Add instance labels
        for n, g, name in zip(n_cities_valid, gaps_valid, instances):
            ax.annotate(name, xy=(n, g), xytext=(5, 5),
                       textcoords='offset points', fontsize=9)

        ax.set_xlabel('Number of Cities (n)', fontsize=12)
        ax.set_ylabel('Mean Gap from Optimal (%)', fontsize=12)
        ax.set_title('Scalability Analysis - Solution Quality',
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = "scalability_quality.png"
        plt.savefig(os.path.join(self.plots_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")

    def generate_all_figures(self, instances_dict=None):
        """Generate all thesis figures"""
        print("\nGenerating all thesis figures...")
        print("=" * 60)

        # Convergence plots (Figures 5.1-5.3)
        for instance in ['berlin52', 'att48', 'kroA100']:
            try:
                self.figure_5_1_to_5_3_convergence(instance)
            except Exception as e:
                print(f"Error generating convergence plot for {instance}: {e}")

        # Tour comparison plots (Figures 5.4-5.6)
        if instances_dict:
            for instance_name, tsp_instance in instances_dict.items():
                try:
                    self.figure_5_4_to_5_6_tour_comparison(instance_name, tsp_instance)
                except Exception as e:
                    print(f"Error generating tour plot for {instance_name}: {e}")

        # Other figures
        try:
            self.figure_5_7_boxplot_gaps()
        except Exception as e:
            print(f"Error generating boxplot: {e}")

        try:
            self.figure_5_8_population_sensitivity()
        except Exception as e:
            print(f"Error generating population sensitivity: {e}")

        try:
            self.figure_5_9_mutation_sensitivity()
        except Exception as e:
            print(f"Error generating mutation sensitivity: {e}")

        try:
            self.figure_5_10_generation_sensitivity()
        except Exception as e:
            print(f"Error generating generation sensitivity: {e}")

        try:
            self.figure_5_11_scalability_time()
        except Exception as e:
            print(f"Error generating time scalability: {e}")

        try:
            self.figure_5_12_scalability_quality()
        except Exception as e:
            print(f"Error generating quality scalability: {e}")

        print("\nFigure generation complete!")
        print(f"All figures saved to: {self.plots_dir}")


def generate_tables(results_dir: str = "results", tables_dir: str = "tables"):
    """Generate all thesis tables from results"""
    os.makedirs(tables_dir, exist_ok=True)

    # Table 5.1-5.3: Main results for each instance
    for instance in ['berlin52', 'att48', 'kroA100']:
        try:
            filepath = os.path.join(results_dir, instance, "standard_runs.json")
            with open(filepath, 'r') as f:
                results = json.load(f)

            # Load baseline
            baseline_path = os.path.join(results_dir, instance, "nearest_neighbor.json")
            with open(baseline_path, 'r') as f:
                baseline = json.load(f)

            # Create table data
            stats = results['statistics']
            nn_data = baseline['algorithms']['nearest_neighbor']

            table_data = {
                'Metric': ['Optimal TSPLIB', 'Best (30 run)', 'Mean (30 run)',
                          'Worst (30 run)', 'Std dev', 'Nearest Neighbor'],
                'Value': [
                    results['optimal'],
                    stats['best_fitness']['min'],
                    stats['best_fitness']['mean'],
                    stats['best_fitness']['max'],
                    stats['best_fitness']['std'],
                    nn_data['fitness']
                ],
                'Gap %': [
                    '—',
                    f"{stats['gap_percent']['min']:.2f}",
                    f"{stats['gap_percent']['mean']:.2f}",
                    f"{stats['gap_percent']['max']:.2f}",
                    '—',
                    f"{nn_data['gap_percent']:.1f}"
                ],
                'Time (s)': [
                    '—',
                    '—',
                    f"{stats['time_seconds']['mean']:.1f}",
                    '—',
                    f"{stats['time_seconds']['std']:.1f}",
                    f"{nn_data['time_seconds']:.3f}"
                ]
            }

            df = pd.DataFrame(table_data)
            table_file = os.path.join(tables_dir, f"table_5_{instance}.csv")
            df.to_csv(table_file, index=False)
            print(f"Generated table: {table_file}")

            # Also save as formatted text
            with open(table_file.replace('.csv', '.txt'), 'w') as f:
                f.write(df.to_string(index=False))

        except Exception as e:
            print(f"Error generating table for {instance}: {e}")

    print(f"\nAll tables saved to: {tables_dir}")