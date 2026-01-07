"""
Genetic Algorithm Engine Module
Main GA implementation for TSP
"""

import random
import numpy as np
import time
from typing import List, Dict, Tuple, Optional
from ga_operators import (
    order_crossover, inversion_mutation, tournament_selection,
    swap_mutation, pmx_crossover, cycle_crossover
)


class GeneticAlgorithm:
    """Genetic Algorithm for TSP"""

    def __init__(self, tsp_instance, pop_size: int = 100,
                 n_generations: int = 500, crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1, tournament_size: int = 3,
                 elite_size: int = 2, random_seed: Optional[int] = None,
                 crossover_type: str = 'ox', mutation_type: str = 'inversion'):
        """
        Initialize GA with parameters

        Args:
            tsp_instance: TSPInstance object
            pop_size: Population size
            n_generations: Number of generations
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            tournament_size: Size of tournament selection
            elite_size: Number of elite individuals to preserve
            random_seed: Random seed for reproducibility
            crossover_type: Type of crossover ('ox', 'pmx', 'cx')
            mutation_type: Type of mutation ('inversion', 'swap')
        """
        self.instance = tsp_instance
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elite_size = elite_size

        # Set crossover operator
        self.crossover_type = crossover_type
        if crossover_type == 'ox':
            self.crossover = order_crossover
        elif crossover_type == 'pmx':
            self.crossover = pmx_crossover
        elif crossover_type == 'cx':
            self.crossover = cycle_crossover
        else:
            self.crossover = order_crossover

        # Set mutation operator
        self.mutation_type = mutation_type
        if mutation_type == 'inversion':
            self.mutation = inversion_mutation
        elif mutation_type == 'swap':
            self.mutation = swap_mutation
        else:
            self.mutation = inversion_mutation

        # Set random seed if provided
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        # Initialize tracking variables
        self.best_fitness_history = []
        self.mean_fitness_history = []
        self.diversity_history = []
        self.best_individual = None
        self.best_fitness = float('inf')
        self.convergence_generation = 0

    def initialize_population(self) -> List[List[int]]:
        """Generate initial population of random tours"""
        pop = []
        cities = list(range(self.instance.n_cities))

        for _ in range(self.pop_size):
            tour = cities[:]
            random.shuffle(tour)
            pop.append(tour)

        return pop

    def evaluate_population(self, pop: List[List[int]]) -> List[float]:
        """Evaluate fitness of entire population"""
        return [self.instance.evaluate_tour(ind) for ind in pop]

    def calculate_diversity(self, population: List[List[int]]) -> float:
        """
        Calculate population diversity as average pairwise distance

        Distance metric: number of different edges between tours
        """
        if len(population) < 2:
            return 0.0

        n_samples = min(50, len(population))  # Sample for efficiency
        sample = random.sample(population, n_samples)

        total_distance = 0
        count = 0

        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                # Count different edges
                edges1 = set()
                edges2 = set()

                tour1 = sample[i]
                tour2 = sample[j]

                for k in range(len(tour1)):
                    next_k = (k + 1) % len(tour1)
                    edges1.add((min(tour1[k], tour1[next_k]),
                               max(tour1[k], tour1[next_k])))
                    edges2.add((min(tour2[k], tour2[next_k]),
                               max(tour2[k], tour2[next_k])))

                different_edges = len(edges1.symmetric_difference(edges2))
                total_distance += different_edges
                count += 1

        return total_distance / count if count > 0 else 0.0

    def evolve(self) -> Tuple[List[int], float]:
        """
        Run the genetic algorithm

        Returns:
            Best tour found and its fitness
        """
        # Initialize population
        population = self.initialize_population()
        fitnesses = self.evaluate_population(population)

        # Track best individual
        best_idx = np.argmin(fitnesses)
        self.best_individual = population[best_idx][:]
        self.best_fitness = fitnesses[best_idx]

        # Track initial state
        self.best_fitness_history.append(self.best_fitness)
        self.mean_fitness_history.append(np.mean(fitnesses))
        self.diversity_history.append(self.calculate_diversity(population))

        # Evolution loop
        for generation in range(self.n_generations):
            # Create offspring population
            offspring = []

            for _ in range(self.pop_size // 2):
                # Selection
                p1 = tournament_selection(population, fitnesses, self.tournament_size)
                p2 = tournament_selection(population, fitnesses, self.tournament_size)

                # Crossover
                if random.random() < self.crossover_rate:
                    c1, c2 = self.crossover(p1, p2)
                else:
                    c1, c2 = p1[:], p2[:]

                # Mutation
                c1 = self.mutation(c1, self.mutation_rate)
                c2 = self.mutation(c2, self.mutation_rate)

                offspring.extend([c1, c2])

            # Ensure we have exactly pop_size individuals
            if len(offspring) < self.pop_size:
                offspring.append(tournament_selection(population, fitnesses, self.tournament_size))

            # Evaluate offspring
            off_fitnesses = self.evaluate_population(offspring)

            # Elitism - preserve best individuals from current population
            elite_idx = np.argsort(fitnesses)[:self.elite_size]
            elite = [population[i][:] for i in elite_idx]
            elite_fit = [fitnesses[i] for i in elite_idx]

            # Select best offspring to fill remaining slots
            remaining_slots = self.pop_size - self.elite_size
            best_off_idx = np.argsort(off_fitnesses)[:remaining_slots]

            # Form new population
            population = elite + [offspring[i][:] for i in best_off_idx]
            fitnesses = elite_fit + [off_fitnesses[i] for i in best_off_idx]

            # Update best solution
            gen_best_idx = np.argmin(fitnesses)
            if fitnesses[gen_best_idx] < self.best_fitness:
                self.best_fitness = fitnesses[gen_best_idx]
                self.best_individual = population[gen_best_idx][:]
                self.convergence_generation = generation + 1

            # Track history
            self.best_fitness_history.append(self.best_fitness)
            self.mean_fitness_history.append(np.mean(fitnesses))
            self.diversity_history.append(self.calculate_diversity(population))

            # Optional: Print progress
            if (generation + 1) % 100 == 0:
                gap = self.instance.gap_from_optimal(self.best_fitness)
                if gap is not None:
                    print(f"Gen {generation + 1}: Best = {self.best_fitness:.0f} "
                          f"(gap: {gap:.2f}%), Mean = {np.mean(fitnesses):.0f}")

        return self.best_individual, self.best_fitness

    def get_statistics(self) -> Dict:
        """Get run statistics"""
        return {
            'best_fitness': self.best_fitness,
            'gap_percent': self.instance.gap_from_optimal(self.best_fitness),
            'convergence_generation': self.convergence_generation,
            'best_fitness_history': self.best_fitness_history,
            'mean_fitness_history': self.mean_fitness_history,
            'diversity_history': self.diversity_history,
            'parameters': {
                'pop_size': self.pop_size,
                'n_generations': self.n_generations,
                'crossover_rate': self.crossover_rate,
                'mutation_rate': self.mutation_rate,
                'tournament_size': self.tournament_size,
                'elite_size': self.elite_size,
                'crossover_type': self.crossover_type,
                'mutation_type': self.mutation_type
            }
        }


class AdaptiveGA(GeneticAlgorithm):
    """GA with adaptive parameters"""

    def __init__(self, tsp_instance, **kwargs):
        super().__init__(tsp_instance, **kwargs)
        self.initial_mutation_rate = self.mutation_rate
        self.stagnation_counter = 0
        self.stagnation_threshold = 20

    def adapt_parameters(self, generation: int, fitness_improvement: float):
        """Adapt GA parameters based on progress"""
        # Reduce mutation rate over time
        progress = generation / self.n_generations
        self.mutation_rate = self.initial_mutation_rate * (1 - progress ** 2)

        # Increase mutation if stagnant
        if fitness_improvement < 0.001:
            self.stagnation_counter += 1
            if self.stagnation_counter > self.stagnation_threshold:
                self.mutation_rate = min(0.3, self.mutation_rate * 1.5)
                self.stagnation_counter = 0
        else:
            self.stagnation_counter = 0