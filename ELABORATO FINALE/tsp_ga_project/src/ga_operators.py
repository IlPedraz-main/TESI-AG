"""
Genetic Algorithm Operators Module
Implementation of crossover, mutation and selection operators for TSP
"""

import random
import numpy as np
from typing import List, Tuple


def order_crossover(parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
    """
    Order Crossover (OX) for permutation representation

    Args:
        parent1: First parent tour
        parent2: Second parent tour

    Returns:
        Tuple of two offspring tours
    """
    size = len(parent1)

    # Select two random crossover points
    cx1 = random.randint(0, size - 2)
    cx2 = random.randint(cx1 + 1, size)

    # Create child1
    child1 = [None] * size
    child1[cx1:cx2] = parent1[cx1:cx2]
    copied = set(parent1[cx1:cx2])

    # Fill remaining positions from parent2
    p2_idx = 0
    for i in range(size):
        if child1[i] is None:
            while parent2[p2_idx] in copied:
                p2_idx += 1
            child1[i] = parent2[p2_idx]
            p2_idx += 1

    # Create child2 (symmetric process)
    child2 = [None] * size
    child2[cx1:cx2] = parent2[cx1:cx2]
    copied = set(parent2[cx1:cx2])

    p1_idx = 0
    for i in range(size):
        if child2[i] is None:
            while parent1[p1_idx] in copied:
                p1_idx += 1
            child2[i] = parent1[p1_idx]
            p1_idx += 1

    return child1, child2


def inversion_mutation(individual: List[int], mutation_rate: float) -> List[int]:
    """
    Inversion mutation - reverses a random segment of the tour

    Args:
        individual: Tour to mutate
        mutation_rate: Probability of mutation

    Returns:
        Mutated tour (modified in-place)
    """
    if random.random() < mutation_rate:
        size = len(individual)
        idx1 = random.randint(0, size - 2)
        idx2 = random.randint(idx1 + 1, size)
        individual[idx1:idx2] = reversed(individual[idx1:idx2])
    return individual


def swap_mutation(individual: List[int], mutation_rate: float) -> List[int]:
    """
    Swap mutation - exchanges two random cities

    Args:
        individual: Tour to mutate
        mutation_rate: Probability of mutation

    Returns:
        Mutated tour (modified in-place)
    """
    if random.random() < mutation_rate:
        size = len(individual)
        idx1 = random.randint(0, size - 1)
        idx2 = random.randint(0, size - 1)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual


def tournament_selection(population: List[List[int]], fitnesses: List[float],
                        k: int = 3) -> List[int]:
    """
    Tournament selection - select best from k random individuals

    Args:
        population: List of tours
        fitnesses: Fitness values for each tour
        k: Tournament size

    Returns:
        Selected individual (copy)
    """
    indices = random.sample(range(len(population)), min(k, len(population)))
    best = min(indices, key=lambda i: fitnesses[i])  # Min because we minimize distance
    return population[best][:]  # Return copy


def roulette_wheel_selection(population: List[List[int]], fitnesses: List[float]) -> List[int]:
    """
    Roulette wheel selection based on fitness proportionate selection

    Args:
        population: List of tours
        fitnesses: Fitness values for each tour

    Returns:
        Selected individual (copy)
    """
    # Convert to maximization problem (invert fitness)
    max_fitness = max(fitnesses)
    adjusted_fitnesses = [max_fitness - f for f in fitnesses]

    # Handle edge case where all fitnesses are equal
    if sum(adjusted_fitnesses) == 0:
        return random.choice(population)[:]

    # Calculate probabilities
    total = sum(adjusted_fitnesses)
    probabilities = [f / total for f in adjusted_fitnesses]

    # Select based on probabilities
    selected_idx = np.random.choice(len(population), p=probabilities)
    return population[selected_idx][:]


def rank_selection(population: List[List[int]], fitnesses: List[float]) -> List[int]:
    """
    Rank-based selection

    Args:
        population: List of tours
        fitnesses: Fitness values for each tour

    Returns:
        Selected individual (copy)
    """
    # Sort by fitness (ascending for minimization)
    sorted_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])

    # Assign ranks (best gets highest rank)
    ranks = [0] * len(population)
    for i, idx in enumerate(sorted_indices):
        ranks[idx] = len(population) - i

    # Select based on ranks
    total_rank = sum(ranks)
    probabilities = [r / total_rank for r in ranks]

    selected_idx = np.random.choice(len(population), p=probabilities)
    return population[selected_idx][:]


def pmx_crossover(parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
    """
    Partially Mapped Crossover (PMX) for permutation representation

    Args:
        parent1: First parent tour
        parent2: Second parent tour

    Returns:
        Tuple of two offspring tours
    """
    size = len(parent1)

    # Select two random crossover points
    cx1 = random.randint(0, size - 2)
    cx2 = random.randint(cx1 + 1, size)

    # Create mapping between segments
    mapping1 = {}
    mapping2 = {}

    for i in range(cx1, cx2):
        mapping1[parent2[i]] = parent1[i]
        mapping2[parent1[i]] = parent2[i]

    # Create child1
    child1 = parent2[:]
    for i in range(cx1, cx2):
        child1[i] = parent1[i]

    # Fix conflicts in child1
    for i in list(range(0, cx1)) + list(range(cx2, size)):
        while child1[i] in parent1[cx1:cx2]:
            child1[i] = mapping1[child1[i]]

    # Create child2 (symmetric process)
    child2 = parent1[:]
    for i in range(cx1, cx2):
        child2[i] = parent2[i]

    # Fix conflicts in child2
    for i in list(range(0, cx1)) + list(range(cx2, size)):
        while child2[i] in parent2[cx1:cx2]:
            child2[i] = mapping2[child2[i]]

    return child1, child2


def cycle_crossover(parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
    """
    Cycle Crossover (CX) for permutation representation

    Args:
        parent1: First parent tour
        parent2: Second parent tour

    Returns:
        Tuple of two offspring tours
    """
    size = len(parent1)
    child1 = [None] * size
    child2 = [None] * size

    # Create position mapping for parent2
    pos_map = {val: idx for idx, val in enumerate(parent2)}

    # Find cycles
    visited = [False] * size
    cycle_num = 0

    for start in range(size):
        if not visited[start]:
            cycle = []
            idx = start

            # Follow the cycle
            while idx not in [c for c in cycle]:
                cycle.append(idx)
                visited[idx] = True
                idx = pos_map[parent1[idx]]

            # Assign cycle elements alternately to children
            if cycle_num % 2 == 0:
                for idx in cycle:
                    child1[idx] = parent1[idx]
                    child2[idx] = parent2[idx]
            else:
                for idx in cycle:
                    child1[idx] = parent2[idx]
                    child2[idx] = parent1[idx]

            cycle_num += 1

    return child1, child2