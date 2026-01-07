"""
Baseline Algorithms Module
Implementation of baseline algorithms for comparison
"""

import random
import numpy as np
from typing import List, Tuple, Optional


def nearest_neighbor(distance_matrix: np.ndarray, start: int = 0) -> Tuple[List[int], float]:
    """
    Nearest Neighbor heuristic for TSP

    Args:
        distance_matrix: Matrix of distances between cities
        start: Starting city index

    Returns:
        Tour and its total length
    """
    n = len(distance_matrix)
    unvisited = set(range(n))
    tour = [start]
    unvisited.remove(start)
    current = start

    while unvisited:
        # Find nearest unvisited city
        nearest = min(unvisited, key=lambda c: distance_matrix[current][c])
        tour.append(nearest)
        unvisited.remove(nearest)
        current = nearest

    # Calculate total tour length
    length = sum(distance_matrix[tour[i]][tour[i+1]] for i in range(n-1))
    length += distance_matrix[tour[-1]][tour[0]]

    return tour, length


def nearest_neighbor_multi_start(distance_matrix: np.ndarray,
                                 n_starts: Optional[int] = None) -> Tuple[List[int], float]:
    """
    Nearest Neighbor with multiple starting points

    Args:
        distance_matrix: Matrix of distances between cities
        n_starts: Number of different starting points to try (default: all cities)

    Returns:
        Best tour found and its total length
    """
    n = len(distance_matrix)
    if n_starts is None:
        n_starts = n

    best_tour = None
    best_length = float('inf')

    for start in range(min(n_starts, n)):
        tour, length = nearest_neighbor(distance_matrix, start)
        if length < best_length:
            best_length = length
            best_tour = tour

    return best_tour, best_length


def random_tour(n_cities: int) -> List[int]:
    """
    Generate a random tour

    Args:
        n_cities: Number of cities

    Returns:
        Random tour
    """
    tour = list(range(n_cities))
    random.shuffle(tour)
    return tour


def greedy_construction(distance_matrix: np.ndarray) -> Tuple[List[int], float]:
    """
    Greedy construction heuristic - always add the shortest edge that doesn't create a subtour

    Args:
        distance_matrix: Matrix of distances between cities

    Returns:
        Tour and its total length
    """
    n = len(distance_matrix)

    # Create list of all edges sorted by distance
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            edges.append((distance_matrix[i][j], i, j))
    edges.sort()

    # Build tour by adding edges
    tour_edges = []
    degree = [0] * n  # Track degree of each node

    for dist, i, j in edges:
        # Check if adding this edge is valid
        if degree[i] < 2 and degree[j] < 2:
            # Check if it would create a subtour (except when completing the tour)
            if len(tour_edges) < n - 1 or (len(tour_edges) == n - 1):
                tour_edges.append((i, j))
                degree[i] += 1
                degree[j] += 1

                if len(tour_edges) == n:
                    break

    # Convert edges to tour
    tour = construct_tour_from_edges(tour_edges)

    # Calculate length
    length = sum(distance_matrix[tour[i]][tour[i+1]] for i in range(n-1))
    length += distance_matrix[tour[-1]][tour[0]]

    return tour, length


def construct_tour_from_edges(edges: List[Tuple[int, int]]) -> List[int]:
    """
    Construct a tour from a list of edges

    Args:
        edges: List of edges (tuples of city indices)

    Returns:
        Tour as ordered list of cities
    """
    if not edges:
        return []

    # Build adjacency list
    adj = {}
    for i, j in edges:
        if i not in adj:
            adj[i] = []
        if j not in adj:
            adj[j] = []
        adj[i].append(j)
        adj[j].append(i)

    # Find starting node (any node with degree > 0)
    start = next(iter(adj))
    tour = [start]
    visited = {start}

    current = start
    while len(tour) < len(adj):
        # Find unvisited neighbor
        next_city = None
        for neighbor in adj[current]:
            if neighbor not in visited:
                next_city = neighbor
                break

        if next_city is None:
            # This shouldn't happen with valid edges
            break

        tour.append(next_city)
        visited.add(next_city)
        current = next_city

    return tour


def two_opt_improvement(tour: List[int], distance_matrix: np.ndarray,
                        max_iterations: int = 1000) -> Tuple[List[int], float]:
    """
    Apply 2-opt local search to improve a tour

    Args:
        tour: Initial tour
        distance_matrix: Matrix of distances between cities
        max_iterations: Maximum number of improvements to attempt

    Returns:
        Improved tour and its length
    """
    n = len(tour)
    improved = True
    iterations = 0

    def tour_length(t):
        return sum(distance_matrix[t[i]][t[(i+1) % n]] for i in range(n))

    current_length = tour_length(tour)
    best_tour = tour[:]

    while improved and iterations < max_iterations:
        improved = False
        iterations += 1

        for i in range(n - 1):
            for j in range(i + 2, n):
                # Skip adjacent edges
                if j == (i + 1) % n:
                    continue

                # Calculate change in tour length
                # Remove edges: (i, i+1) and (j, j+1)
                # Add edges: (i, j) and (i+1, j+1)
                old_dist = (distance_matrix[best_tour[i]][best_tour[i+1]] +
                           distance_matrix[best_tour[j]][best_tour[(j+1) % n]])
                new_dist = (distance_matrix[best_tour[i]][best_tour[j]] +
                           distance_matrix[best_tour[i+1]][best_tour[(j+1) % n]])

                if new_dist < old_dist:
                    # Perform 2-opt swap
                    best_tour[i+1:j+1] = reversed(best_tour[i+1:j+1])
                    current_length = current_length - old_dist + new_dist
                    improved = True
                    break

            if improved:
                break

    return best_tour, current_length


def savings_algorithm(distance_matrix: np.ndarray, depot: int = 0) -> Tuple[List[int], float]:
    """
    Clarke-Wright Savings algorithm adapted for TSP

    Args:
        distance_matrix: Matrix of distances between cities
        depot: Depot city index

    Returns:
        Tour and its total length
    """
    n = len(distance_matrix)

    # Calculate savings for all pairs
    savings = []
    for i in range(n):
        if i == depot:
            continue
        for j in range(i+1, n):
            if j == depot:
                continue
            s = distance_matrix[depot][i] + distance_matrix[depot][j] - distance_matrix[i][j]
            savings.append((s, i, j))

    # Sort by savings (descending)
    savings.sort(reverse=True)

    # Build tour by merging routes
    routes = [[depot, i, depot] for i in range(n) if i != depot]

    for s, i, j in savings:
        # Find routes containing i and j
        route_i = None
        route_j = None

        for route in routes:
            if i in route[1:-1]:  # Exclude depot positions
                route_i = route
            if j in route[1:-1]:
                route_j = route

        if route_i and route_j and route_i != route_j:
            # Merge routes
            # Remove depot from ends and concatenate
            merged = route_i[:-1] + route_j[1:]
            routes.remove(route_i)
            routes.remove(route_j)
            routes.append(merged)

            # If only one route remains, we have a complete tour
            if len(routes) == 1:
                break

    # Convert to tour (remove duplicate depot)
    if routes:
        tour = routes[0][:-1] if routes[0][-1] == depot else routes[0]
    else:
        tour = list(range(n))

    # Calculate length
    length = sum(distance_matrix[tour[i]][tour[(i+1) % n]] for i in range(n))

    return tour, length