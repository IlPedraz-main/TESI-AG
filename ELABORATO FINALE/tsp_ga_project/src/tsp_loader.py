"""
TSP Loader Module - Parser for TSPLIB files and distance calculation
"""

import numpy as np
import os
from typing import Tuple, List, Optional, Dict


class TSPInstance:
    """Class to represent a TSP instance"""

    def __init__(self, name: str, coords: np.ndarray,
                 dist_matrix: np.ndarray, optimal: Optional[float] = None):
        self.name = name
        self.coordinates = coords
        self.distance_matrix = dist_matrix
        self.optimal = optimal
        self.n_cities = len(coords)

    def evaluate_tour(self, tour: List[int]) -> float:
        """Calculate total tour length"""
        total = sum(self.distance_matrix[tour[i]][tour[i+1]]
                   for i in range(self.n_cities - 1))
        total += self.distance_matrix[tour[-1]][tour[0]]
        return total

    def gap_from_optimal(self, tour_length: float) -> Optional[float]:
        """Calculate percentage gap from optimal solution"""
        if self.optimal is None:
            return None
        return 100.0 * (tour_length - self.optimal) / self.optimal


def compute_euclidean_distances(coords: np.ndarray) -> np.ndarray:
    """Compute Euclidean distance matrix with TSPLIB rounding"""
    n = len(coords)
    D = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i+1, n):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            dist = int(np.sqrt(dx*dx + dy*dy) + 0.5)
            D[i][j] = D[j][i] = dist
    return D


def compute_att_distances(coords: np.ndarray) -> np.ndarray:
    """Compute ATT (pseudo-Euclidean) distance matrix"""
    n = len(coords)
    D = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i+1, n):
            xd = coords[i][0] - coords[j][0]
            yd = coords[i][1] - coords[j][1]
            rij = np.sqrt((xd*xd + yd*yd) / 10.0)
            tij = int(rij + 0.5)
            if tij < rij:
                dist = tij + 1
            else:
                dist = tij
            D[i][j] = D[j][i] = dist
    return D


def load_tsplib_instance(filepath: str, optimal_value: Optional[float] = None) -> TSPInstance:
    """Load a TSPLIB instance from file"""
    name, edge_type, dimension, optimal = None, None, None, optimal_value
    coordinates = []

    with open(filepath, 'r') as f:
        reading_coords = False
        for line in f:
            line = line.strip()

            if ':' in line and not reading_coords:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()

                if key == 'NAME':
                    name = value
                elif key == 'DIMENSION':
                    dimension = int(value)
                elif key == 'EDGE_WEIGHT_TYPE':
                    edge_type = value

            elif line == 'NODE_COORD_SECTION':
                reading_coords = True

            elif reading_coords and line and line != 'EOF':
                parts = line.split()
                if len(parts) >= 3:
                    x, y = float(parts[1]), float(parts[2])
                    coordinates.append([x, y])

    coords_array = np.array(coordinates)

    # Compute distance matrix based on edge type
    if edge_type == 'EUC_2D':
        D = compute_euclidean_distances(coords_array)
    elif edge_type == 'ATT':
        D = compute_att_distances(coords_array)
    else:
        raise ValueError(f"Unsupported edge weight type: {edge_type}")

    return TSPInstance(name, coords_array, D, optimal)


def download_tsplib_instances():
    """Download the required TSPLIB instances"""
    import urllib.request

    base_url = "http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/"
    instances = {
        'berlin52': ('berlin52.tsp', 7542),
        'att48': ('att48.tsp', 10628),
        'kroA100': ('kroA100.tsp', 21282)
    }

    data_dir = "data/tsplib"
    os.makedirs(data_dir, exist_ok=True)

    for name, (filename, optimal) in instances.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            try:
                url = base_url + filename
                urllib.request.urlretrieve(url, filepath)
                print(f"Downloaded {filename}")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
                print("Please download manually from TSPLIB95 website")

    return instances


def load_all_instances() -> Dict[str, TSPInstance]:
    """Load all required TSP instances"""
    instances_info = {
        'berlin52': ('data/tsplib/berlin52.tsp', 7542),
        'att48': ('data/tsplib/att48.tsp', 10628),
        'kroA100': ('data/tsplib/kroA100.tsp', 21282)
    }

    instances = {}
    for name, (filepath, optimal) in instances_info.items():
        if os.path.exists(filepath):
            instances[name] = load_tsplib_instance(filepath, optimal)
            print(f"Loaded {name}: {instances[name].n_cities} cities")
        else:
            print(f"Warning: {filepath} not found")

    return instances