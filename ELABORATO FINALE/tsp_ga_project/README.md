# TSP Genetic Algorithm - Thesis Implementation

Implementation of Genetic Algorithms for the Traveling Salesman Problem.
This code generates all experimental data and figures for Chapter 5 of the thesis.

## Project Structure

```
tsp_ga_project/
├── main.py                 # Main script to run all experiments
├── requirements.txt        # Python dependencies
├── src/                   # Source code modules
│   ├── tsp_loader.py      # TSPLIB parser and instance loader
│   ├── ga_engine.py       # Genetic Algorithm implementation
│   ├── ga_operators.py    # GA operators (crossover, mutation, selection)
│   ├── baseline.py        # Baseline algorithms (Nearest Neighbor)
│   ├── experiment_runner.py # Experiment management and batch execution
│   └── visualization.py   # Plot generation for thesis figures
├── data/                  # Data directory
│   └── tsplib/           # TSPLIB instance files
├── results/              # Experiment results (JSON)
│   ├── berlin52/
│   ├── att48/
│   └── kroA100/
├── plots/                # Generated figures
└── tables/               # Generated tables
```

## Installation

1. Install Python 3.8 or higher
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Download TSPLIB instances:
```bash
python main.py --download
```

### Run quick test (reduced parameters):
```bash
python main.py --test
```

### Run complete experiment suite (takes several hours):
```bash
python main.py --all
```

## Detailed Usage

### Run only standard experiments (Section 5.2):
```bash
python main.py --standard
```

### Run only sensitivity analyses (Section 5.4):
```bash
python main.py --sensitivity
```

### Generate figures from existing results (Section 5):
```bash
python main.py --visualize
```

## Experiments Overview

### Standard Experiments
- **Instances**: berlin52, att48, kroA100
- **Runs**: 30 independent runs per instance
- **GA Parameters**:
  - Population: 100
  - Generations: 500
  - Crossover rate: 0.8
  - Mutation rate: 0.1
  - Tournament size: 3
  - Elite size: 2

### Sensitivity Analyses
1. **Population Size** (berlin52): [50, 100, 150, 200]
2. **Number of Generations** (berlin52): [250, 500, 750, 1000]
3. **Mutation Rate** (kroA100): [0.05, 0.10, 0.15, 0.20]

### Generated Outputs

#### Figures (12 total):
- Fig 5.1-5.3: Convergence plots for each instance
- Fig 5.4-5.6: Tour comparisons (optimal vs GA-found)
- Fig 5.7: Box plot of gap distributions
- Fig 5.8: Population size sensitivity
- Fig 5.9: Mutation rate sensitivity
- Fig 5.10: Generation number sensitivity
- Fig 5.11: Scalability (computation time)
- Fig 5.12: Scalability (solution quality)

#### Tables (8 total):
- Tables 5.1-5.3: Results for each instance
- Table 5.4: Comparison with literature
- Tables 5.5-5.7: Sensitivity analysis results
- Table 5.8: Scalability summary

## Results Format

Results are saved in JSON format with the following structure:

```json
{
  "instance": "berlin52",
  "config": {
    "pop_size": 100,
    "n_generations": 500,
    ...
  },
  "optimal": 7542,
  "runs": [
    {
      "run_id": 0,
      "best_fitness": 7568,
      "gap_percent": 0.34,
      "time_seconds": 11.2,
      ...
    }
  ],
  "statistics": {
    "best_fitness": {...},
    "gap_percent": {...},
    ...
  }
}
```

## Performance Notes

- Full experiment suite: ~516 total runs
- Estimated time: 3-5 hours (depending on CPU)
- Memory usage: < 1 GB
- Disk space: < 100 MB for all results

## Algorithm Details

### GA Components:
- **Representation**: Permutation encoding
- **Selection**: Tournament selection (k=3)
- **Crossover**: Order Crossover (OX)
- **Mutation**: Inversion mutation
- **Replacement**: Elitism (top 2 individuals)

### Baseline Algorithm:
- Nearest Neighbor heuristic (greedy construction)

## Thesis Reference

This implementation corresponds to:
- **Chapter 4**: Implementation details
- **Chapter 5**: Experimental results and analysis

## License

This code is part of a thesis project. Academic use only.

## Author

Elaborato Finale - Università degli Studi di Modena e Reggio Emilia
Corso di Laurea in Ingegneria Informatica
Anno Accademico 2024/2025