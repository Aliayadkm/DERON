# APR-ACO UAV Network Simulation

Python implementation of an Adaptive Pheromone-Based Communication Protocol (APR-ACO) for UAV networks in urban environments.

## Project contents

- `apr_aco_sim.py` — simulation implementation and experiment commands.
- `requirements.txt` — required Python packages.
- `results/` — generated figures and CSV result tables included with this project.

## Requirements

- Python 3.10 or later
- Packages listed in `requirements.txt`

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the simulation

Run commands from the project root directory.

Run sanity checks and the paper-number audit:

```bash
python apr_aco_sim.py checks
```

Run the main experiment:

```bash
python apr_aco_sim.py main --runs 30
```

Run an individual experiment:

```bash
python apr_aco_sim.py topology
python apr_aco_sim.py scalability
python apr_aco_sim.py ablation
python apr_aco_sim.py sensitivity
python apr_aco_sim.py lq
```

Run all experiment groups:

```bash
python apr_aco_sim.py all --workers 4
```

For a shorter smoke test:

```bash
python apr_aco_sim.py all --quick
```

The script supports additional options. View them with:

```bash
python apr_aco_sim.py --help
```

## Output

The simulation writes figures and CSV files to the `results/` directory by default. Use `--out DIRECTORY` to select another output directory.

## Notes

- The simulation code documents model assumptions in the `Params` class and its comments.
- The `urban` scenario is the default; the `paper` scenario can be selected with `--scenario paper`.
- Results already present in `results/` are included as supplied; rerunning experiments may overwrite files with matching names.

## Reproducibility

The script exposes random-seed and worker-count options (`--seed` and `--workers`). For repeatable experiments, record the command, seed, scenario, run count, and software environment used.

