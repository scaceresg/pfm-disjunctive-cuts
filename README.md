# PFM Disjunctive Cuts

Permutation Flowshop Scheduling Problem with Makespan criterion (PFM) solver using disjunctive cutting planes built on a compact DOcplex/CPLEX model. The project provides algorithms for naive and cumulative 2n disjunctive cuts, further-improved cuts, and MIP solving after adding disjunctive cuts.

## Features

- Compact PFM MIP model in DOcplex/CPLEX
- Disjunctive-cut generation algorithms:
  - `2n_naive`: naive non-cumulative disjunctive cuts
  - `2n_cumulative`: cumulative job-based and position-based disjunctive cuts
  - `further_improve`: further strengthening of previously generated 2n cuts
  - `mip_disjunctive`: solve the MIP after adding disjunctive cuts

- CLI interface via `python -m pfm_disjunctive_cuts`
- Makefile workflow for setup, formatting, running, and building
- Instance loaders for both Taillard and Vallada benchmark sets

## Requirements

- Python 3.10+
- IBM CPLEX Optimization Studio with DOcplex (`docplex>=2.23.0`)
- NumPy and Pandas

Data files are expected under:

- `data/taillard_instances/` for Taillard instances
- `data/vallada_etal_instances/` for Vallada instances and `Vallada-bounds.csv`

## Quick Start

Clone and run from source (Windows PowerShell shown; similar on macOS/Linux):

```powershell
# Get the code
git clone https://github.com/scaceresg/pfm-disjunctive-cuts.git
cd pfm-disjunctive-cuts

# Create venv and install dev dependencies
make dev-setup

# Format code and imports (optional)
make fmt

# Run the default Taillard example
make run

# Or run the CLI directly
.venv\Scripts\python.exe -m pfm_disjunctive_cuts ^
    --instance tai20_5_1.txt ^
    --inst-type taillard ^
    --algorithm 2n_cumulative
```

On macOS/Linux, replace the last command with:

```bash
.venv/bin/python -m pfm_disjunctive_cuts \
    --instance tai20_5_1.txt \
    --inst-type taillard \
    --algorithm 2n_cumulative
```

## CLI Usage

The package exposes a single CLI entrypoint:

```text
--instance              Instance file name (e.g., tai20_5_1.txt) [required]
--inst-type             {taillard, vallada} [required]
--algorithm             {2n_cumulative, 2n_naive, further_improve, mip_disjunctive} [required]
--sorted-jobs           Sort jobs by workload before generating cuts
--decreasing-order      Sort jobs in decreasing order; requires --sorted-jobs
--position-cuts-first   Generate position-based cuts before job-based cuts
--sparser-cuts          Reduce cut coefficients by subtracting the minimum bound
--priority-order        {alpha, fractional}; valid only for mip_disjunctive
--show-output           Show solver output during optimization
--obj-diff-rounding     Objective rounding tolerance (default: 0.002)
--threads               Number of CPLEX threads (default: 16)
--mip-emphasis          CPLEX MIP emphasis (default: 0)
--time-limit            Time limit in seconds (default: 10800)
--no-cuts / --no-no-cuts
                        Enable or disable CPLEX cuts during MIP solving
```

The CLI prints a one-line execution summary before running and then reports the returned result dictionary in a compact format.

## Algorithms

Key algorithmic routines exposed in `PFMDisjunctiveCuts`:

- `run_2n_cumulative_disjunctive_cuts`
  - Builds the LP relaxation, computes lower bounds, and adds cumulative job-based and position-based cuts.
- `run_2n_naive_disjunctive_cuts`
  - Computes all disjunction lower bounds independently and adds naive 2n cuts.
- `run_further_improve_disj_cuts`
  - Starts from a 2n cuts approach and revisits `(j, k)` pairs to strengthen the generated cuts further.
- `run_mip_disjunctive_cuts`
  - Generates disjunctive cuts and then solves the MIP, optionally with variable priority ordering.

Underlying building blocks also include:

- `PFMproblem` for reading Taillard and Vallada instances
- `PFMmip` for building and solving the base PFM MIP formulation

## Makefile Highlights

Common targets (cross-platform):

- `make venv` — Create a virtual environment
- `make install` — Install the package in editable mode
- `make install-dev` — Install package plus development tools
- `make fmt` — Format code and imports
- `make run` — Run a Taillard example with `2n_cumulative`
- `make run-vallada` — Run a Vallada example with `2n_naive`
- `make run-further` — Run the further-improve algorithm on a Taillard instance
- `make run-mip` — Run MIP after disjunctive cuts on a Taillard instance
- `make build` — Build distributable packages
- `make clean` — Remove build artifacts
- `make clean-all` — Deep clean including the virtual environment

Run `make` with no arguments to view the categorized help output.

## Programmatic Usage

```python
from pfm_disjunctive_cuts import PFMDisjunctiveCuts

pfm = PFMDisjunctiveCuts(data_file="tai20_5_1.txt", inst_name="taillard")

results = pfm.run_2n_cumulative_disjunctive_cuts(
    sorted_jobs=True,
    decreasing_order=True,
    position_cuts_first=False,
    sparser_cuts=True,
)

print(results["lb_lp"], results["lb_n"], results["lb_2n"])
```

For MIP runs, user-defined CPLEX parameters can be set explicitly:

```python
from pfm_disjunctive_cuts import PFMDisjunctiveCuts

pfm = PFMDisjunctiveCuts(data_file="tai20_5_1.txt", inst_name="taillard")
pfm.set_cplex_params_user(
    show_output=False,
    obj_diff_rounding=0.002,
    n_threads=16,
    mip_emphasis=0,
    time_limit=10800,
    no_cuts=True,
)

results = pfm.run_mip_disjunctive_cuts(
    mip_priority_order=None,
    disj_cuts_approach="2n_cumulative",
    sorted_jobs=False,
    decreasing_order=False,
    position_cuts_first=False,
    sparser_cuts=True,
)

print(results["mip"])
```

## Data Notes

- Taillard instances are read from `data/taillard_instances/`.
- Vallada instances are read from `data/vallada_etal_instances/`.
- Vallada upper bounds are loaded from `Vallada-bounds.csv` and stored in `best`.
- The loaders validate the final processing-time matrix dimensions.

## Troubleshooting

- Ensure IBM CPLEX and DOcplex are installed and licensed before running solver-based routines.
- Instance names passed to the CLI must match filenames inside the relevant `data/` subdirectory.
- `--decreasing-order` must be used together with `--sorted-jobs`.
- `--priority-order` is valid only with `--algorithm mip_disjunctive`.

## License

MIT
