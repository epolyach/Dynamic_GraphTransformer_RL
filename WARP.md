# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Common Commands

### Environment Setup

1.  **Set up Python virtual environment (one-time):**
    ```bash
    ./setup_venv.sh
    ```

2.  **Activate the environment (each session):**
    ```bash
    source activate_env.sh
    ```

### Training

-   **Train a single model:**
    ```bash
    cd training_cpu/scripts
    python run_training.py --model <model_name> --config ../../configs/<config_file>.yaml
    ```
    -   `<model_name>` can be `GAT+RL`, `GT+RL`, `DGT+RL`, or `GT-Greedy`.
    -   `<config_file>` can be `tiny`, `small`, `medium`, or `production`.

-   **Train all models:**
    ```bash
    cd training_cpu/scripts
    python run_training.py --all --config ../../configs/<config_file>.yaml
    ```

-   **Generate comparison plots from saved results:**
    ```bash
    cd training_cpu/scripts
    python make_comparative_plot.py --config ../../configs/<config_file>.yaml
    ```

### CPU Benchmarking

-   **Run the unified CPU benchmark:**
    ```bash
    cd benchmark_cpu/scripts
    python run_exact.py --config ../../configs/small.yaml --n-start 5 --n-end 12 --instances 20 --time-limit 5 --csv ../results/csv/cpu_benchmark.csv
    ```

-   **Plot CPU benchmark results:**
    ```bash
    cd benchmark_cpu/scripts
    python plot_cpu_benchmark.py --csv ../results/csv/cpu_benchmark.csv --output ../plots/cpu_benchmark.png
    ```

### GPU Benchmarking

-   **Run high-precision GPU benchmark (10,000 instances):**
    ```bash
    cd benchmark_gpu/scripts
    python benchmark_gpu_10k.py
    ```

-   **Run adaptive multi-N GPU benchmark:**
    ```bash
    cd benchmark_gpu/scripts
    python benchmark_gpu_adaptive_n.py
    ```

-   **Plot GPU benchmark results:**
    ```bash
    cd benchmark_gpu/scripts
    python plot_gpu_benchmark.py --csv ../results/csv/gpu_benchmark_results.csv
    ```

-   **Plot CPU vs. GPU comparison:**
    ```bash
    cd benchmark_gpu/scripts
    python plot_cpu_gpu_comparison.py --cpu-csv ../../benchmark_cpu/results/csv/cpu_benchmark.csv --gpu-csv ../results/csv/gpu_benchmark_results.csv
    ```

## Code Architecture

This repository is structured to separate concerns between core logic, training, and benchmarking.

-   `src/`: Contains the core, shared source code.
    -   `src/generator/generator.py`: The single, strict CVRP instance generator used for all tasks to ensure consistency.
    -   `src/models/`: Contains the implementations of the different models (GAT, GT, DGT). `model_factory.py` is used to create model instances.
    -   `src/eval/validation.py`: Implements strict validation of CVRP solutions.
    -   `src/benchmarking/solvers/`: Contains CPU and GPU solver implementations.
    -   `src/utils/`: Shared utilities for configuration loading (`config.py`) and seeding for reproducibility (`seeding.py`).

-   `configs/`: Holds all YAML configuration files. There's a `default.yaml` with all possible parameters, and other files (`tiny.yaml`, `small.yaml`, etc.) override the defaults for different scenarios.

-   `training_cpu/`: Contains everything related to model training on the CPU.
    -   `scripts/`: Holds the scripts to run training and plotting.
    -   `lib/`: Contains the training library, including the main RL training logic in `advanced_trainer.py`.
    -   `results/`: Where training outputs (models, CSVs, plots) are saved locally.

-   `benchmark_cpu/`: CPU-specific benchmarking scripts and results. It has a similar structure to `training_cpu/`.

-   `benchmark_gpu/`: GPU-specific benchmarking scripts and results.

This modular design ensures that training and benchmarking are isolated, using the core components from `src/`. All parameters are controlled via YAML files to ensure reproducibility.

