# InvestAlloc a Resource Allocation Framework

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

This project is an investment optimization tool that maximizes the efficiency of resource allocation across various investment areas. Using numerical optimization techniques, the tool takes into account different parameters such as efficiency, synergies between investments, budget constraints, and bounds for each investment. The project is written in Python and utilizes powerful optimization algorithms available in libraries like `scipy` and `numpy`.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Optimization Methods](#optimization-methods)
- [Visualization](#visualization)
- [License](#license)

## Features

- Optimizes investment allocation across multiple areas given efficiency and synergy effects.
- Supports multiple optimization methods including:
  - Sequential Least Squares Programming (SLSQP)
  - Differential Evolution (DE)
  - Basin Hopping
  - Truncated Newton Conjugate-Gradient (TNC)
- Sensitivity analysis of optimal allocations based on changing budget or investment parameters.
- Robustness analysis to evaluate the stability of allocations under uncertainty.
- Interactive and static visualizations of sensitivity, robustness, and synergy effects.

## Installation

To install and run the Investment Optimizer, you need Python 3.6 or higher and the following Python packages:

- `numpy`
- `scipy`
- `matplotlib`
- `pandas`
- `seaborn`
- `plotly`
- `logging`

To install the required packages, you can run:

```sh
pip install -r requirements.txt
```

## Usage

The Investment Optimizer can be used via command line. It reads configuration parameters from a JSON file or uses default values.

### Command-Line Arguments

- `--config`: Path to a JSON configuration file that provides parameters such as efficiency (`a`), synergy effects (`b`), budget (`B`), lower bounds (`L`), upper bounds (`U`), and initial guess (`x0`).
- `--interactive`: If provided, will use interactive visualizations (`plotly`).

To run the script, use:

```sh
python investment_optimizer.py --config config.json --interactive
```

### Example Parameters

If no configuration file is provided, the following default parameters are used:

- Investment areas: `['F&E', 'Marketing', 'Vertrieb', 'Kundenservice']`
- Efficiency parameters: `[1, 2, 3, 4]`
- Synergy matrix:
  ```
  [[0, 0.1, 0.2, 0.3],
   [0.1, 0, 0.4, 0.5],
   [0.2, 0.4, 0, 0.6],
   [0.3, 0.5, 0.6, 0]]
  ```
- Total budget (`B`): `10.0`
- Minimum investments (`L`): `[1, 1, 1, 1]`
- Maximum investments (`U`): `[5, 5, 5, 5]`
- Initial guess (`x0`): `[2, 2, 2, 4]`

## Configuration

You can provide a configuration file in JSON format with the following keys:

- `a`: List of efficiency parameters.
- `b`: 2D list representing the synergy matrix (must be symmetric).
- `B`: Total budget.
- `L`: List of minimum investments for each area.
- `U`: List of maximum investments for each area.
- `x0`: Initial investment allocation guess.
- `investment_labels` (optional): List of names for the investment areas.

## Optimization Methods

The Investment Optimizer provides several optimization methods:

- **SLSQP (Sequential Least Squares Programming)**: Suitable for constrained optimization.
- **Differential Evolution (DE)**: Useful for global optimization and finding the global minimum.
- **Basin Hopping**: Combines local optimization with random sampling to find the global minimum.
- **TNC (Truncated Newton Conjugate-Gradient)**: An efficient optimization method for large-scale problems.

The optimization can be started with different methods using the `optimize()` function in the `InvestmentOptimizer` class.

## Visualization

The project includes the following visualization capabilities:

- **Synergy Heatmap**: Shows the synergy between different investment areas.
- **Sensitivity Analysis**: Evaluates how varying the total budget affects optimal allocations.
- **Parameter Sensitivity Analysis**: Examines the effect of changing an individual parameter (`a` or `b`) on the overall allocation.
- **Robustness Analysis**: Visualizes the stability of allocations when the input parameters are varied randomly.
- **Interactive Dashboard**: Uses `plotly` to create interactive plots for a more detailed exploration of results.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to clone, modify, and use this project for your investment optimization tasks.

### Contributions

Contributions are welcome! Please open an issue or create a pull request for new features, bug fixes, or documentation improvements.


