# InvestAlloc: A Resource Allocation Optimization Framework

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

This project provides an investment optimization tool designed to maximize the efficiency of resource allocation across various investment areas. Utilizing numerical optimization techniques, the tool considers parameters such as efficiency factors (ROI), synergies between investments, budget constraints, and individual investment bounds. The project is implemented in Python, leveraging powerful libraries like `scipy`, `numpy`, and `pandas`.

## Table of Contents

- [Features](#features)
- [Objective Function Models](#objective-function-models)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Optimization Methods](#optimization-methods)
- [Analysis Capabilities](#analysis-capabilities)
- [Visualization](#visualization)
- [License](#license)
- [Contributions](#contributions)

## Features

- Optimizes investment allocation across multiple areas based on efficiency and synergy effects.
- Supports multiple optimization algorithms from `scipy.optimize`:
  - Sequential Least Squares Programming (SLSQP)
  - Differential Evolution (DE)
  - Basin Hopping
  - Truncated Newton Conjugate-Gradient (TNC)
- Performs sensitivity analysis to show how allocations change with varying budgets or parameters.
- Conducts robustness analysis to evaluate allocation stability under parameter uncertainty (with parallel processing support).
- Includes a framework for multi-criteria optimization considering custom risk and sustainability functions.
- Offers interactive (`plotly`) and static (`matplotlib`/`seaborn`) visualizations.
- Checks for required packages upon startup.
- Configurable via JSON file or command-line arguments.

## Objective Function Models

The optimizer aims to maximize a utility function composed of direct returns from investments and synergy effects. Two models for direct returns are supported:

1.  **Logarithmic (Default):** Maximizes `sum(roi_factors * log(x)) + Synergy`.
    * **Assumption:** Assumes diminishing marginal returns (each additional unit of investment yields less additional utility than the previous one). Common in economic modeling.
    * **Behavior:** Utility continuously increases with investment, but at a decreasing rate.
    * **Requirement:** Requires all `roi_factors` to be positive. Can be numerically sensitive if investments approach zero.

2.  **Quadratic (Optional, requires parameter `c`):** Maximizes `sum(roi_factors * x - 0.5 * c * x^2) + Synergy`.
    * **Assumption:** Also models diminishing marginal returns, but can additionally represent a saturation point or even a decrease in utility if investment becomes excessive (depending on `roi_factors` and `c`).
    * **Behavior:** Utility increases initially, potentially reaches a peak, and might decrease afterwards.
    * **Requirement:** Requires the parameter `c` (a list/array of non-negative values, one for each area) to be provided in the configuration. `c` controls the strength of the saturation effect.

The synergy effect is always calculated as `0.5 * x^T * synergy_matrix * x`.

## Installation

To install and run the Investment Optimizer, you need Python 3.7 or higher. Install the required Python packages using pip:

```sh
pip install -r requirements.txt
```

The necessary packages are:
- `numpy`
- `scipy`
- `matplotlib`
- `pandas`
- `seaborn`
- `plotly` (optional, for interactive plots)
- `packaging`

## Usage

The Investment Optimizer is typically run from the command line.

### Command-Line Arguments

- `--config FILE`: Path to a JSON configuration file (see [Configuration](#configuration)). If omitted, default parameters are used.
- `--interactive`: Use interactive visualizations with `plotly` instead of static `matplotlib`/`seaborn` plots (requires `plotly` to be installed).
- `--log LEVEL`: Set the logging level (e.g., `DEBUG`, `INFO`, `WARNING`). Default is `INFO`.

Example command:
```sh
python investment_optimizer.py --config config.json --interactive --log DEBUG
```

### Example Parameters (Defaults if no config file is provided)

- Investment areas (`investment_labels`): `['F&E', 'Marketing', 'Vertrieb', 'Kundenservice']`
- Efficiency parameters (`roi_factors`): `[1.5, 2.0, 2.5, 1.8]`
- Synergy matrix (`synergy_matrix`):
  ```
  [[0.0, 0.1, 0.15, 0.05],
   [0.1, 0.0, 0.2,  0.1 ],
   [0.15,0.2, 0.0,  0.25],
   [0.05,0.1, 0.25, 0.0 ]]
  ```
- Total budget (`total_budget`): `100.0`
- Minimum investments (`lower_bounds`): `[10.0, 10.0, 10.0, 10.0]`
- Maximum investments (`upper_bounds`): `[50.0, 50.0, 50.0, 50.0]`
- Quadratic utility parameters (`c`): `[0.01, 0.015, 0.02, 0.01]` (If `c` is present, quadratic utility is used; otherwise, logarithmic)
- Initial guess (`initial_allocation`): Automatically calculated as an equal distribution (`[25.0, 25.0, 25.0, 25.0]`) and adjusted to meet bounds/budget constraints.

## Configuration

Provide parameters via a JSON file using the `--config` argument. The keys should be:

- `investment_labels` (optional): List of names for the investment areas (string array).
- `roi_factors`: List or array of efficiency parameters for each area (float array). Must be > 0 if using the default logarithmic objective.
- `synergy_matrix`: 2D list or array representing the symmetric synergy matrix (float matrix). Diagonal elements should ideally be 0. Off-diagonal elements must be >= 0.
- `total_budget`: The total available budget (float).
- `lower_bounds`: List or array of minimum required investments for each area (float array). Must be >= 0.
- `upper_bounds`: List or array of maximum allowed investments for each area (float array). Must be >= `lower_bounds`.
- `initial_allocation`: Initial guess for the investment allocation (float array). The script will attempt to adjust this guess to meet bounds and budget constraints before starting optimization.
- `c` (optional): List or array of parameters for the quadratic utility function (float array). If provided, the quadratic model (`roi * x - 0.5 * c * x^2`) is used for the direct return part of the objective. If omitted or `null`, the logarithmic model (`roi * log(x)`) is used. Values must be >= 0.

See `config.json` for an example file structure.

## Optimization Methods

The Investment Optimizer interfaces with several `scipy.optimize` methods:

- **SLSQP (Sequential Least Squares Programming)**: Good for general constrained optimization problems. Often fast and reliable. Handles equality and inequality constraints.
- **Differential Evolution (DE)**: A global optimization algorithm, suitable for finding global optima in complex landscapes, potentially at the cost of more function evaluations. Robust to noisy objectives.
- **Basin Hopping**: Another global optimization method that combines random perturbation with local optimization (e.g., SLSQP) to escape local minima.
- **TNC (Truncated Newton Conjugate-Gradient)**: Efficient for unconstrained or bound-constrained problems, particularly with many variables. Uses a penalty approach for the budget constraint in this implementation.

The `optimize()` method in the `InvestmentOptimizer` class allows selecting the desired algorithm.

## Analysis Capabilities

Beyond basic optimization, the tool offers:

- **Sensitivity Analysis:** Evaluates how the optimal allocation and maximum utility change as the `total_budget` varies.
- **Robustness Analysis:** Simulates the optimization many times while randomly varying `roi_factors` and `synergy_matrix` (and `c` if applicable) within a specified percentage. Provides statistics (mean, std dev, quantiles, CV) on the resulting allocations to assess their stability.
- **Parameter Sensitivity:** Analyzes how the maximum utility changes when a single `roi_factor` or `synergy_matrix` element is varied systematically.
- **Top Synergy Identification:** Lists the pairs of investment areas with the highest positive synergy effects.
- **Multi-Criteria Optimization:** Provides a structure to optimize based on a weighted combination of utility, a custom risk function, and a custom sustainability function.

## Visualization

The script can generate several plots to help understand the results:

- **Synergy Heatmap**: Visualizes the `synergy_matrix`.
- **Sensitivity Analysis Plot**: Shows optimal allocations and total utility across different budget levels (static or interactive).
- **Robustness Analysis Plots**: Includes boxplots and histograms (or KDE plots) showing the distribution of allocations from the robustness simulations (static). A pairplot is generated if the number of areas is small (<=5).
- **Parameter Sensitivity Plot**: Displays the relationship between a changing parameter (ROI or synergy) and the resulting maximum utility (static or interactive).

Use the `--interactive` flag to enable `plotly`-based interactive plots where available.

## License

This project is licensed under the MIT License - see the LICENSE file (if available) or the license header for details.

## Contributions

Contributions are welcome! Please feel free to open an issue to report bugs or suggest features, or create a pull request for improvements to the code or documentation.