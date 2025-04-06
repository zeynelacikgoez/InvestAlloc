# InvestAlloc: A Resource Allocation Optimization Framework

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

This project provides an investment optimization tool designed to maximize the efficiency of resource allocation across various investment areas. Utilizing numerical optimization techniques, the tool considers parameters such as efficiency factors (ROI), synergies between investments, budget constraints, and individual investment bounds. The project is implemented in Python, leveraging powerful libraries like `scipy`, `numpy`, and `pandas`.

## Table of Contents

- [Quick Start](#quick-start)
- [Features](#features)
- [Objective Function Models](#objective-function-models)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Optimization Methods](#optimization-methods)
- [Analysis Capabilities](#analysis-capabilities)
- [Multi-Criteria Optimization and Pareto Approximation](#multi-criteria-optimization-and-pareto-approximation)
- [Visualization](#visualization)
- [License](#license)
- [Contributions](#contributions)

## Quick Start

1.  **Install:** Ensure Python >= 3.7 and install requirements:
    ```sh
    pip install -r requirements.txt
    ```
2.  **Configure:** Edit `config.json` to match your investment scenario (ROI, budget, bounds, utility function type, S-curve params, analysis params, etc.).
3.  **Run:** Execute the optimizer:
    ```sh
    python investment_optimizer.py --config config.json
    ```
    (Add `--interactive` for interactive plots if `plotly` is installed).
4.  **Analyze:** Review the console output and the generated plots showing single-objective results, sensitivity, robustness, top synergies, and Pareto approximation.

## Features

- Optimizes investment allocation across multiple areas based on efficiency and synergy effects.
- Supports multiple utility function models: Logarithmic (default), Quadratic, S-Curve (logistic).
- Supports multiple optimization algorithms from `scipy.optimize`:
  - Sequential Least Squares Programming (SLSQP)
  - Differential Evolution (DE) (with native constraint handling if available)
  - Basin Hopping
  - Truncated Newton Conjugate-Gradient (TNC)
  - L-BFGS-B
  - Trust-Region Constrained (trust-constr)
  - Nelder-Mead
- Performs sensitivity analysis to show how allocations change with varying budgets or parameters.
- Conducts robustness analysis to evaluate allocation stability under parameter uncertainty (with parallel processing support).
- Includes a framework for multi-criteria optimization using weighted sums.
- Approximates the Pareto-optimal front for multiple objectives (Utility, Risk, Sustainability) using weighted sum iteration.
- Offers interactive (`plotly`) and static (`matplotlib`/`seaborn`) visualizations for various analyses, including Pareto fronts.
- Checks for required packages upon startup.
- Highly configurable via JSON file (optimization parameters, utility function selection, analysis settings) or command-line arguments.

## Objective Function Models

The optimizer aims to maximize a utility function composed of direct returns from investments and synergy effects. Three models for direct returns are supported, selected via the `utility_function_type` parameter in the configuration:

1.  **`'log'` (Default):** Maximizes `sum(roi_factors * log(x)) + Synergy`.
    * **Assumption:** Diminishing marginal returns. Standard economic model.
    * **Requirement:** Requires `roi_factors` > small positive constant (`MIN_INVESTMENT`). Ignores `c` and S-curve parameters.

2.  **`'quadratic'`:** Maximizes `sum(roi_factors * x - 0.5 * c * x^2) + Synergy`.
    * **Assumption:** Models diminishing returns and potential saturation/oversaturation.
    * **Requirement:** Requires `roi_factors` and the parameter `c` (non-negative array). Ignores S-curve parameters.

3.  **`'s_curve'`:** Maximizes `sum(L / (1 + exp(-k * (x - x0)))) + Synergy`.
    * **Assumption:** Models growth with saturation, following a logistic curve shape. Useful for modeling adoption, market share, or learning effects.
    * **Requirement:** Requires parameters `s_curve_L` (max value, >0 array), `s_curve_k` (steepness, >0 array), and `s_curve_x0` (midpoint/inflection point array). Ignores `roi_factors` (which only define dimension `n`) and `c`.

The synergy effect is always calculated as `0.5 * x^T * synergy_matrix * x`.

## Installation

(See Quick Start or below)

To install and run the Investment Optimizer, you need Python 3.7 or higher. Install the required Python packages using pip:

```sh
pip install -r requirements.txt
```

The necessary packages are:
- `numpy`
- `scipy` (Version >= ~1.7 recommended for full DE constraint handling)
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

### Default Parameters (if no config file is provided)

(Example values, see `main` function for exact defaults)
- Investment areas: `['F&E', 'Marketing', 'Vertrieb', 'Kundenservice']`
- `utility_function_type`: `'log'` (Logarithmic)
- `roi_factors`: `[1.5, 2.0, 2.5, 1.8]` (Used for log/quadratic)
- `synergy_matrix`: (Example 4x4 matrix)
- `total_budget`: `100.0`
- `lower_bounds`: `[10.0, 10.0, 10.0, 10.0]`
- `upper_bounds`: `[50.0, 50.0, 50.0, 50.0]`
- `c`: `None` (Not used by default log utility)
- `s_curve_L`, `s_curve_k`, `s_curve_x0`: `None` (Not used by default log utility)
- `initial_allocation`: Equal distribution, adjusted to bounds/budget.

## Configuration

Provide parameters via a JSON file using the `--config` argument.

### Core Parameters:

- `investment_labels` (optional): List of names for the investment areas (string array). Length `n`.
- `roi_factors`: Efficiency/scale parameters (float array, length `n`). Used by `'log'` and `'quadratic'`. Must be > `MIN_INVESTMENT` if using log objective.
- `synergy_matrix`: Symmetric synergy matrix (`n x n` float matrix, >= 0 off-diagonal).
- `total_budget`: Total available budget (float).
- `lower_bounds`: Minimum investments per area (float array, length `n`, >= 0).
- `upper_bounds`: Maximum investments per area (float array, length `n`, >= `lower_bounds`).
- `initial_allocation`: Initial guess for optimization (float array, length `n`). Adjusted by script.

### Utility Function Parameters:

- `utility_function_type` (optional): Selects the utility function model. Options: `"log"` (default), `"quadratic"`, `"s_curve"`.
- `c` (optional): Parameters for `'quadratic'` utility (float array, length `n`, >= 0). Required if `utility_function_type` is `"quadratic"`. Set to `null` or omit otherwise.
- `s_curve_L` (optional): Max value parameters for `'s_curve'` (float array, length `n`, > 0). Required if `utility_function_type` is `"s_curve"`. Set to `null` or omit otherwise.
- `s_curve_k` (optional): Steepness parameters for `'s_curve'` (float array, length `n`, > 0). Required if `utility_function_type` is `"s_curve"`. Set to `null` or omit otherwise.
- `s_curve_x0` (optional): Midpoint parameters for `'s_curve'` (float array, length `n`). Required if `utility_function_type` is `"s_curve"`. Set to `null` or omit otherwise.

### Analysis Parameters (Optional Block):

Add an optional `"analysis_parameters"` object to the JSON to control analysis details:

```json
"analysis_parameters": {
    "sensitivity_num_steps": 15,            # Number of budget steps in sensitivity analysis
    "sensitivity_budget_min_factor": 0.7,   # Min budget = factor * total_budget
    "sensitivity_budget_max_factor": 1.5,   # Max budget = factor * total_budget

    "robustness_num_simulations": 200,      # Number of simulations for robustness
    "robustness_variation_percentage": 0.15,# +/- variation percentage for parameters
    "robustness_num_workers_factor": 4,     # Max workers = min(CPUs, factor)

    "parameter_sensitivity_num_steps": 11,  # Number of steps for parameter sensitivity
    "parameter_sensitivity_min_factor": 0.5,# Min value = factor * current_value (or 0)
    "parameter_sensitivity_max_factor": 1.5,# Max value = factor * current_value

    "de_maxiter": 1500,                     # Max iterations for Differential Evolution
    "de_popsize": 20,                       # Population size for Differential Evolution

    "multi_criteria_weights": {             # Weights for single multi-criteria example run
        "alpha": 0.6,                       # Utility weight
        "beta": 0.2,                        # Risk weight
        "gamma": 0.2                        # Sustainability weight
    },
    "pareto_num_samples": 50                # Approx. samples for Pareto approximation
}
```
If this block or individual parameters within it are omitted, the default values specified in the script will be used.

See `config.json` for a full example file structure.

## Optimization Methods

The Investment Optimizer interfaces with several `scipy.optimize` methods:

- **SLSQP (Sequential Least Squares Programming)**: Good for general constrained optimization problems. Often fast and reliable. Handles equality and inequality constraints directly.
- **trust-constr (Trust-Region Constrained)**: Modern trust-region method suitable for constrained optimization. Handles bounds and constraints directly.
- **Differential Evolution (DE)**: A global optimization algorithm, suitable for finding global optima in complex landscapes. Robust to noisy objectives. Uses SciPy's built-in constraint handling for the budget constraint if available (SciPy >= ~1.7), otherwise falls back to a penalty approach.
- **Basin Hopping**: Another global optimization method combining random perturbation with local optimization (e.g., SLSQP) to escape local minima. The underlying local optimizer handles constraints.
- **L-BFGS-B**: Efficient quasi-Newton method for bound-constrained problems. Uses a penalty approach for the budget constraint in this implementation.
- **TNC (Truncated Newton Conjugate-Gradient)**: Efficient for bound-constrained problems. Uses a penalty approach for the budget constraint in this implementation.
- **Nelder-Mead**: Simplex-based method, does not require gradient information. Robust but potentially slow and less precise. Uses a penalty approach for the budget constraint and bounds handling can be weak.

The `optimize()` method in the `InvestmentOptimizer` class allows selecting the desired algorithm. The implementation automatically chooses whether to use penalties based on the method's capabilities and performs constraint checks after optimization for robustness.

## Analysis Capabilities

Beyond basic optimization, the tool offers:

- **Sensitivity Analysis:** Evaluates how the optimal allocation and maximum utility change as the `total_budget` varies (range and steps configurable).
- **Robustness Analysis:** Simulates the optimization many times while randomly varying core parameters (`roi_factors`, `synergy_matrix`, `c`, or S-curve params based on selected utility function) within a specified percentage (simulations count, variation % configurable). Provides statistics (mean, std dev, quantiles, CV) on the resulting allocations to assess their stability.
- **Parameter Sensitivity:** Analyzes how the maximum utility changes when a single parameter (`roi_factor`, `synergy_matrix` element, `c` element) is varied systematically (range and steps configurable).
- **Top Synergy Identification:** Lists the pairs of investment areas with the highest positive synergy effects.
- **Multi-Criteria Optimization (Example):** Provides a structure to optimize based on a weighted combination of utility, a custom risk function, and a custom sustainability function (weights configurable).
- **Pareto Front Approximation:** Approximates the set of non-dominated solutions (Pareto front) considering Utility, Risk, and Sustainability objectives using weighted sum iteration (number of samples configurable).

## Multi-Criteria Optimization and Pareto Approximation

The script includes a basic framework for multi-criteria optimization using a weighted sum approach (`multi_criteria_optimization` method). You provide functions for risk and sustainability (examples in `main`) and weights (alpha, beta, gamma) in the configuration's `analysis_parameters.multi_criteria_weights`. This finds a *single* solution based on the chosen weights.

Additionally, the `find_pareto_approximation` method attempts to find a *set* of non-dominated solutions representing the trade-offs between Utility, Risk, and Sustainability. It does this by running the weighted-sum optimization multiple times with different weights.

**Usage:**
1.  Define your `risk_func` and `sustainability_func` (examples provided in `main`).
2.  The script will automatically run `find_pareto_approximation` after the single optimizations.
3.  Configure the approximate number of weight combinations to sample via `"pareto_num_samples"` in the `analysis_parameters` section of `config.json`.
4.  The results (a set of non-dominated allocations and their objective values) are printed and visualized (as 2D projections or an interactive 3D scatter plot). This helps understand the available trade-offs, e.g., how much risk must be accepted to achieve a certain utility level.

**Limitations:**
* The weighted sum iteration primarily finds points on the *convex* hull of the true Pareto front. It may miss solutions in non-convex regions.
* It's an approximation, not a guaranteed computation of the full, true Pareto front. For that, specialized multi-objective algorithms (e.g., NSGA-II from libraries like `pymoo`) would be needed, which are currently not implemented.

## Visualization

The script can generate several plots to help understand the results:

- **Synergy Heatmap**: Visualizes the `synergy_matrix`.
- **Sensitivity Analysis Plot**: Shows optimal allocations and total utility across different budget levels (static or interactive).
- **Robustness Analysis Plots**: Includes boxplots and histograms (or KDE plots) showing the distribution of allocations from the robustness simulations (static). A pairplot is generated if the number of areas is small (<=5).
- **Parameter Sensitivity Plot**: Displays the relationship between a changing parameter (ROI, synergy, or c) and the resulting maximum utility (static or interactive).
- **Pareto Approximation Plot**: Shows the approximated non-dominated front in 2D projections or an interactive 3D plot (Utility vs. Risk vs. Sustainability).

Use the `--interactive` flag to enable `plotly`-based interactive plots where available. Static plots use `matplotlib` and `seaborn`.

## License

This project is licensed under the MIT License.

## Contributions

Contributions are welcome! Please feel free to open an issue to report bugs or suggest features, or create a pull request for improvements to the code or documentation.