# InvestAlloc a Resource Allocation Framework

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Configuration](#configuration)
  - [Running Analyses](#running-analyses)
    - [Budget Optimization](#budget-optimization)
    - [Sensitivity Analysis](#sensitivity-analysis)
    - [Robustness Analysis](#robustness-analysis)
- [Examples](#examples)
- [Configuration File](#configuration-file)
- [Plotting](#plotting)
- [Logging](#logging)
- [License](#license)
- [Contributing](#contributing)
- [Contact](#contact)

## Overview

The **Resource Allocation Framework** is a Python-based tool designed to optimize the allocation of resources under various constraints. It offers functionalities for budget optimization, sensitivity analysis, and robustness analysis, making it suitable for businesses and researchers aiming to maximize utility while effectively managing resources.

## Features

- **Budget Optimization**: Optimize resource allocations to maximize utility under a given budget.
- **Sensitivity Analysis**: Analyze how changes in budget affect optimal allocations and utility.
- **Robustness Analysis**: Assess the stability of allocations under parameter perturbations.
- **Configurable**: Easily customize parameters via configuration files.
- **Visualization**: Generate plots to visualize sensitivity analysis results.
- **Logging**: Provides informative logging to track the execution process.

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/zeynel/InvestAlloc.git
   cd InvestAlloc
   ```

2. **Create a Virtual Environment (Optional but Recommended)**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   *If `requirements.txt` is not provided, you can install the necessary packages manually:*

   ```bash
   pip install numpy scipy matplotlib pandas
   ```

## Usage

The framework is executed via the command line and supports three types of analyses:

1. **Budget Optimization**
2. **Sensitivity Analysis**
3. **Robustness Analysis**

### Configuration

You can provide a configuration file in JSON format to specify parameters for the analyses. If no configuration file is provided, the framework uses default settings.

**Sample Configuration (`config.json`):**

```json
{
    "a": [1, 2, 3, 4],
    "b": [
        [0, 0.1, 0.2, 0.3],
        [0, 0, 0.4, 0.5],
        [0, 0, 0, 0.6],
        [0, 0, 0, 0]
    ],
    "B": 10,
    "L": [1, 1, 1, 1],
    "U": [5, 5, 5, 5],
    "x0": [2, 2, 2, 4],
    "method": "SLSQP",
    "resource_names": ["R&D", "Marketing", "Sales", "Customer Service"],
    "penalty_factor": 1000000
}
```

### Running Analyses

Use the `main.py` script to perform analyses. The general syntax is:

```bash
python main.py ANALYSIS_TYPE [OPTIONS]
```

#### Budget Optimization

Optimize resource allocations to maximize utility within a budget.

**Command:**

```bash
python main.py budget_optimization --config config.json
```

**Options:**

- `--config`: Path to the configuration file.
- `--save_config`: Path to save the default configuration file.

**Example Output:**

```
Optimal Allocation: [x1, x2, x3, x4]
Maximum Utility: 12.345
```

#### Sensitivity Analysis

Analyze how different budget values affect the optimal allocations and utility.

**Command:**

```bash
python main.py sensitivity_analysis --config config.json
```

**Options:**

- `--config`: Path to the configuration file.
- `--save_config`: Path to save the default configuration file.

**Example Output:**

```
Sensitivity analysis plot saved as 'sensitivity_analysis.png'
```

#### Robustness Analysis

Assess the stability of resource allocations under parameter perturbations.

**Command:**

```bash
python main.py robustness_analysis --config config.json
```

**Options:**

- `--config`: Path to the configuration file.
- `--save_config`: Path to save the default configuration file.

**Example Output:**

```
Robustness Analysis Results:
         R&D  Marketing    Sales  Customer Service
count  100.0       100.0    100.0              100.0
mean     ...         ...      ...                ...
std      ...         ...      ...                ...
min      ...         ...      ...                ...
25%      ...         ...      ...                ...
50%      ...         ...      ...                ...
75%      ...         ...      ...                ...
max      ...         ...      ...                ...
```

## Examples

### Running Budget Optimization with Default Settings

```bash
python main.py budget_optimization
```

### Saving the Default Configuration

```bash
python main.py budget_optimization --save_config default_config.json
```

### Running Sensitivity Analysis with a Custom Configuration

```bash
python main.py sensitivity_analysis --config my_config.json
```

## Configuration File

The configuration file is a JSON file that specifies the parameters for the analyses. Below are the parameters you can set:

- `a` (list): Linear utility coefficients.
- `b` (list of lists): Quadratic utility coefficients matrix (upper triangular).
- `B` (float): Total budget.
- `L` (list): Minimum investments for each resource.
- `U` (list): Maximum investments for each resource.
- `x0` (list): Initial guess for investments.
- `method` (str): Optimization method (`SLSQP` or `DE`).
- `resource_names` (list): Names of the resources.
- `penalty_factor` (float): Penalty factor for the Differential Evolution method.

**Example:**

```json
{
    "a": [1, 2, 3, 4],
    "b": [
        [0, 0.1, 0.2, 0.3],
        [0, 0, 0.4, 0.5],
        [0, 0, 0, 0.6],
        [0, 0, 0, 0]
    ],
    "B": 10,
    "L": [1, 1, 1, 1],
    "U": [5, 5, 5, 5],
    "x0": [2, 2, 2, 4],
    "method": "SLSQP",
    "resource_names": ["R&D", "Marketing", "Sales", "Customer Service"],
    "penalty_factor": 1000000
}
```

## Plotting

When performing sensitivity analysis, the framework generates a plot showing how optimal investments for each resource change with varying budgets.

**Default Plot File:** `sensitivity_analysis.png`

You can customize the filename via the configuration or modify the `main.py` script as needed.

## Logging

The framework uses Python's `logging` module to provide informative messages during execution. By default, the logging level is set to `INFO`.

**Example Logs:**

```
INFO: Configuration loaded from config.json.
INFO: Sensitivity analysis plot saved as 'sensitivity_analysis.png'.
WARNING: Optimization failed for B=15: ...
ERROR: Simulation 5: Error during optimization: ...
```

## License

This project is licensed under the [MIT License](LICENSE).

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the Repository**

2. **Create a New Branch**

   ```bash
   git checkout -b feature/YourFeature
   ```

3. **Commit Your Changes**

   ```bash
   git commit -m "Add some feature"
   ```

4. **Push to the Branch**

   ```bash
   git push origin feature/YourFeature
   ```

5. **Open a Pull Request**

Please ensure your code follows the project's coding standards and includes appropriate tests.
