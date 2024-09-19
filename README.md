# PyIRT_SDT: Python Item Response Theory and Signal Detection Theory Library

PyIRT_SDT is a comprehensive Python library for Item Response Theory (IRT) and Signal Detection Theory (SDT) estimation and analysis. It provides robust tools for modeling and analyzing participant response data using advanced psychometric techniques.

NOTE: This library is not designed for memory efficiency because it was used on large memory machines.

MOTIVATION: At the time when I developed this, there was no Python IRT library that could (1) handle sparse data, and (2) run on my system without errors. 
This library was primarily developed to be used as input parameters (in addition to primary data) for large predictive models on future student performance and to serve as an automated QA tool for generation of practice questions.

METHOD: 
# IRT
First, all IRT parameters are calculated (default uses a 4PL, but a 3PL is available) using scipy.optimize.curve_fit() (refer to scipy.optimize.least_squares for how the Jacobian is computed and solved) via an iterative process which seeks to maximize the fit over all items, for all participants, within the skill.
All IRT parameters and IRT model error variables (independent for each parameter within the 3PL or 4PL model) are captured once convergence or iteration cycles are reached. 4PL fits a logistic curve using 4 parameters: discriminability, difficulty, guessing level, inattention level.

# SDT
Finally, Signal Detection Theory (SDT) parameters are computed from the estimated participant thetas (estimated participant ability) and raw performance, including: Area Under Curve of Receiving Operating Characteristic (AUC_ROC), TPR, TNR, and Optimal Threshold (optimal criterion maximizing TNR and TPR). 

Note: We don't use IRT parameters to calculate the SDT parameters; only the estimated student abilties from the IRT process. With SDT, we are asking: How well does each question do in separating lower ability students from higher ability students? We answer that quantitatively with traditional SDT parameters, such as True Positive Rate, AUC, and Optimal Threshold (optimal criterion).

From my experience, AUC_ROC can be used as a SDT substitute for IRT Discirminability. 1 - Optimal Threshold can be used as a SDT substitute for IRT Difficulty. 

Several benefits of SDT include:
1) A normalized range (0 to 1) for difficulty making across skill comparisons easier and more intrepretable, for content teams.
2) More accurate when using these (instead of IRT parameters) to predict future performance inside of models.
3) Values more in line with ML approaches, making them intrepretable to MLEs and other users familiar with SDT.

This approach generates both types of parameters so that traditional psychometric teams have access to familiar parameters, while servicing SDT parameters for ML teams.

## Features

- Support for 3PL and 4PL IRT models
- Signal Detection Theory (SDT) analysis integration
- Parallel parameter estimation for improved performance
- Visualization tools for parameter convergence and item characteristic curves
- Flexible data input and output formats
- Applicable to various types of questions and response data, not limited to mathematical problems

## Installation

To install PyIRT_SDT, you can use pip:

```bash
pip install pyirt-sdt
```

Or clone the repository and install from source:

```bash
git clone https://github.com/yourusername/pyirt-sdt.git
cd pyirt-sdt
pip install -e .
```

## Quick Start

Here's a simple example of how to use PyIRT_SDT:

```python
import pandas as pd
from pyirt_sdt import returnTable, solve_IRT_for_matrix

# Load your data
df = pd.read_csv('your_data.csv')

# Prepare the data
table = returnTable(df)

# Solve the IRT model and perform SDT analysis
results = solve_IRT_for_matrix(table, FOUR_PL=True)

# Access the results
print(results.thetas)  # Participant ability estimates
print(results.est_params)  # Item parameter estimates
print(results.sdt_results)  # SDT analysis results

# Plot parameter convergence
results.plot_parameter_convergence()

# Export results to CSV
results.export_to_csv('irt_sdt_results.csv')
```

## Data Format

Your input CSV should have the following columns:
- `participant_id`: Unique identifier for each participant
- `item_id`: Unique identifier for each item or question
- `response`: The participant's response or score for the item

## Documentation

For detailed documentation, please refer to the [docs](docs/) directory or visit our [Read the Docs](https://pyirt-sdt.readthedocs.io/) page.

## Command-line Interface

PyIRT_SDT provides a command-line interface for quick analysis:

```bash
pyirt-sdt analyze --input your_data.csv --output results.csv --model 4pl
```

For more options, run:

```bash
pyirt-sdt --help
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for more details.

## License

PyIRT_SDT is released under the [MIT License](LICENSE).

## Contact

For any questions or issues, please open an issue on GitHub or contact [your email].
