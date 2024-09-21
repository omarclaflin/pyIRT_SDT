# PyIRT_SDT: Python Item Response Theory and Signal Detection Theory Library

PyIRT_SDT is a comprehensive Python library for Item Response Theory (IRT) and Signal Detection Theory (SDT) estimation and analysis. It provides robust tools for modeling and analyzing participant response data using advanced psychometric techniques.

NOTE: This library was not designed with memory efficiency in mind because it was developed on large memory instances.

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

## Motivation: ## 
At the time when I developed this, there was no Python IRT library that could (1) handle sparse data, and (2) run on my system without errors. 
This library was primarily developed to generate IRT parameters from student performance data, to be used as input parameters (in addition to other primary data) for large predictive models on future student performance and to serve as an automated QA tool for generation of practice questions. It was expanded to try to include more engineered features (SDT parameters, modelling error, etc), as well, as some validation tools to confirm our approach.

# Approach/Method: 

**IRT**

First, all IRT parameters are calculated (default uses a 4PL, but a 3PL is available) using scipy.optimize.curve_fit() (refer to scipy.optimize.least_squares for how the Jacobian is computed and solved) via an iterative process which seeks to maximize the fit over all items, for all participants, within the skill.
All IRT parameters and IRT model error variables (independent for each parameter within the 3PL or 4PL model) are captured once convergence or iteration cycles are reached. Default is a 4PL which fits a logistic curve using 4 parameters: discriminability, difficulty, guessing level, inattention level. For comprehensiveness, total modelling fit error is captured for each item, along with modelling error associated with each parameter.

**SDT**

Secondly, Signal Detection Theory (SDT) parameters are computed from the estimated participant thetas (estimated participant ability from IRT) and raw performance, including: Area Under Curve of Receiving Operating Characteristic (AUC_ROC), TPR, TNR, and Optimal Threshold (the criterion on the x-axis of the ROC curve that maximizes TNR and TPR). 

Note: We don't use IRT parameters to calculate the SDT parameters; only the estimated student abilties from the IRT process. With SDT, we are asking: *How well does each question do in separating lower ability students from higher ability students?* We answer that quantitatively with traditional SDT parameters, such as True Positive Rate, AUC, and Optimal Threshold (optimal criterion).

From my experience: 
1) AUC_ROC can be used as a SDT substitute for IRT Discirminability. 
2) 1 - Optimal Threshold can be used as a SDT substitute for IRT Difficulty. 
3) Optimal Threshold, TPR, TNR seem to provide the most predictability in models trying to predict future student performance, over IRT parameters, from the large datasets I've worked on.

Several benefits of SDT include:
1) A normalized range (0 to 1) for difficulty making across skill comparisons easier and more intrepretable, for content teams.
2) More accurate when using these (instead of IRT parameters) to predict future performance inside of models.
3) Values more in line with ML approaches, making them intrepretable to MLEs, engineers, and other users familiar with SDT.
4) Note, we still rely on the logistical modelling of IRT to sort the students. In the future, we can add 2PL, 1PL models as well.

This approach generates both types of parameters so that traditional psychometric teams have access to familiar parameters, while servicing SDT parameters for ML teams.

Validation Approaches (using data and the tools in this package; not pedagogical validity):
1) Convergence plots are available to view. Delta thetas (after each iterative run) -- we expect this to shrink over time.
2) Eyeball visualization of raw performance data against modelled IRT curve, for each item.
3) Information plots for each item, and the entire skill/test (traditionally, an "ICC"), along with a distribution of student theta plot.
4) Correlation plots that show a strong but not overwhelming relationship between student performance and their estimated thetas, aggregate performance on that item versus its estimated "beta" (estimated item IRT difficulty, SDT optimal threshold).
5) The lack of strong correlations (independence) of discriminabilty, auc_roc with aggregate_performance and item_mean_correct patterns.

## Features

- Support for 3PL and 4PL IRT models
- Signal Detection Theory (SDT) analysis integration
- Parallel parameter estimation for improved performance
- Visualization tools for parameter convergence, item characteristic curves, and other analyses
- Can handle sparse data and/or continuous data (including for SDT parameters)
- Applicable to various types of questions and response data, not limited to educational psychometric problems

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


## Additional information on use in jupyter notebooks

```
!pip install PyIRT_SDT
import PyIRT_SDT as pyirt_sdt
# Returns tabular data from an array of performance data ['participant_id', 'item_id', 'response'] 
table = pyirt_sdt.returnTable(skill_data)
# Returns an object containing IRT and SDT estimated parameters. 
results = pyirt_sdt.solve_IRT_for_matrix(table, FOUR_PL=True, iterations=250, verbose=False)
# Returns a pandas dataframe of the results. Optionally, writes out a csv (set no_csv_export=False to turn this off). 
results_df = pyirt_sdt.export_object_to_csv(results, skill_id, filename='estimatedItemParameters.csv', version='1.0')
results_df
```

```
# Some examples of the estimated IRT parameters and SDT parameters found in the results object (and dataframe). In addition to below, aggregate performance data across students and items are also included.  
# item label found in table.columns
results.question_ids
```

```
# IRT estimated parameters 
# estimated abilities of all students
results.thetas            
```

```
# four arrays, each containing estimated parameters (alpha, beta, etc) for 3PL/4PL IRT model, plus an additional error parameter 
results.est_params
```
        
```
# SDT estimated parameters 
results.auc_roc, results.optimal_threshold, results.tpr, results.tnr```
```

Refer to "Local Tests.ipynb" in "Test" folder for a full example of most of the functionality, with simulated data.

  
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
