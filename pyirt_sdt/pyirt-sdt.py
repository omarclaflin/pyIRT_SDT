"""
PyIRT_SDT: A Python library for Item Response Theory (IRT) and Signal Detection Theory (SDT) estimation.

This module provides functions and classes for performing IRT and SDT analysis on tabular participant performance data.
It includes methods for parameter estimation, model fitting, and various utility functions for data manipulation and visualization.

Author: Omar Claflin (original code, annotated and powerwashed by Claude)
"""

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, auc
from joblib import Parallel, delayed
import datetime
import os

# Constants
MINIMUM_DATA_POINTS = 30

# IRT Models
def three_param_logistic(theta, a, b, c):
    """Three-parameter logistic model for Item Response Theory."""
    return c + (1 - c) * np.exp(a * (theta - b)) / (1 + np.exp(a * (theta - b)))

def four_param_logistic(theta, a, b, c, d):
    """Four-parameter logistic model for Item Response Theory."""
    return c + (d - c) * np.exp(a * (theta - b)) / (1 + np.exp(a * (theta - b)))

# ETL
def returnTable(df, roundValues=True):
    """
    Convert a DataFrame into a pivot table of participant responses.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing participant responses.
        roundValues (bool): Whether to discretize values. Defaults to True.
    
    Returns:
        pd.DataFrame: Pivot table with participants as rows and items as columns.
    """
    if not all(col in df.columns for col in ['participant_id', 'item_id', 'response']):
        raise ValueError("Input DataFrame must contain 'participant_id', 'item_id', and 'response' columns")

    table = pd.pivot_table(df, values='response', index='participant_id', columns='item_id', aggfunc='first')
    
    if np.nanmax(table.values.flatten()) > 1:
        print('Large response values detected, normalizing table')
        table = table / 100
    
    if roundValues:
        print('roundValues=True, discretizing values')
        table = table.apply(np.floor)
    
    return table

# Model Estimation
def parallel_estimate_parameters(table, thetas, PLOT_ON=True, FOUR_PL=True, est_kernel='trf', parallel=True, bounds=None):
    """
    Estimate IRT parameters for all items in parallel.
    
    Args:
        table (pd.DataFrame): Pivot table of participant responses.
        thetas (np.ndarray): Array of participant ability estimates.
        PLOT_ON (bool): Whether to plot results. Defaults to True.
        FOUR_PL (bool): Whether to use 4PL model. Defaults to True.
        est_kernel (str): Estimation kernel. Defaults to 'trf'.
        parallel (bool): Whether to use parallel processing. Defaults to True.
        bounds (tuple): Bounds for parameter estimation. Defaults to None.
    
    Returns:
        tuple: Estimated parameters (discriminability, difficulty, guessing, attention_errors, estimation_errors)
    """
    num_items = len(table.columns)
    model = four_param_logistic if FOUR_PL else three_param_logistic
    
    if bounds is None:
        bounds = ((1, -30, 0.001, .5), (100, 30, .5, .999)) if FOUR_PL else ((1, -30, 0), (100, 30, .5))

    def process_item(item_num):
        item = table.columns[item_num]
        item_series = table[[item]][table[item].notna()]
        item_thetas = thetas[table[item].notna()]
        
        if len(item_series) > MINIMUM_DATA_POINTS:
            try:
                popt, pcov = curve_fit(model, item_thetas, item_series[item], bounds=bounds, method='trf')
                if PLOT_ON:
                    plot_item_fit(item, item_thetas, item_series[item], model, popt)
                
                return (*popt, np.sqrt(np.diag(pcov)))
            except RuntimeError:
                print(f"Fitting failed for item {item}")
        
        return tuple([np.nan] * (5 if FOUR_PL else 4))

    job_function = delayed(process_item)
    results = Parallel(n_jobs=-1 if parallel else 1)(job_function(i) for i in range(num_items))
    
    return tuple(np.array([r[i] for r in results]) for i in range(5 if FOUR_PL else 4))

def solve_IRT_for_matrix(table, all_thetas=None, iterations=50, FOUR_PL=True, show_convergence=10, show_discriminability=0, bounds=None, verbose=False):
    """
    Solve IRT model for a given response matrix.
    
    Args:
        table (pd.DataFrame): Pivot table of participant responses.
        all_thetas (np.ndarray): Initial theta estimates. Defaults to None.
        iterations (int): Number of iterations. Defaults to 50.
        FOUR_PL (bool): Whether to use 4PL model. Defaults to True.
        show_convergence (int): Frequency of convergence plots. Defaults to 10.
        show_discriminability (int): Frequency of discriminability plots. Defaults to 0.
        bounds (tuple): Bounds for parameter estimation. Defaults to None.
        verbose (bool): Whether to print verbose output. Defaults to False.
    
    Returns:
        IRTResults: Object containing estimated parameters and other results.
    """
    num_students, num_items = table.shape
    history_arrays = initialize_history_arrays(iterations, num_students, num_items)

    if all_thetas is None:
        all_thetas = initialize_thetas(table)

    for iter_num in range(iterations):
        if show_convergence > 0 and iter_num % show_convergence == 0:
            print(f"Iteration #{iter_num}")

        all_est_params = parallel_estimate_parameters(table, all_thetas, 
                                                      PLOT_ON=(iter_num == 0 or iter_num == iterations - 1),
                                                      FOUR_PL=FOUR_PL, bounds=bounds)

        update_history_arrays(history_arrays, all_est_params, all_thetas, iter_num)

        all_delta_thetas, all_est_delta_confidence = update_thetas(all_thetas, FOUR_PL, all_est_params, table)

        if check_convergence(all_thetas, all_delta_thetas):
            print(f'Convergence stopped, cycle: {iter_num}')
            break

        all_thetas = np.nansum([all_thetas, all_delta_thetas], 0)

        if sum(all_est_params[0] < 0) > sum(all_est_params[0] >= 0):
            print('Inverting theta/beta spectrum')
            all_thetas *= -1

        plot_convergence(all_delta_thetas, iter_num, show_convergence)
        plot_discriminability(all_est_params[0], iter_num, show_discriminability)

    sdt_results = calculate_sdt_results(table, all_thetas, all_est_params, FOUR_PL, verbose)

    return IRTResults(table, all_thetas, all_est_params, all_delta_thetas, history_arrays, sdt_results)

def calculate_sdt_results(table, all_thetas, all_est_params, FOUR_PL, verbose):
    """Calculate Signal Detection Theory results."""
    num_items = len(table.columns)
    sdt_results = {
        'auc_roc': np.full(num_items, np.nan),
        'optimal_threshold': np.full(num_items, np.nan),
        'tpr': np.full(num_items, np.nan),
        'tnr': np.full(num_items, np.nan)
    }

    for item_num in range(num_items):
        item = table.columns[item_num]
        item_series = table[[item]][table[item].notna()]
        item_thetas = all_thetas[table[item].notna()]
        model_params = [all_est_params[i][item_num] for i in range(4 if FOUR_PL else 3)]

        try:
            predicted = four_param_logistic(item_thetas, *model_params) if FOUR_PL else three_param_logistic(item_thetas, *model_params)
            
            if np.array_equal(item_series.values, item_series.values.astype(bool).astype(int)):
                fpr, tpr, thresholds = roc_curve(item_series.values, predicted)
                sdt_results['auc_roc'][item_num] = roc_auc_score(item_series.values, predicted)
            else:
                if verbose:
                    print(f'Continuous values detected, using custom ROC functions for {item}')
                fpr, tpr, thresholds = custom_roc_curve(item_series.values, predicted)
                sdt_results['auc_roc'][item_num] = auc(fpr, tpr)

            optimal_idx = np.argmax(tpr - fpr)
            sdt_results['optimal_threshold'][item_num] = thresholds[optimal_idx]
            sdt_results['tpr'][item_num] = tpr[optimal_idx]
            sdt_results['tnr'][item_num] = 1 - fpr[optimal_idx]

        except Exception as e:
            if verbose:
                print(f'Error calculating SDT results for item {item}: {str(e)}')

    return sdt_results

# Helper functions for solve_IRT_for_matrix
def initialize_history_arrays(iterations, num_students, num_items):
    """Initialize arrays to store estimation history."""
    return {
        'student_thetas_hx': np.full((iterations, num_students), np.nan),
        'student_delta_thetas_hx': np.full((iterations, num_students), np.nan),
        'discriminability_hx': np.full((iterations, num_items), np.nan),
        'difficulty_hx': np.full((iterations, num_items), np.nan),
        'guessing_hx': np.full((iterations, num_items), np.nan),
        'attention_hx': np.full((iterations, num_items), np.nan),
        'item_confidence_hx': np.full((iterations, num_items), np.nan),
        'item_power_hx': np.full((iterations, num_items), np.nan),
        'theta_confidence_hx': np.full((iterations, num_students), np.nan),
        'theta_power_hx': np.full((iterations, num_students), np.nan),
        'discriminability_error_hx': np.full((iterations, num_items), np.nan),
        'difficulty_error_hx': np.full((iterations, num_items), np.nan),
        'guessing_error_hx': np.full((iterations, num_items), np.nan),
        'attention_error_hx': np.full((iterations, num_items), np.nan),
    }

def initialize_thetas(table):
    """Initialize theta values if not provided."""
    all_thetas = table.mean(axis=1)
    all_thetas[:] = -0.05 + 0.1 * np.random.random(len(all_thetas))
    return all_thetas.values

def update_history_arrays(history_arrays, all_est_params, all_thetas, iter_num):
    """Update history arrays with current iteration results."""
    history_arrays['discriminability_hx'][iter_num, :] = all_est_params[0]
    history_arrays['difficulty_hx'][iter_num, :] = all_est_params[1]
    history_arrays['guessing_hx'][iter_num, :] = all_est_params[2]
    history_arrays['discriminability_error_hx'][iter_num, :] = all_est_params[-1][:, 0]
    history_arrays['difficulty_error_hx'][iter_num, :] = all_est_params[-1][:, 1]
    history_arrays['guessing_error_hx'][iter_num, :] = all_est_params[-1][:, 2]
    
    if len(all_est_params) > 4:  # 4PL model
        history_arrays['attention_hx'][iter_num, :] = all_est_params[3]
        history_arrays['attention_error_hx'][iter_num, :] = all_est_params[-1][:, 3]
    
    history_arrays['student_thetas_hx'][iter_num, :] = all_thetas

def update_thetas(all_thetas, FOUR_PL, all_est_params, table):
    """Update theta estimates."""
    model = four_param_logistic if FOUR_PL else three_param_logistic
    all_params = zip(*all_est_params[:4 if FOUR_PL else 3])
    all_probs = np.array([model(all_thetas, *params) for params in all_params]).T
    all_delta_thetas = np.nansum(all_est_params[0] * (table.values - all_probs), 1) / np.nansum(np.power(all_est_params[0], 2) * all_probs * (1 - all_probs), 1)
    all_est_delta_confidence = 1 / np.nansum(np.power(all_est_params[0], 2) * all_probs * (1 - all_probs), 1)
    return all_delta_thetas, all_est_delta_confidence

def check_convergence(all_thetas, all_delta_thetas, threshold=6):
    """Check if the estimation has converged."""
    updated_thetas = np.nansum([all_thetas, all_delta_thetas], 0)
    theta_range = np.ptp(updated_thetas)
    return theta_range > threshold

def custom_roc_curve(y_true, y_pred):
    """Custom ROC curve calculation for non-binary data."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    sorted_indices = np.argsort(y_pred)[::-1]

# Export

def export_object_to_csv(solvedIRT, skill_id, filename='estimatedItemParameters.csv', version='1.0', no_csv_export=False):
    #inputs solved IRT object with all estimated parameters
    #exports a 10 field csv with 4 estimated parameters and 4 error scores for each question_id

    qid = solvedIRT.question_ids
    alpha,beta,gamma,epsilon = [solvedIRT.est_params[i] for i in range(4)]
    #center beta
    beta = beta + ((min(beta) - max(beta))/2) - min(beta)
    #could scale beta but then need to scale alpha -- skip for now, just interpretability for laypeople
    alpha_c,beta_c,gamma_c,epsilon_c = [np.asarray(solvedIRT.est_params[-1])[:,i] for i in range(4)]

    currentdate = datetime.datetime.today().strftime("%Y-%m-%d")
    
    auc_roc,optimal_threshold,tpr,tnr,sample_size = solvedIRT.auc_roc,solvedIRT.optimal_threshold,solvedIRT.tpr,solvedIRT.tnr,solvedIRT.sample_size

    skill_optimal_threshold = np.mean(solvedIRT.optimal_threshold) 

    student_correct=solvedIRT.student_correct
    
    export_df = pd.DataFrame({'question_id': qid,
                       'skill_id': skill_id,
                       'discriminability': alpha,
                       'difficulty': beta,
                       'guessing': gamma,
                       'inattention': epsilon,
                       'discriminability_error': alpha_c,
                       'difficulty_error': beta_c,
                       'guessing_error': gamma_c,
                       'inattention_error': epsilon_c,
                       'auc_roc': auc_roc,
                       'optimal_threshold': optimal_threshold,
                       'tpr': tpr,
                       'tnr': tnr,
                       'skill_optimal_threshold': skill_optimal_threshold, 
                       'student_mean_accuracy': student_correct, 
                       'sample_size': sample_size,
                       'date_created': currentdate,
                       'version': version})

    
    if no_csv_export:
        return export_df
    else:
        #make directory, store csv in that directory
        if not os.path.isfile(filename):
            export_df.to_csv(filename, index=False)
        else:
            #if it does exist, read in old file and overwrite existing skill_id/question_id lines
            
            # Read the existing CSV file into a DataFrame
            old_df = pd.read_csv(filename)

            # Concatenate the new DataFrame with the old one
            combined_df = pd.concat([export_df, old_df])

            # Drop duplicate rows based on 'question_id' and 'skill_id', keep the first occurrence
            final_df = combined_df.drop_duplicates(subset=['question_id', 'skill_id'], keep='first')

            if len(final_df) != len(combined_df):
                print('len(old), :', str(len(old_df)), 'len(final): ', str(len(final_df)), ' len(combined): ', str(len(combined_df)))
                print('overwriting ',str(len(combined_df)-len(final_df)),' old lines found in existing csv file...')
            # Write the result back to the CSV file
            final_df.to_csv(filename, index=False)
            
            #old method of just updating, assumed skill_id/question_id weren't in there
            #export_df.to_csv(filename, mode='a', header=False, index=False)            
        
def writetolog(comment,filename="logfile"):
    """Exports info to log for large jobs."""
    now = datetime.datetime.now()
    nowtime = now.strftime("%d/%m/%Y %H:%M:%S")
    f = open(filename, "a")
    f.write(nowtime+': '+comment+'\n')
    f.close()

# Result Visualization and Reporting
def plot_convergence(all_delta_thetas, iter_num, show_convergence):
    """Plot convergence information if requested."""
    if show_convergence > 0 and iter_num % show_convergence == 0:
        plt.figure(figsize=(10, 6))
        plt.hist(all_delta_thetas, bins=100)
        plt.title(f'Delta thetas for all participants for run #{iter_num}')
        plt.xlabel('Delta theta')
        plt.ylabel('Frequency')
        plt.show()

def plot_discriminability(discriminability, iter_num, show_discriminability):
    """Plot discriminability information if requested."""
    if show_discriminability > 0 and iter_num % show_discriminability == 0:
        plt.figure(figsize=(10, 6))
        plt.hist(discriminability, bins=100)
        plt.title(f'Discriminability for all items for run #{iter_num}')
        plt.xlabel('Discriminability')
        plt.ylabel('Frequency')
        plt.show()

def plot_item_with_model(model, thetas, item_num, table, popt, FOUR_PL=True):
    """
    Plot the Item Characteristic Curve (ICC) for a specific item.

    Args:
        model (function): The IRT model function.
        thetas (np.array): Array of participant ability estimates.
        item_num (int): Index of the item to plot.
        table (pd.DataFrame): Response data table.
        popt (tuple): Estimated parameters for the item.
        FOUR_PL (bool): Whether to use the 4PL model. Defaults to True.
    """
    item = table.columns[item_num]
    item_series = table[[item]][table[item].notna()]
    item_thetas = thetas[table[item].notna()]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(item_thetas, item_series[item], alpha=0.5)
    
    theta_range = np.linspace(min(item_thetas), max(item_thetas), 100)
    plt.plot(theta_range, model(theta_range, *popt))
    
    plt.title(f'Item Characteristic Curve for Item {item}')
    plt.xlabel('Theta (Ability)')
    plt.ylabel('Probability of Correct Response')
    
    if FOUR_PL:
        plt.text(0.05, 0.95, f'a: {popt[0]:.2f}, b: {popt[1]:.2f}, c: {popt[2]:.2f}, d: {popt[3]:.2f}', 
                 transform=plt.gca().transAxes, verticalalignment='top')
    else:
        plt.text(0.05, 0.95, f'a: {popt[0]:.2f}, b: {popt[1]:.2f}, c: {popt[2]:.2f}', 
                 transform=plt.gca().transAxes, verticalalignment='top')
    
    plt.show()

def plot_information_curves(model, table, thetas, est_params, x_axis=None):
    """
    Plot item information curves and the test information curve.

    Args:
        model (function): The IRT model function.
        table (pd.DataFrame): Response data table.
        thetas (np.array): Array of participant ability estimates.
        est_params (tuple): Estimated parameters for all items.
        x_axis (tuple, optional): Range for x-axis. Defaults to None.
    """
    theta_range = np.linspace(min(thetas), max(thetas), 100)
    total_info = np.zeros_like(theta_range)

    plt.figure(figsize=(12, 6))
    
    for item_num in range(len(table.columns)):
        if ~np.isnan(est_params[0][item_num]):
            item_params = [param[item_num] for param in est_params]
            item_info = calculate_item_information(model, theta_range, *item_params)
            plt.plot(theta_range, item_info, alpha=0.3)
            total_info += item_info

    plt.plot(theta_range, total_info, 'k-', linewidth=2, label='Test Information')
    plt.title('Item and Test Information Curves')
    plt.xlabel('Theta (Ability)')
    plt.ylabel('Information')
    plt.legend()
    
    if x_axis:
        plt.xlim(x_axis)
    
    plt.show()

    # Plot test information curve and theta distribution
    plt.figure(figsize=(12, 6))
    plt.plot(theta_range, total_info / np.sum(total_info), label='Normalized Test Information')
    
    hist, bin_edges = np.histogram(thetas, bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.plot(bin_centers, hist, label='Theta Distribution')
    
    plt.title('Normalized Test Information and Theta Distribution')
    plt.xlabel('Theta (Ability)')
    plt.ylabel('Density')
    plt.legend()
    
    if x_axis:
        plt.xlim(x_axis)
    
    plt.show()

def calculate_item_information(model, theta, *params):
    """Calculate the item information function."""
    p = model(theta, *params)
    if len(params) == 4:  # 4PL model
        a, b, c, d = params
        q = 1 - p
        numerator = (a ** 2) * ((p - c) ** 2) * (q ** 2)
        denominator = (p * q) * ((d - c) ** 2)
        return numerator / denominator
    else:  # 3PL model
        a, b, c = params
        q = 1 - p
        return (a ** 2) * ((p - c) ** 2) / (p * q)

def distributionsOfEstimatedItemParameters(solvedIRT, FOUR_PL=True):
    """
    Plot distributions of estimated item parameters.

    Args:
        solvedIRT (IRTResults): Object containing IRT results.
        FOUR_PL (bool): Whether 4PL model was used. Defaults to True.
    """
    param_names = ['Discriminability', 'Difficulty', 'Guessing']
    if FOUR_PL:
        param_names.append('Inattention')

    fig, axes = plt.subplots(len(param_names), 1, figsize=(10, 5*len(param_names)))
    
    for i, (param, ax) in enumerate(zip(solvedIRT.est_params, axes)):
        ax.hist(param, bins=int(2*np.sqrt(len(param))))
        ax.set_title(f'{param_names[i]} Distribution of All Items')
        ax.set_xlabel(param_names[i])
        ax.set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def plot_sample_parameter_convergence(solvedIRT, sample_of_items=10, sample_of_students=100):
    """
    Plot convergence of parameter estimates for a sample of items and students.

    Args:
        solvedIRT (IRTResults): Object containing IRT results.
        sample_of_items (int): Number of items to sample. Defaults to 10.
        sample_of_students (int): Number of students to sample. Defaults to 100.
    """
    param_names = ['Discriminability', 'Difficulty', 'Guessing', 'Inattention', 'Theta']
    history_arrays = [
        solvedIRT.discriminability_hx,
        solvedIRT.difficulty_hx,
        solvedIRT.guessing_hx,
        solvedIRT.attention_hx,
        solvedIRT.student_thetas_hx
    ]

    fig, axes = plt.subplots(len(param_names), 1, figsize=(10, 5*len(param_names)))

    for param_name, history, ax in zip(param_names, history_arrays, axes):
        if param_name == 'Theta':
            sample = np.random.choice(history.shape[1], sample_of_students, replace=False)
        else:
            sample = np.random.choice(history.shape[1], sample_of_items, replace=False)
        
        ax.plot(history[:, sample])
        ax.set_title(f'History of Estimates of {param_name}')
        ax.set_xlabel('Iteration')
        ax.set_ylabel(param_name)

    plt.tight_layout()
    plt.show()

def timeCourseOfParameterConvergence(solvedIRT, exclusion_percentage=5):
    """
    Plot the time course of parameter convergence.

    Args:
        solvedIRT (IRTResults): Object containing IRT results.
        exclusion_percentage (float): Percentage of initial runs to exclude. Defaults to 5.
    """
    param_names = ['Discriminability', 'Difficulty', 'Guessing', 'Inattention']
    history_arrays = [
        solvedIRT.discriminability_hx,
        solvedIRT.difficulty_hx,
        solvedIRT.guessing_hx,
        solvedIRT.attention_hx
    ]
    error_arrays = [
        solvedIRT.discriminability_error_hx,
        solvedIRT.difficulty_error_hx,
        solvedIRT.guessing_error_hx,
        solvedIRT.attention_error_hx
    ]

    exclude = int(exclusion_percentage / 100 * history_arrays[0].shape[0])

    for param_name, history, error_history in zip(param_names, history_arrays, error_arrays):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        ax1.plot(np.diff(np.nanmean(history, 1), 1))
        ax1.set_title(f'Mean Changes in {param_name} Estimates')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Change')

        ax2.plot(np.nanmean(error_history, 1))
        ax2.set_title(f'Mean Error in {param_name} Estimates')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Error')

        ax3.plot(np.nanmean(error_history[exclude:, :], 1))
        ax3.set_title(f'Mean Error in {param_name} Estimates\n(Excluding First {exclusion_percentage}%)')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Error')

        plt.tight_layout()
        plt.show()

def correlationOfParametersByPerformance(solvedIRT, exclusion_percentage=10):
    """
    Plot correlations between estimated parameters and overall performance.

    Args:
        solvedIRT (IRTResults): Object containing IRT results.
        exclusion_percentage (float): Percentage of outliers to exclude. Defaults to 10.
    """
    # Theta vs Item Correct %
    plt.figure(figsize=(10, 6))
    plt.scatter(solvedIRT.thetas, solvedIRT.item_correct)
    plt.title("Student Correct % by Estimated Student Ability (Theta)")
    plt.xlabel("Theta")
    plt.ylabel("Correct %")
    plt.show()
    print('Correlation:', np.corrcoef(solvedIRT.thetas, solvedIRT.item_correct)[0][1])

    # Theta vs Item Correct % (outliers removed)
    outlier = int(len(solvedIRT.thetas) * (exclusion_percentage / 100))
    indx = np.argsort(solvedIRT.thetas)[outlier:-outlier]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(solvedIRT.thetas[indx], solvedIRT.item_correct.values[indx])
    plt.title(f"Student Correct % by Estimated Student Ability (Theta)\n{exclusion_percentage}% Outliers Removed")
    plt.xlabel("Theta")
    plt.ylabel("Correct %")
    plt.show()
    print('Correlation:', np.corrcoef(solvedIRT.thetas[indx], solvedIRT.item_correct.values[indx])[0][1])

    # Item Difficulty vs Student Correct %
    plt.figure(figsize=(10, 6))
    plt.scatter(solvedIRT.est_params[1], solvedIRT.student_correct)
    plt.title("Item % Correct by Estimated Item Difficulty (Beta)")
    plt.xlabel("Difficulty (Beta)")
    plt.ylabel("Correct %")
    plt.show()
    indx = ~np.isnan(solvedIRT.est_params[1])
    print('Correlation:', np.corrcoef(solvedIRT.est_params[1][indx], solvedIRT.student_correct[indx])[0][1])

    # Item Discriminability vs Student Correct %
    plt.figure(figsize=(10, 6))
    plt.scatter(solvedIRT.est_params[0], solvedIRT.student_correct)
    plt.title("Item % Correct by Estimated Item Discriminability (Alpha)")
    plt.xlabel("Discriminability (Alpha)")
    plt.ylabel("Correct %")
    plt.show()
    indx = ~np.isnan(solvedIRT.est_params[0])
    print('Correlation:', np.corrcoef(solvedIRT.est_params[0][indx], solvedIRT.student_correct[indx])[0][1])

def compareRuns(A, B