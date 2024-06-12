import os

import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV files
accuracies_df = pd.read_csv('accuracies.csv')
coverage_df = pd.read_csv('coverage_values.csv', skiprows=1, header=None)

# Define the columns including the 'Test Suite' column
coverage_columns = ['Test Suite', 'kMNC', 'TKNC', 'NBC', 'SNAC', 'NC', 'MCDC_SS', 'MCDC_SV', 'MCDC_VS', 'MCDC_VV']
coverage_df.columns = coverage_columns

# Merge the two dataframes on 'Test Suite'
merged_df = pd.merge(accuracies_df, coverage_df, on='Test Suite')

# Transpose the DataFrame for better visualization
transposed_df = merged_df.set_index('Test Suite').T

# Exclude the Accuracy row for separate plotting
coverage_df_transposed = transposed_df.drop('Accuracy Improvement', axis=0)
accuracy_series = transposed_df.loc['Accuracy Improvement']

# Function to plot each coverage criterion against accuracy with flipped axes
def plot_coverage_vs_accuracy_flipped_line_chart(coverage_criterion, accuracy_series, coverage_series, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(coverage_series, accuracy_series, label='Accuracy Improvement', marker='o')
    plt.title(f'{coverage_criterion} vs Accuracy Improvement')
    plt.xlabel(coverage_criterion)
    plt.ylabel('Accuracy Improvement')
    plt.yticks(range(0, 101, 10))
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_coverage_vs_accuracy_points(coverage_criterion, accuracy_series, coverage_series, save_path):
    plt.figure(figsize=(10, 6))
    plt.scatter(coverage_series, accuracy_series, label='Accuracy Improvement')
    plt.title(f'{coverage_criterion} vs Accuracy Improvement')
    plt.xlabel(coverage_criterion)
    plt.ylabel('Accuracy Improvement')
    plt.yticks(range(0, 30, 5))
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig(save_path)
    plt.close()
# Paths to save the flipped plots

def plot_accuracies(original_csv, retrained_csv, output_path):
    # Read the accuracy data
    original_data = pd.read_csv(original_csv)
    retrained_data = pd.read_csv(retrained_csv)

    # Ensure Test Suite columns are of the same data type
    original_data['Test Suite'] = original_data['Test Suite'].astype(str)
    retrained_data['Test Suite'] = retrained_data['Test Suite'].astype(str)

    # Merge the datasets on 'Test Suite'
    merged_data = pd.merge(original_data, retrained_data, on='Test Suite', suffixes=('_Original', '_Retrained'))

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(merged_data['Test Suite'], merged_data['Accuracy_Original'], label='Original Model', marker='o')
    plt.plot(merged_data['Test Suite'], merged_data['Accuracy_Retrained'], label='Retrained Model', marker='x', linestyle='--')

    # Add titles and labels
    plt.title('Comparison of Model Accuracies')
    plt.xlabel('Test Suite')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45)
    plt.xticks(range(0, 110, 10))
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

flipped_plot_paths = {}

# Generate flipped plots for each coverage criterion
for criterion in coverage_columns[1:]:
    save_path = os.path.join(f'charts', f'Retrain_{criterion}_vs_accuracy_chart.png')  # Replace '/path/to/save/' with your desired save path
    flipped_plot_paths[criterion] = save_path
    original_csv = 'test_suites_retrain_den/accuracies.csv'
    retrained_csv = 'test_suites_retrain_den/retrained_model_accuracies.csv'
    output_path = 'charts/comparison_plot.png'

    # Generate the plot
    #plot_accuracies(original_csv, retrained_csv, output_path)
    plot_coverage_vs_accuracy_points(criterion, accuracy_series.values, coverage_df_transposed.loc[criterion].values, save_path)
    #plot_coverage_vs_accuracy_flipped_line_chart(criterion, accuracy_series.values, coverage_df_transposed.loc[criterion].values, save_path)
