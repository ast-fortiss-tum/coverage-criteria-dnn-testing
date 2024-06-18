from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import csv
import numpy as np
from keras.datasets import mnist
from scipy.stats import kendalltau
import os
import subprocess
import numpy as np
import pickle
from keras.models import load_model
from keras import backend as K
from scipy.stats import kendalltau
import csv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

import sys
import sys

import matplotlib.pyplot as plt

from mcdc import MCDC, coverage_report_mcdc
from nnet import NNet
from keras.models import load_model
from keras.models import Sequential
#from keras.layers import Conv2D, Dense, Dropout, Input, InpuxtLayer
from keras.layers import Activation, Flatten, BatchNormalization

from Profile import downsampleImage

import argparse, pickle
import shutil

import csv

from keras.models import load_model
import tensorflow as tf
import os

from test_low_accuracy import test_generate

sys.path.append('../')
from keras import Input
from coverage import Coverage
from keras import backend as K
from keras.applications import mobilenet, vgg19, resnet

from keras.applications.vgg16 import preprocess_input
import random
import time
import numpy as np
from test_queue import ImageInputCorpus
from coverage_computer import dry_run
from coverage_computer import save_coverage_values_to_csv
from output_fetcher import build_fetch_function
from Queue import Seed
from tqdm import tqdm

from keras.models import load_model
from coverage import Coverage
#from test_accuracy import TestClass

def imagenet_preprocessing(input_img_data):
    temp = np.copy(input_img_data)
    temp = np.float32(temp)
    qq = preprocess_input(temp)
    return qq




def load_coverage_values(csv_path):
    coverage_values = []
    with open(csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        for row in reader:
            coverage_values.append([float(value) for value in row[1:]])  # Skip the first column (Test Suite)
    return np.array(coverage_values)

def load_accuracies(csv_path):
    accuracies = []
    with open(csv_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip header
        for row in csvreader:
            accuracies.append(float(row[1]))
    return np.array(accuracies)

def calculate_correlations(coverage_values, accuracies):
    correlation_matrix = []
    p_values_matrix = []
    coverage_names = ['kMNC', 'TKNC', 'NBC', 'SNAC', 'NC', 'MCDC_SS', 'MCDC_SV', 'MCDC_VS', 'MCDC_VV']
    for i, coverage_name in enumerate(coverage_names):
        coverage_metric = coverage_values[:, i]
        tau, p_value = kendalltau(coverage_metric, accuracies)
        correlation_matrix.append(tau)
        p_values_matrix.append(p_value)
        print(f'The Kendall Tau correlation between {coverage_name} and accuracy is {tau} with p-value {p_value}')
    return correlation_matrix, p_values_matrix

def plot_heatmap(correlation_matrix, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title(title)
    plt.show()
def mnist_preprocessing(x_test):
    temp = np.copy(x_test)
    temp = temp.reshape(-1, 28, 28, 1)
    temp = temp.astype('float32')
    temp /= 255
    return temp


def cifar_preprocessing(x_test):
    temp = np.copy(x_test)
    temp = temp.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        temp[:, :, :, i] = (temp[:, :, :, i] - mean[i]) / std[i]
    return temp


def tiny_taxinet_preprocessing(x_test):
    return x_test

preprocess_dic = {
    'cifar': cifar_preprocessing,
    'mnist': mnist_preprocessing,
    'tinytaxinet': tiny_taxinet_preprocessing
}

metrics_para = {
    'kmnc': 1000,
    'bknc': 10,
    'tknc': 10,
    'nbc': 10,
    'newnc': 10,
    'nc': 0.75,
    'fann': 1.0,
    'snac': 10
}

if __name__ == '__main__':


# Full path to the Python executable inside the virtual environment
    python_executable = r'C:\Windows\System32\thesis\DNNCov\venv\Scripts\python.exe'

    # Full path to Profile.py
    profile_py_path = r'Profile.py'

    # Ensure the output directory exists
    output_path = r'lenet5output'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Model path
    model_path = r'lenet5.h5'

    # Profile command
    profile_command = [
        python_executable, profile_py_path,
        '-model', model_path,
        '-output_path', output_path,
        '--mnist'
    ]

    # Run the command
    subprocess.run(profile_command, check=True)




suite_limit = 100  # Limit for the number of test suites to save failed tests
noise_and_blur = False
noise_level = 0.4  # Adjust to increase/decrease noise 0.4 0.8 1
blur_sigma = 1.7  # Adjust to increase/decrease blur 1.7 2 2.2

# Parameters for modifying existing test suites with failed tests
failed_test_injection=False
num_suites_to_modify = 100  # Number of existing test suites to modify
num_tests_to_replace = 10  # Number of tests to replace with failed tests in each suite

test_generate(dataset_name, output_path, num_suites, tests_per_suite, suite_limit, noise_and_blur, num_suites_to_modify,
              num_tests_to_replace,failed_test_injection)
model = load_model('lenet5.h5')
dataset_name = 'mnist'  # or 'cifar10' based on your requirement
output_path = 'test_suites_reel10'  # Path to the output directory
num_suites = 100  # Number of test suites
tests_per_suite = 100  # Number of tests per suite

with open(r'C:\Windows\System32\thesis\DNNCov\lenet5output\lenet5output.pickle', 'rb') as f:
  profiling_data = pickle.load(f)
cri = [0, 0, 0, 0, 0]

cri[0] = metrics_para['kmnc']
cri[1] = metrics_para['tknc']
cri[2] = metrics_para['nbc']
cri[3] = metrics_para['snac']
cri[4] = metrics_para['nc']

suite_coverages = []
suite_accuracies = []
img_rows, img_cols = 224, 224
input_shape = (img_rows, img_cols, 3)
input_tensor = Input(shape=input_shape)
exclude_layer_list = ['input', 'flatten']

coverage_dir ='CoverageReport'
if not os.path.exists(coverage_dir):
   os.makedirs(coverage_dir)
directory = os.path.dirname('test_suites/queue')
if not os.path.exists(directory):
    os.makedirs(directory)
summary_file_path = os.path.join(coverage_dir, 'summary_coverage.txt')
summary_file= open(summary_file_path, 'w')
summary_file.write("Coverage Summary Across All Test Suites\n")
summary_file.write("======================================\n\n")

preprocess = preprocess_dic['mnist']
all_suite_coverages = []

# Load original training data
(x_train, y_train), (_, _) = mnist.load_data()  # Use appropriate dataset loading method
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255

for i in tqdm(range(1, num_suites + 1)):  # Assuming 100 test suites as per your scenario
   coverage_handler = Coverage(model=model, criteria="all", k=cri,
                               profiling_dict=profiling_data, exclude_layer=exclude_layer_list)
   # fetch_function is to perform the prediction and obtain the outputs of each layers
   dry_run_fetch = build_fetch_function(coverage_handler, preprocess)
   # The function to update coverage
   coverage_function = coverage_handler.update_coverage
   coverage_file_path = os.path.join(coverage_dir, f'coverage_suite_{i}.csv')
# f'coverage_suite_{i}.csv'
   csvPath = coverage_file_path + 'NC_CSV'
   suite_dir = f'Test Suite {i}'
   test_dir = 'test_suites/' + suite_dir

   



   queue = ImageInputCorpus('test_suites', coverage_handler.total_size, "all")
   with open(csvPath, 'w', newline='') as file:
       writer = csv.writer(file)
       writer.writerow(['Coverage Metric', 'Value'])
       coverage_values = dry_run(test_dir, dry_run_fetch, coverage_function, queue, summary_file, writer)

       all_suite_coverages.append(coverage_values)
   with open(coverage_file_path, 'a', newline='') as file:
       writer = csv.writer(file)
       writer.writerows(coverage_handler.nccsvArr)
   test_images = []
   # Load the generated test suite
   test_list = os.listdir(test_dir)
   for test_name in tqdm(test_list):
       image_path = test_dir +"/"+ test_name
       if os.path.exists(image_path):
           img = np.load(image_path)
           test_images.append(img)
   if test_images:
       input_batches = np.array(test_images)
   else:
       print("Warning: No images loaded. Skipping this suite.")
       continue

   # Preprocess the test suite
   preprocessed_tests = preprocess(input_batches)

   # Calculate MCDC coverage for this suite using the original training data and the preprocessed test suite
   mcdc_coverage = coverage_report_mcdc(model, x_train, preprocessed_tests, trainlimit=100, testlimit=100)
   all_suite_coverages[-1].extend(mcdc_coverage)


   # writes the nccsv array in CSV fashion to the desired file

save_coverage_values_to_csv(all_suite_coverages, 'coverage_values.csv')
print(f"Coverage results saved in '{coverage_dir}' directory and summarized in '{summary_file_path}'.")

summary_file.close()
coverage_csv_path = 'coverage_values.csv'
accuracies_csv_path = 'accuracies.csv'

coverage_values = load_coverage_values(coverage_csv_path)
accuracies = load_accuracies(accuracies_csv_path)



correlation_matrix, p_values_matrix = calculate_correlations(coverage_values, accuracies)

# Convert the correlation matrix into a format suitable for a heatmap
correlation_matrix = np.array(correlation_matrix).reshape((1, -1))

# Plotting the heatmap
coverage_names = ['kMNC', 'TKNC', 'NBC', 'SNAC', 'NC', 'MCDC_SS', 'MCDC_SV', 'MCDC_VS', 'MCDC_VV']
sns.heatmap(correlation_matrix, annot=True, xticklabels=coverage_names, yticklabels=['Kendall Tau'], cmap='coolwarm', fmt=".2f")
plt.title('Kendall Tau Correlation between Coverage Criteria and Model Accuracy')
plt.show()

# Example of extracting KMNC coverage for correlation
#kmnc_coverages = [metrics_para['kmnc'] for cover