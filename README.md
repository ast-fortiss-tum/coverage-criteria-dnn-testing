# Coverage Criteria DNN Testing

Codebase for the BSc thesis by Bengisu Duran: "Evaluation of Coverage Criteria in Deep Neural Network Testing"

## Dependencies

- Python 3.8
- NumPy 1.18.5
- SciPy 1.5.4
- Keras 2.3.1
- TensorFlow 2.2.0
- Matplotlib 3.2.2
- Seaborn 0.10.1
- tqdm 4.46.0

## Run Correlation Analysis

To run the correlation analysis, fill in the following variables in the main method. Below is an example:

```python
# Full path to the Python executable inside the virtual environment. 
# If you have done the profiling once, you can comment out the profiling part.
python_executable = r'C:\Windows\System32\thesis\DNNCov\venv\Scripts\python.exe'

# Model path
model_path = r'lenet5.h5'

# Profile command (change the dataset name if you are using something different)
profile_command = [
    python_executable, profile_py_path,
    '-model', model_path,
    '-output_path', output_path,
    '--mnist'
]

dataset_name = 'mnist'
output_path = 'test_suites'  # Path to the output directory
num_suites = 100  # Number of test suites
tests_per_suite = 100  # Number of tests per suite
suite_limit = 100  # Limit for the number of test suites to save failed tests
noise_and_blur = False
noise_level = 0.4  # Adjust to increase/decrease noise (0.4, 0.8, 1)
blur_sigma = 1.7  # Adjust to increase/decrease blur (1.7, 2, 2.2)

# Parameters for modifying existing test suites with failed tests
failed_test_injection = False
num_suites_to_modify = 100  # Number of existing test suites to modify
num_tests_to_replace = 10  # Number of tests to replace with failed tests in each suite
model = load_model('lenet5.h5')
```
For the test generation test_low_accuracy.py is used with this you can generate the test suites however you like the test_accuracy.py is redundant. It is the version that only produces test with noise and blur. 
# Using your own tests

You can comment out the test generation function if you want to use your own tests:
```python
test_generate(
    dataset_name, output_path, num_suites, tests_per_suite, suite_limit,
    noise_and_blur, num_suites_to_modify, num_tests_to_replace, failed_test_injection
)
```



You have to put your files in a directory called `test_suites` and test suites need to be named `Test Suite i`. However, you can change this in the following lines:
```python
suite_dir = f'Test Suite {i}'
test_dir = 'test_suites/' + suite_dir
```

You also need to place your accuracies in the following path:
```python
accuracies_csv_path = 'accuracies.csv'
```

The format of the `accuracies.csv` file should be like this:
```
Test Suite,Accuracy
Test Suite 1,50.0
Test Suite 2,50.0
Test Suite 3,49.0
Test Suite 4,49.0
Test Suite 5,50.0
Test Suite 6,48.0
```

![image](https://github.com/ast-fortiss-tum/coverage-criteria-dnn-testing/assets/57235879/be08d71b-1387-4d1b-8800-a6ede3c702be)

The first line doesn't matter what it says, but the format `Test Suite 1,50.0` is important. You can change the format in the following function if you have another format:
```python
def load_accuracies(csv_path)
```

# Retrain the model
You can use accuracy_improvement.py to retrain the model with each test suite. The retrain_test.py script is redundant; it only trains the model once, but you can use multiple failed tests from multiple test suites.

An example usage of the variables:

python
```python
dataset_name = 'mnist'
output_path = 'test_suites_imp'  # Path to the output directory
num_suites = 100  # Number of test suites
tests_per_suite = 100  # Number of tests per suite
suite_limit = 1  # Limit for the number of test suites to save failed tests
noise_and_blur = True
noise_level = 0.4  # Adjust to increase/decrease noise 0.4 0.8 1
blur_sigma = 1.7  # Adjust to increase/decrease blur 1.7 2 2.2
```


# Charts
You can use line_chart.py for visualization with line charts and scatter points.
