import os
import sys
from keras.datasets import mnist, cifar10
import numpy as np
from tqdm import tqdm
import csv
from keras.models import load_model, save_model
from keras.utils.np_utils import to_categorical
from scipy.ndimage import gaussian_filter

sys.path.append('../')


def create_batch(x_batch, output_path, prefix):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    batches = np.split(x_batch, len(x_batch), axis=0)
    for i, batch in enumerate(batches):
        saved_name = prefix + str(i) + '.npy'
        np.save(os.path.join(output_path, saved_name), batch)


def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    y_test = to_categorical(y_test, 10)
    return x_test, y_test


def load_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_test = to_categorical(y_test.flatten(), 10)
    return x_test, y_test


def calculate_accuracies(test_suites, label_suites, model):
    accuracies = []
    failed_indices = []
    for suite, labels in zip(test_suites, label_suites):
        predictions = model.predict(suite)
        predicted_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(labels, axis=1)
        accuracy = np.mean(predicted_labels == true_labels)
        accuracies.append(accuracy * 100)
        failed_indices.append(np.where(predicted_labels != true_labels)[0])
    return accuracies, failed_indices


def save_accuracies(accuracies, output_path):
    with open(output_path, 'w', newline='') as file:  # Removed the extra parenthesis
        writer = csv.writer(file)
        writer.writerow(['Test Suite', 'Accuracy'])
        for i, accuracy in enumerate(accuracies):
            writer.writerow([f'Test Suite {i + 1}', accuracy])
    print(f'Accuracies saved to {output_path}')


def add_noise_and_blur(x_batch, noise_level=0.1, blur_sigma=1.0):
    noisy_batch = x_batch + noise_level * np.random.normal(loc=0.0, scale=1.0, size=x_batch.shape)
    noisy_batch = np.clip(noisy_batch, 0., 1.)
    blurred_batch = np.array([gaussian_filter(image, sigma=blur_sigma) for image in noisy_batch])
    blurred_batch = (blurred_batch * 255).astype(np.uint8)  # Convert back to uint8
    return blurred_batch


def save_failed_tests(failed_indices, all_test_suites, all_label_suites, output_path, suite_limit):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    total_suites_saved = 0
    for i, indices in enumerate(failed_indices):
        if total_suites_saved >= suite_limit:
            break
        failed_tests = all_test_suites[i][indices]
        failed_labels = all_label_suites[i][indices]
        np.save(os.path.join(output_path, f'failed_tests_suite_{i + 1}.npy'), failed_tests)
        np.save(os.path.join(output_path, f'labels_failed_tests_suite_{i + 1}.npy'), failed_labels)
        total_suites_saved += 1
    print(f'Failed tests saved from first {suite_limit} test suites.')


def load_failed_tests(output_path, suite_limit):
    failed_tests = []
    failed_labels = []
    for i in range(suite_limit):
        test_file_path = os.path.join(output_path, f'failed_tests_suite_{i + 1}.npy')
        label_file_path = os.path.join(output_path, f'labels_failed_tests_suite_{i + 1}.npy')
        print(f'Trying to load: {test_file_path} and {label_file_path}')
        if os.path.exists(test_file_path) and os.path.exists(label_file_path):
            print(f'Loading files: {test_file_path} and {label_file_path}')
            failed_tests.append(np.load(test_file_path, allow_pickle=True))
            failed_labels.append(np.load(label_file_path, allow_pickle=True))
        else:
            print(f'Files not found: {test_file_path} or {label_file_path}')
    if failed_tests and failed_labels:
        failed_tests = np.concatenate(failed_tests, axis=0)
        failed_labels = np.concatenate(failed_labels, axis=0)
        print(f'Loaded {len(failed_tests)} failed tests')
    else:
        print('No failed tests loaded.')
    return failed_tests, failed_labels


def replace_tests_with_failed_tests(test_suites, label_suites, failed_tests, failed_labels, num_suites_to_modify,
                                    num_tests_to_replace, output_path):
    total_failed_tests = len(failed_tests)
    suite_indices = np.random.choice(len(test_suites), num_suites_to_modify, replace=False)
    modified_test_suites = []
    modified_label_suites = []
    i=0
    for suite_idx in suite_indices:
        indices_to_replace = np.random.choice(len(test_suites[suite_idx]), num_tests_to_replace, replace=False)
        failed_indices = np.random.choice(total_failed_tests, num_tests_to_replace, replace=False)
        print(f'Modifying test suite {suite_idx + 1}')
        print(f'Indices to replace in the suite: {indices_to_replace}')
        print(f'Indices of failed tests used for replacement: {failed_indices}')

        original_tests = test_suites[suite_idx][indices_to_replace].copy()
        test_suites[suite_idx][indices_to_replace] = failed_tests[failed_indices]
        label_suites[suite_idx][indices_to_replace] = failed_labels[failed_indices]

        # Validate that the replacements have been made correctly
        assert not np.array_equal(original_tests, test_suites[suite_idx][
            indices_to_replace]), "Test replacement did not occur properly."

        # Save each modified test suite as an individual file
        modified_test_suite_path = os.path.join(output_path, f'Test Suite {i + 1}')
        if not os.path.exists(modified_test_suite_path):
            os.makedirs(modified_test_suite_path)
        create_batch(test_suites[suite_idx], modified_test_suite_path, f'modified_test_suite_{suite_idx + 1}_')
        np.save(os.path.join(modified_test_suite_path, f'modified_test_suite_{suite_idx + 1}_labels.npy'),
                label_suites[suite_idx])

        modified_test_suites.append(test_suites[suite_idx])
        modified_label_suites.append(label_suites[suite_idx])
        i=i+1

    print(f'{num_suites_to_modify} test suites modified with {num_tests_to_replace} failed test cases each.')
    return modified_test_suites, modified_label_suites, suite_indices


def calculate_and_save_modified_accuracies(modified_test_suites, modified_label_suites, model, output_path,suite_indices):
    accuracies, _ = calculate_accuracies(modified_test_suites, modified_label_suites, model)
    modified_accuracies_path = os.path.join(output_path, 'accuracies.modified.csv')
    save_accuracies(accuracies, modified_accuracies_path)
    print(f'Modified accuracies saved to {modified_accuracies_path}')
    i=0
    # Delete label files after use
    for suite_idx in suite_indices:
        label_file_path = os.path.join(output_path, f'Test Suite {i + 1}',
                                       f'modified_test_suite_{suite_idx + 1}_labels.npy')
        if os.path.exists(label_file_path):
            os.remove(label_file_path)
            print(f'Deleted label file: {label_file_path}')
        i=i+1


def test_generate(dataset_name, output_path, num_suites, tests_per_suite, suite_limit, noise_and_blur,
                  num_suites_to_modify=0, num_tests_to_replace=0, failed_test_injection=False):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if dataset_name == 'mnist':
        x_test, y_test = load_mnist()
    elif dataset_name == 'cifar10':
        x_test, y_test = load_cifar10()
    else:
        raise ValueError("Unknown dataset name")

    total_tests = x_test.shape[0]

    all_test_suites = []
    all_label_suites = []

    for i in range(num_suites):
        indices = np.random.choice(total_tests, tests_per_suite, replace=False)
        x_batch = x_test[indices]
        y_batch = y_test[indices]
        noise_level = 0.3  # Adjust to increase/decrease noise 0.4 0.8 1
        blur_sigma = 1.3
        if noise_and_blur:
            x_batch = add_noise_and_blur(x_batch, noise_level, blur_sigma)

        print(f"Creating batch for Test Suite {i + 1}, with {len(x_batch)} samples")
        create_batch(x_batch, os.path.join(output_path, f'Test Suite {i + 1}'), f'test_{i}_')
        all_test_suites.append(x_batch)
        all_label_suites.append(y_batch)


    model = load_model('lenet5.h5')
    accuracies, failed_indices = calculate_accuracies(all_test_suites, all_label_suites, model)
    save_accuracies(accuracies, os.path.join(output_path, 'accuracies.csv'))

    if failed_test_injection:
        save_failed_tests(failed_indices, all_test_suites, all_label_suites, os.path.join(output_path, 'failed_tests'),
                          suite_limit)

        # Load all failed tests
        failed_tests, failed_labels = load_failed_tests(os.path.join(output_path, 'failed_tests'), suite_limit)
        if len(failed_tests) > 0 and num_suites_to_modify > 0 and num_tests_to_replace > 0:
            # Replace tests in existing test suites with failed tests
            modified_test_suites, modified_label_suites, suite_indices= replace_tests_with_failed_tests(all_test_suites, all_label_suites,
                                                                                          failed_tests, failed_labels,
                                                                                          num_suites_to_modify,
                                                                                          num_tests_to_replace,
                                                                                          os.path.join(output_path,
                                                                                                       'modified_test_suites'))

            # Calculate and save accuracies of the modified test suites
            calculate_and_save_modified_accuracies(modified_test_suites, modified_label_suites, model,
                                                   os.path.join(output_path, 'modified_test_suites'),suite_indices)
    '''
dataset_name = 'mnist'  # or 'cifar10' based on your requirement
output_path = 'test_suites_re'  # Path to the output directory
num_suites = 100  # Number of test suites
tests_per_suite = 100  # Number of tests per suite
suite_limit = 100  # Limit for the number of test suites to save failed tests
noise_and_blur = False
noise_level = 0.4  # Adjust to increase/decrease noise 0.4 0.8 1
blur_sigma = 1.7  # Adjust to increase/decrease blur 1.7 2 2.2

# Parameters for modifying existing test suites with failed tests
failed_test_injection=False
num_suites_to_modify = 100  # Number of existing test suites to modify
num_tests_to_replace = 10  # Number of tests to replace with failed tests in each suite

test_generate(dataset_name, output_path, num_suites, tests_per_suite, suite_limit, noise_and_blur, num_suites_to_modify,
              num_tests_to_replace,failed_test_injection)'''