import os
import sys
from keras.datasets import mnist, cifar10
import numpy as np
from tqdm import tqdm
import csv
from keras.models import load_model, save_model
from keras.utils.np_utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle
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
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return x_train, y_train, x_test, y_test

def load_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    y_train = to_categorical(y_train.flatten(), 10)
    y_test = to_categorical(y_test.flatten(), 10)
    return x_train, y_train, x_test, y_test

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

def save_accuracies(accuracies, output_path, start_index=0):
    with open(output_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Test Suite', 'Accuracy'])
        for i, accuracy in enumerate(accuracies):
            writer.writerow([f'Test Suite {start_index + i + 1}', accuracy])
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

def recalculate_remaining_accuracies(test_suites, label_suites, retrained_model, start_index, output_path):
    remaining_test_suites = test_suites[start_index:]
    remaining_label_suites = label_suites[start_index:]
    accuracies, _ = calculate_accuracies(remaining_test_suites, remaining_label_suites, retrained_model)
    save_accuracies(accuracies, os.path.join(output_path, 'retrained_model_accuracies.csv'),start_index)
    print(f'Recalculated accuracies saved to {os.path.join(output_path, "retrained_model_accuracies.csv")}')

def test_generate(dataset_name, output_path, num_suites, tests_per_suite, suite_limit, noise_and_blur):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if dataset_name == 'mnist':
        x_train, y_train, x_test, y_test = load_mnist()
    elif dataset_name == 'cifar10':
        x_train, y_train, x_test, y_test = load_cifar10()
    else:
        raise ValueError("Unknown dataset name")

    total_tests = x_test.shape[0]

    all_test_suites = []
    all_label_suites = []

    for i in range(num_suites):
        indices = np.random.choice(total_tests, tests_per_suite, replace=False)
        x_batch = x_test[indices]
        y_batch = y_test[indices]
        if noise_and_blur:
            x_batch = add_noise_and_blur(x_batch, noise_level, blur_sigma)

        print(f"Creating batch for Test Suite {i + 1}, with {len(x_batch)} samples")
        create_batch(x_batch, os.path.join(output_path, f'Test Suite {i + 1}'), f'test_{i}_')
        all_test_suites.append(x_batch)
        all_label_suites.append(y_batch)

    model = load_model('lenet5.h5')
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    accuracies, failed_indices = calculate_accuracies(all_test_suites, all_label_suites, model)
    save_accuracies(accuracies, os.path.join(output_path, 'accuracies.csv'))
    save_failed_tests(failed_indices, all_test_suites, all_label_suites, os.path.join(output_path, 'failed_tests'), suite_limit)

    # Load all failed tests
    failed_tests, failed_labels = load_failed_tests(os.path.join(output_path, 'failed_tests'), suite_limit)
    if len(failed_tests) > 0:
        # Combine original training data with failed tests
        x_train_combined, y_train_combined = shuffle(np.concatenate((x_train, failed_tests), axis=0),
                                                     np.concatenate((y_train, failed_labels), axis=0))

        print(f"Retraining model with {len(x_train_combined)} combined training samples.")
        model.fit(x_train_combined, y_train_combined, epochs=5, batch_size=32)
        retrained_model_path = os.path.join(output_path, 'retrained_lenet5.h5')
        save_model(model, retrained_model_path)
        print(f"Retrained model saved as {retrained_model_path}")

    # Load the retrained model
    retrained_model = load_model(retrained_model_path)

    # Recalculate accuracies for the remaining test suites
    recalculate_remaining_accuracies(all_test_suites, all_label_suites, retrained_model, suite_limit, output_path)


dataset_name = 'mnist'  # or 'cifar10' based on your requirement
output_path = 'test_suites'  # Path to the output directory
num_suites = 100  # Number of test suites
tests_per_suite = 100  # Number of tests per suite
suite_limit = 10  # Limit for the number of test suites to save failed tests
noise_and_blur = False
noise_level = 0.4  # Adjust to increase/decrease noise 0.4 0.8 1
blur_sigma = 1.7  # Adjust to increase/decrease blur 1.7 2 2.2

test_generate(dataset_name, 'test_suites', num_suites, tests_per_suite, suite_limit, noise_and_blur)
