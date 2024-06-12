import pickle

# Path to the pickle file
pickle_file_path = r'C:\Windows\System32\thesis\DNNCov\lenet5output\lenet5output.pickle'

csvPath = 'test_suites' + '//NC_CSV'
nccsv = open(csvPath, "w")
nccsv.close()
nccsv = open(csvPath, "a")
# Open the pickle file and load data
with open(pickle_file_path, 'rb') as file:
    your_data = pickle.load(file)

# Print the loaded data
print(your_data)
