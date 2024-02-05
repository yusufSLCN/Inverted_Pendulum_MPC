import numpy as np

# Function to generate n random values and save them to a file using numpy
def generate_and_save_random_values(n, filename, mean, std):
    random_values = np.random.normal(mean, std, n)
    np.savetxt(filename, random_values)

# Function to read random values from a file as a numpy array
def read_random_values(filename):
    random_values = np.loadtxt(filename)
    return random_values

# Example usage:
n = 1000
filename = 'random_values.txt'
np.random.seed(0)
# Generate and save random values
generate_and_save_random_values(n, filename, 0, 0.05)

# Read random values from the file as a numpy array
read_values = read_random_values(filename)

# Display the generated and read values
print(f"Generated random values: {generate_and_save_random_values}")
print(f"Read random values as numpy array: {read_values}")
