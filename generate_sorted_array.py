import numpy as np

# Create a sorted array of integers
arr = np.arange(10000)

# Save to text file with space separation
np.savetxt('sorted_array.txt', arr, fmt='%d')