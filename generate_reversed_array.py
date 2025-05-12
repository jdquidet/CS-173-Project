import numpy as np

# Create a reversed array of integers
arr = np.arange(10000)[::-1]

# Save to text file with space separation
np.savetxt('reversed_array.txt', arr, fmt='%d')