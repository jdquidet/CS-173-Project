import numpy as np

arr = np.arange(10000)
np.random.shuffle(arr)

# Save to text file with space separation
np.savetxt('random_array.txt', arr, fmt='%d')