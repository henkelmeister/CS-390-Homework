import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

arr = np.array([
       [ 1, 2, 3],
       [4, 5, 6],
       [7, 8, 9],
       [10, 11, 12]])
row = np.array([9, 9, 9, 9])

arr = np.insert(row ,arr, axis=1)
print(arr)
m, n = arr.shape
arr2 = np.ones((m, 1))




