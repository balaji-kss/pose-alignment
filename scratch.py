def pick_equidistant_elements(A, m):
    n = len(A)
    if m > n:
        raise ValueError("m cannot be greater than the number of elements in the list")

    # Calculate the step size
    step = n // m

    # Select elements
    selected_elements = [A[i] for i in range(0, n, step)]

    # Adjust the length of the result if it's longer than m
    return selected_elements[:m]

# Example usage
A = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
m = 6
print(pick_equidistant_elements(A, m))

import numpy as np
idx = np.round(np.linspace(0, len(A) - 1, m)).astype(int)
print(np.array(A)[idx])