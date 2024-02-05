import numpy as np

def get_sigmoid_parameters(raw_a, raw_b, a_range=(None, None), b_range=(None, None)):
    """Gets sigmoid parameter variables in NumPy."""

    def maybe_clamp(x, x_range, ignored_if_non_positive):
        """Clamps `x` to `x_range`."""
        x_min, x_max = x_range
        if x_min is not None and x_max is not None and x_min > x_max:
            raise ValueError('Invalid range: %s.' % str(x_range))
        if x_min is not None and (not ignored_if_non_positive or x_min > 0.0):
            x = np.maximum(np.full_like(x, x_min), x)
        if x_max is not None and (not ignored_if_non_positive or x_max > 0.0):
            x = np.minimum(np.full_like(x, x_max), x)
        return x

    # Exponential Linear Unit (ELU) plus 1, similar to PyTorch's F.elu
    a = np.where(raw_a > 0, raw_a, np.exp(raw_a) - 1) + 1.0
    a = maybe_clamp(a, a_range, ignored_if_non_positive=True)
    raw_b = maybe_clamp(raw_b, b_range, ignored_if_non_positive=False)

    return a, raw_b

def get_raw_sigmoid_parameters(sigmoid_a, sigmoid_b):
    return sigmoid_a-1, sigmoid_b

"""
This function is used for CPU which is equaivalient to the training CUDA version
Note that we use this metric for ANN search, so we flatten the [k, 16] into [k*16] vector,
Here we should reshape the [k*16] vector into [k, 16]

Using Numpy broadcasting technique speedup at least 100 times than double for loops.
"""
def probabilistic_distance(A, B, sigmoid_a=1, sigmoid_b=0):
    def scaled_sigmoid(inner_distances, sigmoid_a, sigmoid_b, smoothing=0.1):
        z = sigmoid_a * inner_distances - sigmoid_b
        z = np.clip(z, -20, 20)  # max_value could be something like 10 or 20
        p = 1 / (1 + np.exp(z))    
        # p = 1 / (1 + np.exp(sigmoid_a * inner_distances - sigmoid_b))
        if smoothing > 0.0:
            p = (1.0 - smoothing) * p + smoothing / 2.0
        return p    
    # Reshape A and B to [..., 20, 16]
    A_reshaped = A.reshape(*A.shape[:-1], 20, 1, 16)
    B_reshaped = B.reshape(*B.shape[:-1], 1, 20, 16)    
    # Compute pairwise L2 distances
    distances = np.linalg.norm(A_reshaped - B_reshaped, axis=-1)    

    # Apply the scaled sigmoid function
    distances = scaled_sigmoid(distances, sigmoid_a, sigmoid_b)
    
    return -np.log(np.mean(distances, axis=(-1, -2)))
