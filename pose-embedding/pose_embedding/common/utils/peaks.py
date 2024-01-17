import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def valleys_dynamic_weighting_thresholding_nan(data, window_size, base_global_weight=0.01, adjusted_global_weight=0.05):
    """Valley detection using dynamically adjusted prominence weights
        Sliding window moves one step for each stride.
    """
    global_std = np.nanstd(data)
    window_size = min(len(data), window_size)
    all_local_stds = [np.nanstd(data[i:i+window_size]) for i in range(0, len(data) - window_size + 1)]
    threshold = np.percentile(all_local_stds, 25)
    all_valleys = []

    for i in range(0, len(data) - window_size + 1):
        window = data[i:i+window_size]
        local_std = np.nanstd(window)
        local_prominence = local_std

        # Adjust the global weight based on the local standard deviation
        if local_std < threshold:
            global_weight = adjusted_global_weight
        else:
            global_weight = base_global_weight
        # print(f"local_std {local_std:.2f} -- threshold {threshold:.2f}, global_weight = {global_weight:.2f}")

        combined_prominence = (1 - global_weight) * local_prominence + global_weight * global_std
        valleys, _ = find_peaks(-window, prominence=combined_prominence)
        adjusted_valleys = [v + i for v in valleys]
        all_valleys.extend(adjusted_valleys)

    # Check for the leftmost valley for each dataset
    if data[0] < np.nanmean(data) - np.nanstd(data):
        all_valleys = [0] + all_valleys

    # Deduplicate and sort
    all_valleys = sorted(list(set(all_valleys)))

    return all_valleys


def height_based_detection_without_prominence_nan(data, window_size=30, height_factor=0.7):
    """
    Detect valleys using a hybrid statistical approach for height without considering prominence.
    """
    global_mean = np.nanmean(data)
    global_std = np.nanstd(data)
    global_height = global_mean - height_factor * global_std
    window_size = min(len(data), window_size)

    all_valleys = []

    for i in range(0, len(data) - window_size + 1):
        window = data[i:i+window_size]
        local_mean = np.nanmean(window)
        local_std = np.nanstd(window)
        local_height = local_mean - height_factor * local_std

        # Compute adaptive height value
        beta = min(1, max(0, local_std / global_std))
        height_value = beta * local_height + (1 - beta) * global_height

        valleys, _ = find_peaks(-window, height=-height_value)
        adjusted_valleys = [v + i for v in valleys]
        all_valleys.extend(adjusted_valleys)

    # Check for the leftmost valley for each dataset
    if data[0] < np.nanmean(data) - np.nanstd(data):
        all_valleys = [0] + all_valleys

    all_valleys = sorted(list(set(all_valleys)))

    return all_valleys


def valleys_dynamic_weighting_thresholding(data, window_size, base_global_weight=0.01, adjusted_global_weight=0.05):
    """Valley detection using dynamically adjusted prominence weights
        Sliding window moves one step for each stride.
    """
    global_std = np.std(data)
    window_size = min(len(data), window_size)
    all_local_stds = [np.std(data[i:i+window_size]) for i in range(0, len(data) - window_size + 1)]
    threshold = np.percentile(all_local_stds, 25)
    all_valleys = []

    for i in range(0, len(data) - window_size + 1):
        window = data[i:i+window_size]
        local_std = np.std(window)
        local_prominence = local_std

        # Adjust the global weight based on the local standard deviation
        if local_std < threshold:
            global_weight = adjusted_global_weight
        else:
            global_weight = base_global_weight
        # print(f"local_std {local_std:.2f} -- threshold {threshold:.2f}, global_weight = {global_weight:.2f}")

        combined_prominence = (1 - global_weight) * local_prominence + global_weight * global_std
        valleys, _ = find_peaks(-window, prominence=combined_prominence)
        adjusted_valleys = [v + i for v in valleys]
        all_valleys.extend(adjusted_valleys)

    # Check for the leftmost valley for each dataset
    if data[0] < np.mean(data) - np.std(data):
        all_valleys = [0] + all_valleys

    # Deduplicate and sort
    all_valleys = sorted(list(set(all_valleys)))

    return all_valleys


def height_based_detection_without_prominence(data, window_size=30, height_factor=0.7):
    """
    Detect valleys using a hybrid statistical approach for height without considering prominence.
    """
    global_mean = np.mean(data)
    global_std = np.std(data)
    global_height = global_mean - height_factor * global_std
    window_size = min(len(data), window_size)

    all_valleys = []

    for i in range(0, len(data) - window_size + 1):
        window = data[i:i+window_size]
        local_mean = np.mean(window)
        local_std = np.std(window)
        local_height = local_mean - height_factor * local_std

        # Compute adaptive height value
        beta = min(1, max(0, local_std / global_std))
        height_value = beta * local_height + (1 - beta) * global_height

        valleys, _ = find_peaks(-window, height=-height_value)
        adjusted_valleys = [v + i for v in valleys]
        all_valleys.extend(adjusted_valleys)

    # Check for the leftmost valley for each dataset
    if data[0] < np.mean(data) - np.std(data):
        all_valleys = [0] + all_valleys

    all_valleys = sorted(list(set(all_valleys)))

    return all_valleys

# detect local valleys
def detect_local_valleys(data, window_size, overlap, alpha=1):
    step = window_size - overlap
    all_valleys = []
    
    for i in range(0, len(data) - window_size + 1, step):
        window = data[i:i+window_size]
        local_height = np.mean(window) - 1 * np.std(window)
        
        # Calculate local prominence based on the window's standard deviation
        local_prominence = alpha * np.std(window)
        
        valleys, _ = find_peaks(-window, threshold=-local_height, prominence=local_prominence)
        adjusted_valleys = [v + i for v in valleys]
        all_valleys.extend(adjusted_valleys)
    
    all_valleys = sorted(list(set(all_valleys)))
    # Check for the leftmost valley
    if data[0] < np.mean(data) - 1 * np.std(data):
        all_valleys = [0] + all_valleys    

    return np.array(all_valleys)

def plot_and_save_valleys(data, x_values_seconds, valleys, stride, fps, title_prefix="", filename="valleys_plot.png"):
    """
    Plot the given data and highlight the detected valleys.
    
    Parameters:
    - data: A numpy array representing the data.
    - valleys: A numpy array containing indices of the detected valleys.
    - filename: The name of the PNG file to save the plot.
    """
    
    # Plot the data
    plt.figure(figsize=(10, 6))
    # x_values_seconds = [i * stride / fps for i in range(len(data))]

    plt.plot(x_values_seconds, data, label="Data", color="blue")
    
    # Highlight the detected valleys
    if valleys.shape[0] > 0:
        plt.scatter(valleys * stride / fps, data[valleys], color="red", label="Detected Repetitions")
    
    # Setting labels, title and legend
    plt.xlabel("Seconds")
    plt.ylabel("Similarity Scores")
    plt.title(title_prefix+f"Detected {len(valleys)} Repetitions")
    plt.legend()
    plt.minorticks_on()
    
    # Save the plot to a PNG file
    plt.savefig(filename)
    plt.close()
