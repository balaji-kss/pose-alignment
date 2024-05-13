import matplotlib.pyplot as plt
import json
import os
import glob
import numpy as np

directory_path = "/home/tumeke-balaji/Documents/results/delta/input_videos/delta_all_data/delta_data/Closing_Overhead_Bin/"
json_paths = glob.glob(os.path.join(directory_path, '*.json'), recursive=False)

def plot_hist(candidate_name):
    for json_path in json_paths:
        name1, name2 = json_path.rsplit('/', 1)[1].rsplit('-')[0].rsplit('_')
        if name2 != candidate_name:continue

        print('json_path ', json_path)

        with open(json_path, 'r') as file:
            data = json.load(file)

        filtered_values = [item[2] for item in data if item[3]]
        
        # Create a histogram of the filtered values
        plt.figure(figsize=(10, 6))  # Set the size of the figure (optional)
        plt.hist(filtered_values, bins=10, color='blue', edgecolor='black')  # You can adjust the number of bins
        plt.title('Histogram of raw distance matrix: ' + str(len(filtered_values)))
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.ylim(0, 50)
        plt.xlim(1.5, 7)
        plt.grid(True) 
        plot_name = json_path.rsplit('/', 1)[1].rsplit('.', 1)[0].rsplit('-', 1)[0] + '.jpg'
        plot_path = os.path.join(json_path.rsplit('/', 1)[0], plot_name)
        plt.savefig(plot_path, format='jpg', dpi=300) # Add grid lines for better readability (optional)
    # plt.show()

from scipy.optimize import curve_fit
def exponential_decay(x, a, b):
    return a * np.exp(-b * x)

def fit_exponential_decay(hist_data, bins):
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    popt, _ = curve_fit(exponential_decay, bin_centers, hist_data, maxfev=5000)
    return popt

def calc_hist_score(hist_data, bins):

    params = fit_exponential_decay(hist_data, bins)
    residuals = hist_data - exponential_decay(0.5 * (bins[:-1] + bins[1:]), *params)

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((hist_data - np.mean(hist_data))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    return r_squared

def calc_metric():
    
    gfiltered_values = []

    for json_path in json_paths:

        print('json_path ', json_path)

        with open(json_path, 'r') as file:
            data = json.load(file)

        filtered_values = np.array([item[2] for item in data if item[3]])
        gfiltered_values.append(filtered_values)
    
    overall_min = min([data.min() for data in gfiltered_values])
    overall_max = max([data.max() for data in gfiltered_values])

    # Define bin edges
    num_bins = 20  # You can adjust the number of bins based on your data's nature
    bin_edges = np.linspace(overall_min, overall_max, num_bins + 1)
    print('bin_edges ', bin_edges)

    gcounts = []
    gcounts_all = []
    for i in range(len(json_paths)):
        json_path = json_paths[i]
        print('json_path ', json_path)
        filtered_values = gfiltered_values[i]
        bin_edges = np.linspace(min(filtered_values), max(filtered_values), num_bins + 1)
        counts, bin_edges = np.histogram(filtered_values, bins=bin_edges)
        print('bin_edges ', bin_edges[:5])
        # print('counts ', counts)
        # rsq = calc_hist_score(counts, bin_edges)
        gcounts.append(np.sum(counts[:5]))

    sorted_pairs = sorted(enumerate(gcounts), key=lambda x: x[1], reverse=True)
    # Extract the sorted indices
    sorted_indices = [index for index, value in sorted_pairs]
    for idx in sorted_indices:
        print(json_paths[idx], 'tot: ', gcounts[idx])
        print('*****************')

# calc_metric()
plot_hist('candidate2')