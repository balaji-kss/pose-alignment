import matplotlib.pyplot as plt
import json
import os
import glob
import numpy as np

directory_path = "/home/tumeke-balaji/Documents/results/delta/input_videos/delta_all_data/delta_data/Closing_Overhead_Bin/"
json_paths = glob.glob(os.path.join(directory_path, '*.json'), recursive=False)

def plot_hist():
    for json_path in json_paths:

        print('json_path ', json_path)

        with open(json_path, 'r') as file:
            data = json.load(file)

        filtered_values = [item[2] for item in data if item[3]]
        
        # Create a histogram of the filtered values
        plt.figure(figsize=(10, 6))  # Set the size of the figure (optional)
        plt.hist(filtered_values, bins=20, color='blue', edgecolor='black')  # You can adjust the number of bins
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

        counts, bin_edges = np.histogram(filtered_values, bins=bin_edges)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_centers -= bin_centers[0]
        print('bin_centers ', bin_centers)
        weighted_average = np.sum(counts[:3] * bin_centers[:3]) / np.sum(counts[:3])
        print('counts ', np.sum(counts[:3]))
        print('weighted_average ', weighted_average)
        gcounts.append(np.sum(counts[:3]))
        gcounts_all.append(counts)

    sorted_pairs = sorted(enumerate(gcounts), key=lambda x: x[1], reverse=True)
    # Extract the sorted indices
    sorted_indices = [index for index, value in sorted_pairs]
    for idx in sorted_indices:
        print(json_paths[idx], 'tot: ', gcounts[idx])
        print('counts: ', gcounts_all[idx])
        print('*****************')

# calc_metric()
plot_hist()