import numpy as np

def exponential_decay(x, a, b):
    return a * np.exp(-b * x)

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Generate sample histograms with varying distributions
np.random.seed(42)  # For reproducible results
data = [np.random.exponential(scale=1/(0.7-i*0.05), size=1000) for i in range(10)]

# Define bins
bins = np.linspace(1.5, 7, 30)
hist_data = [np.histogram(d, bins=bins) for d in data]

def fit_exponential_decay(hist_data, bins):
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    popt, _ = curve_fit(exponential_decay, bin_centers, hist_data, maxfev=5000)
    return popt

selected_histograms = []
threshold = 0.5  # Define a threshold for selection based on some goodness-of-fit metric

for i, (hist, bins) in enumerate(hist_data):
    try:
        params = fit_exponential_decay(hist, bins)
        residuals = hist - exponential_decay(0.5 * (bins[:-1] + bins[1:]), *params)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((hist - np.mean(hist))**2)
        r_squared = 1 - (ss_res / ss_tot)

        if r_squared > threshold:
            selected_histograms.append((i, r_squared))
    except:
        continue

# Plot selected histograms
for index, r_squared in selected_histograms:
    plt.figure()
    plt.hist(data[index], bins=bins, alpha=0.7, label=f'Histogram {index} with R^2 {r_squared:.2f}')
    plt.plot(bins, exponential_decay(bins, *fit_exponential_decay(*np.histogram(data[index], bins=bins))), 'r-', label='Fit')
    plt.title(f"Selected Histogram {index}")
    plt.legend()
    plt.show()
