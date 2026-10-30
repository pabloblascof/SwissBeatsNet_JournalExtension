import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
# evenly sampled time at 200ms intervals
#t = np.arange(0., 5., 0.2)
t = np.array([0.6, 0.715, 0.72, 0.7329, 0.7379, 0.7405, 0.7424, 0.7443])
k = t - 0.1
std_dev = np.array([0.01, 0.02, 0.01, 0.012, 0.01, 0.008, 0.009, 0.01])  # Example standard deviations
x = np.array([500, 5000, 10000, 20000, 40000, 60000, 80000, 100000])
x_ticks = ("500", '5k', '10k', '20k', '40k', '60k', '80k', '100k')

# Simulate 14 lines of data around the average `t`
np.random.seed(0)  # For reproducibility
all_lines = np.random.normal(loc=t, scale=std_dev, size=(14, len(t)))

# Create a colormap
colors = cm.inferno(np.linspace(0, 1, 14))



# Simulate standard deviations for the 14 lines (for illustrative purposes)
individual_std_devs = np.random.uniform(0.005, 0.002, size=(14, len(t)))

# Create a figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the 14 individual lines with error bars, different colors, and transparency
for line, ind_std, color in zip(all_lines, individual_std_devs, colors):
    ax.errorbar(x, line, yerr=ind_std, fmt='-', color=color, alpha=0.3, linewidth=0.8)

# Plot the data with error bars for standard deviation
ax.plot(x, t,  color='black', marker='o', label='AUROC')


# Generate a baseline that mimics the trend but with lower values
baseline = t - 0.1  # Subtracting a value to lower the baseline
baseline_std_dev = std_dev * 0.8  # Slightly smaller standard deviation for the baseline

# Simulate 14 lines of data around the baseline
baseline_lines = np.random.normal(loc=baseline, scale=baseline_std_dev, size=(14, len(baseline)))

# Simulate standard deviations for the 14 lines (for illustrative purposes)
individual_std_devs = np.random.uniform(0.005, 0.02, size=(14, len(t)))
baseline_individual_std_devs = np.random.uniform(0.005, 0.02, size=(14, len(baseline)))

# Plot the 14 individual baseline lines with error bars, different colors, and transparency
for line, ind_std, color in zip(baseline_lines, baseline_individual_std_devs, colors):
    ax.errorbar(x, line, yerr=ind_std, fmt='-', color=color, alpha=0.4, linewidth=1)

# Plot the baseline line
ax.plot(x, baseline, marker='o', color='red', label='Baseline AUROC', linewidth=2)



# Set the x-ticks and rotate the labels
ax.set_xticks(x)
ax.set_xticklabels(x_ticks, rotation=45)

# labels and titles
ax.set_xlabel('Number of data points')
ax.set_ylabel('AUROC')
#ax.set_title('Your Plot Title')


# Adjust the layout to prevent label cropping
plt.tight_layout()

# Save the figure as a PNG file
plt.savefig('output_plot.png', format='png', dpi=300, bbox_inches='tight')

# Display the plot
plt.show()