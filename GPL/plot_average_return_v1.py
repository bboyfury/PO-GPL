import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Settings
sns.set(font_scale=1.5, style="whitegrid")
num_experiments = 8
num_checkpoints = 40

# File paths (relative to the script location)
base_dir = os.path.dirname(os.path.abspath(__file__))
files = {
    'PF-GPL-1': os.path.join(base_dir, 'train_average_return_po_gpl-1.csv'),
    'PF-GPL-5': os.path.join(base_dir, 'train_average_return_po_gpl-5.csv'),
    'PF-GPL-10': os.path.join(base_dir, 'train_average_return_po_gpl-10.csv'),
    'PF-GPL-20': os.path.join(base_dir, 'train_average_return_po_gpl-20.csv'),
    'GPL': os.path.join(base_dir, 'train_average_return_gpl.csv'),
    'AE': os.path.join(base_dir, 'train_average_return_ae.csv')
}

# Load and process all data into a single DataFrame
data_frames = []
for label, file_path in files.items():
    try:
        df = pd.read_csv(file_path)
        df['Algorithm'] = label  # Add a column for the algorithm name
        data_frames.append(df)
    except FileNotFoundError:
        print(f"File not found: {file_path}. Skipping...")

# Concatenate all data
if data_frames:
    data = pd.concat(data_frames, ignore_index=True)
else:
    raise FileNotFoundError("No files were found. Please check the paths.")

# Plot for each environment
environments = data['Env'].unique()
for env in environments:
    env_data = data[data['Env'] == env]
    fig, ax = plt.subplots(figsize=(10, 6))

    # Color palette
    palette = sns.color_palette("husl", len(files))

    # Plot each algorithm
    legend_count = 0
    pallette = sns.color_palette()
    color_i = -1

    for (algorithm, algorithm_data) in env_data.groupby('Algorithm'):
        color_i += 1
        grouped = algorithm_data.groupby('Step')['Value'].agg(['mean', 'std'])
        steps = grouped.index
        mean_values = grouped['mean']
        std_values = grouped['std']
        ci_values = 1.96 * std_values / np.sqrt(num_experiments)

        # Plot the mean line with shaded std deviation
        ax.plot(steps, mean_values, label=algorithm, color=pallette[color_i])
        ax.fill_between(steps, mean_values - ci_values, mean_values + ci_values, color=pallette[color_i], alpha=0.2)
        legend_count += 1

    # Labels, title, and legend
    ax.set_ylabel("Average Return per Episode")
    ax.set_xlabel("Total Steps (x160000)")
    ax.set_title(f"Average Training Return in {env} (8 seed)")
    lgd = ax.legend(framealpha=1, frameon=True, loc='upper left', bbox_to_anchor=(1.01, 1.01))

    # Save and show the figure
    plt.tight_layout()
    plt.savefig(f"average_return_comparison_{env.lower()}.png", dpi=300, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()
