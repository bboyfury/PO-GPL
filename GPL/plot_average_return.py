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
file_types = {
    'Training': 'train',
    'Evaluation': 'eval'
}

# Dynamically load files
def load_files(base_dir, prefix):
    files = {}
    for file_name in os.listdir(base_dir):
        if file_name.startswith(prefix) and file_name.endswith(".csv"):
            label = file_name.replace(f"{prefix}_average_return_", "").replace(".csv", "")
            files[label] = os.path.join(base_dir, file_name)
    return files

# Plot function
def plot_results(data, title_prefix, save_prefix):
    environments = data['Env'].unique()
    for env in environments:
        env_data = data[data['Env'] == env]
        fig, ax = plt.subplots(figsize=(10, 6))

        # Color palette
        palette = sns.color_palette("husl", len(data['Algorithm'].unique()))

        # Plot each algorithm
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

        # Labels, title, and legend
        ax.set_ylabel("Average Return per Episode")
        ax.set_xlabel("Total Steps (x160000)")
        ax.set_title(f"{title_prefix} Return in {env} (8 seed)")
        lgd = ax.legend(framealpha=1, frameon=True, loc='upper left', bbox_to_anchor=(1.01, 1.01))

        # Save and show the figure
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_return_comparison_{env.lower()}.png", dpi=300, bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.show()

# Process and plot for both file types
for category, prefix in file_types.items():
    files = load_files(base_dir, prefix)
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
        plot_results(data, title_prefix=f"Average {category}", save_prefix=prefix)
    else:
        print(f"No data found for {category} files.")
