# templates/ppo_rnd_atari/plot.py
import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tensorboard.backend.event_processing import event_accumulator

PLOT_X_AXIS_LIMIT = 100_000_000

# Define labels for the runs to be plotted.
# Keys are the basenames of the run directories (e.g., "run_0", "run_1").
# Values are the display names for legends and titles.
labels = {
    "run_0": "Baseline (RND)",
    "run_1": "DG-RND (EMA α=0.01, Sigmoid κ=1.0)",
    "run_2": "DG-RND (EMA α=0.001, Sigmoid κ=1.0)"
}

# Define metrics to plot from TensorBoard
TB_METRICS_TO_PLOT = {
    "Mean Episodic Return (100 Episodes)": "charts/mean_episodic_return_100",
    "Mean Episodic Num Rooms (100 Episodes)": "charts/mean_episodic_num_rooms_100",
    "Episodic Return (Raw)": "charts/episodic_return",
    "Episodic Length (Raw)": "charts/episodic_length",
    "Episodic Num Rooms (Raw)": "charts/episodic_num_rooms",
    "RND Loss": "losses/rnd_loss",
    "Mean Normalized Intrinsic Reward": "rnd/mean_norm_intrinsic_reward",
    "RND Novelty EMA": "rnd/novelty_ema"
}

# Define metrics to plot from final_info.json (for bar charts)
FINAL_METRICS_TO_PLOT = [
    "Best Mean Episodic Return (Extrinsic)",
    "Max Mean Num Rooms (100)",
    "Mean Episodic Length (Last 100)",
    "Mean Episodic Num Rooms (Last 100)",
    "Max number of Rooms Reached",
    "Max reward in an episode",
    "Wall Clock Time (seconds)",
]

def load_tb_data(log_dir, tag):
    """Loads scalar data for a specific tag from TensorBoard logs."""
    ea = event_accumulator.EventAccumulator(
        log_dir,
        size_guidance={event_accumulator.SCALARS: 0} # Load all scalars
    )
    ea.Reload()
    if tag not in ea.Tags()['scalars']:
        print(f"Warning: Tag '{tag}' not found in {log_dir}")
        return None, None
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values

def plot_learning_curves(run_dirs, output_dir):
    """Plots learning curves from TensorBoard logs for multiple runs."""
    plt.style.use('seaborn-v0_8-darkgrid') # Or any preferred style

    for plot_title, tb_tag in TB_METRICS_TO_PLOT.items():
        plt.figure(figsize=(12, 7)) # Slightly wider for potentially longer labels
        print(f"Plotting: {plot_title} ({tb_tag})")
        base_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        color_idx = 0
        max_steps = 0

        for run_dir_path in run_dirs:
            run_basename = os.path.basename(run_dir_path)
            if run_basename == ".": # Handle case where current directory is passed
                # Try to find if '.' corresponds to a key in labels, e.g. if labels had a "." key
                # For now, assume basenames like "run_0", "run_1" are used.
                # If "." is truly "run_0", it should be passed as "run_0" or labels updated.
                # This logic primarily targets explicit directory names like "run_0", "run_1".
                pass # Let it be skipped if not in labels or handled if "." is a key

            if run_basename not in labels:
                print(f"Skipping {run_dir_path} for plot '{plot_title}': Basename '{run_basename}' not in labels dictionary.")
                continue

            run_label = labels[run_basename]
            tb_log_dir = os.path.join(run_dir_path, "tb_logs")

            if not os.path.exists(tb_log_dir):
                print(f"Skipping {run_label} ({run_dir_path}): TensorBoard log directory not found at {tb_log_dir}")
                continue

            steps, values = load_tb_data(tb_log_dir, tb_tag)

            if not steps or not values:
                print(f"No data found for tag '{tb_tag}' in {run_label} ({tb_log_dir})")
                continue

            if plot_title == "RND Loss" and len(steps) > 10: # Ensure there are enough points to skip
                steps = steps[10:]
                values = values[10:]
            
            # Choose this run’s base color
            c = base_colors[color_idx % len(base_colors)]
            color_idx += 1

            # 1) Raw trace in a light tint of the same color
            plt.plot(
                steps,
                values,
                color=c,
                alpha=0.2,
                linewidth=1,
                zorder=1
            )

            # 2) Compute a rolling‐mean and plot it boldly
            window = 500  # Adjust smoothing window as needed
            # For metrics that are already means (e.g., "Mean ... (100 Episodes)"),
            # a large window might over-smooth. Consider adjusting window based on plot_title.
            # For now, keep it consistent.
            values_smooth = (
                pd.Series(values)
                .rolling(window, min_periods=1) # min_periods=1 ensures start of plot is shown
                .mean()
                .values
            )
            plt.plot(
                steps,
                values_smooth,
                label=run_label, # Use the descriptive label
                color=c,
                alpha=1.0,
                linewidth=2.5,
                zorder=2
            )
            if steps:
                 max_steps = max(max_steps, steps[-1])


        plt.title(f"{plot_title} vs. Global Steps")
        plt.xlabel("Global Steps")
        plt.ylabel(plot_title)
        plt.legend(loc='best')
        actual_max_steps = max_steps if max_steps > 0 else PLOT_X_AXIS_LIMIT # Use limit if no data found
        plt.xlim(0, min(PLOT_X_AXIS_LIMIT, actual_max_steps))
        #plt.xlim(0, max_steps if max_steps > 0 else None) # Set x-limit based on max steps found
        # Optional: Log scale for y-axis if appropriate (e.g., for loss)
        # if "Loss" in plot_title: plt.yscale('log')
        plt.tight_layout()
        plot_filename = f"{plot_title.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
        plt.savefig(os.path.join(output_dir, plot_filename))
        print(f"Saved plot: {os.path.join(output_dir, plot_filename)}")
        plt.close()


def plot_final_metrics(run_dirs, output_dir):
    """Plots bar charts of final metrics from final_info.json."""
    plt.style.use('seaborn-v0_8-darkgrid')
    processed_final_data = []

    for run_dir_path in run_dirs:
        run_basename = os.path.basename(run_dir_path)
        if run_basename == ".": # Similar handling as in learning curves
            pass

        if run_basename not in labels:
            print(f"Skipping {run_dir_path} for final metrics: Basename '{run_basename}' not in labels dictionary.")
            continue
        
        run_label = labels[run_basename]
        info_path = os.path.join(run_dir_path, "final_info.json")

        if not os.path.exists(info_path):
            print(f"Skipping {run_label} ({run_dir_path}): final_info.json not found at {info_path}")
            continue
        try:
            with open(info_path, 'r') as f:
                data = json.load(f)
                # Store with the descriptive label for DataFrame construction
                entry = {'run_name': run_label} 
                for metric_key, metric_value_dict in data.items():
                    if isinstance(metric_value_dict, dict) and 'means' in metric_value_dict:
                        entry[metric_key] = metric_value_dict['means']
                    else: # Handle older format or direct values if necessary
                        entry[metric_key] = metric_value_dict 
                processed_final_data.append(entry)
        except Exception as e:
            print(f"Error loading or processing {info_path} for {run_label}: {e}")

    if not processed_final_data:
        print("No final_info.json data processed to plot.")
        return

    # Create a pandas DataFrame from the processed data
    df = pd.DataFrame(processed_final_data)

    if df.empty:
        print("DataFrame for final metrics is empty. No plots will be generated.")
        return

    for metric in FINAL_METRICS_TO_PLOT:
        if metric in df.columns: # Check if metric is a column in the DataFrame
            plt.figure(figsize=(max(8, len(df['run_name']) * 1.5), 7)) # Adjust width based on num runs
            
            # Use seaborn for nicer bar plots
            # The order of bars will depend on the order in `processed_final_data` / `df`
            # which is determined by the order of `run_dirs` filtered by `labels`.
            sns.barplot(x='run_name', y=metric, data=df, palette="viridis", hue='run_name', dodge=False)
            plt.legend([],[], frameon=False) # Remove legend if hue is used for coloring only

            plt.title(f"Final {metric}", fontsize=15)
            plt.xlabel("Experiment Run", fontsize=12)
            plt.ylabel(metric)
            plt.xticks(rotation=45, ha='right') # Rotate labels if many runs
            plt.tight_layout()
            plot_filename = f"final_{metric.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
            plt.savefig(os.path.join(output_dir, plot_filename))
            print(f"Saved plot: {os.path.join(output_dir, plot_filename)}")
            plt.close()
        else:
             print(f"Metric '{metric}' not found in any final_info.json files, skipping plot.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot results from AI Scientist runs.")
    parser.add_argument(
        "--runs",
        nargs='*',  # 0 or more arguments
        default=None, # Default to None, logic below will use keys from 'labels' dict if None
        help="List of run directory basenames/paths to plot (e.g., run_0 results/exp/run_1). "
             "If not provided, all runs defined in the 'labels' dictionary will be attempted."
    )
    parser.add_argument("--output_dir", default=".", help="Directory to save plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    runs_to_attempt_plotting = []
    if args.runs is None:  # --runs flag was not used
        runs_to_attempt_plotting = list(labels.keys())
        print(f"No specific runs provided via --runs. Attempting to plot for all runs in 'labels' dictionary: {runs_to_attempt_plotting}")
    elif not args.runs:  # --runs flag was used, but no values were provided
        runs_to_attempt_plotting = list(labels.keys())
        print(f"--runs flag used but no run names specified. Attempting to plot for all runs in 'labels' dictionary: {runs_to_attempt_plotting}")
    else: # --runs flag was used with specific run names/paths
        runs_to_attempt_plotting = args.runs
        print(f"Attempting to plot for specified runs: {runs_to_attempt_plotting}")

    if not runs_to_attempt_plotting:
        print("No runs identified for plotting. Exiting.")
    else:
        print(f"Saving plots to: {args.output_dir}")
        plot_learning_curves(runs_to_attempt_plotting, args.output_dir)
        plot_final_metrics(runs_to_attempt_plotting, args.output_dir)
        print("Plotting complete.")
