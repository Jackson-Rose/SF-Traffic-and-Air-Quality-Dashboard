import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
from datetime import datetime


def run_rush_hour_ttest(input_file="consolidated_traffic.csv", output_dir="ttest_results"):
    """
    Perform a t-test comparing traffic congestion during rush hours vs. non-rush hours.

    Args:
        input_file (str): Path to the traffic data CSV file
        output_dir (str): Directory to save results
    """
    print(f"Loading traffic data from {input_file}...")

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    try:
        # Load the data
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} traffic records")

        # Prepare the data
        # Check if we have hour column, otherwise try to extract it from timestamp
        if 'hour' not in df.columns:
            if 'hour_pst' in df.columns:
                df['hour'] = df['hour_pst']
                print("Using hour_pst column for analysis")
            elif 'timestamp' in df.columns:
                # Try to convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df['hour'] = df['timestamp'].dt.hour
                print("Extracted hour from timestamp column")
            else:
                print("Error: No hour information found in the data")
                return

        # Make sure we have congestion data
        if 'congestion_factor' not in df.columns:
            if 'speed' in df.columns and 'free_flow_speed' in df.columns:
                # Calculate congestion factor (free flow speed / current speed)
                # Handle division by zero
                df['speed_safe'] = df['speed'].replace(0, np.nan)
                df['congestion_factor'] = df['free_flow_speed'] / df['speed_safe']

                # Handle infinity and NaN values
                df['congestion_factor'] = df['congestion_factor'].replace([np.inf, -np.inf], np.nan)
                df['congestion_factor'] = df['congestion_factor'].fillna(1.0)

                # Cap extreme values
                max_congestion = 5.0  # Cap at 5x normal travel time
                df['congestion_factor'] = df['congestion_factor'].clip(upper=max_congestion)

                print("Calculated congestion_factor from speed and free_flow_speed")
            else:
                print("Error: No congestion data available")
                return

        # Define rush hours (7-9 AM and 4-6 PM)
        morning_rush = [7, 8, 9]
        evening_rush = [16, 17, 18]
        rush_hours = morning_rush + evening_rush

        # Split data into rush hour and non-rush hour groups
        rush_hour_data = df[df['hour'].isin(rush_hours)]['congestion_factor']
        non_rush_hour_data = df[~df['hour'].isin(rush_hours)]['congestion_factor']

        # Check if we have enough data for a meaningful t-test
        min_sample_size = 30

        if len(rush_hour_data) < min_sample_size or len(non_rush_hour_data) < min_sample_size:
            print(f"Warning: Sample size may be too small for reliable results")
            print(f"Rush hour samples: {len(rush_hour_data)}")
            print(f"Non-rush hour samples: {len(non_rush_hour_data)}")

        # Calculate basic statistics
        rush_mean = rush_hour_data.mean()
        rush_std = rush_hour_data.std()
        non_rush_mean = non_rush_hour_data.mean()
        non_rush_std = non_rush_hour_data.std()

        # Perform the t-test (Welch's t-test for unequal variances)
        t_stat, p_value = stats.ttest_ind(rush_hour_data, non_rush_hour_data, equal_var=False)

        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(rush_hour_data) - 1) * rush_std ** 2 +
                              (len(non_rush_hour_data) - 1) * non_rush_std ** 2) /
                             (len(rush_hour_data) + len(non_rush_hour_data) - 2))

        effect_size = abs(rush_mean - non_rush_mean) / pooled_std

        # Interpret effect size
        if effect_size < 0.2:
            effect_interpretation = "negligible"
        elif effect_size < 0.5:
            effect_interpretation = "small"
        elif effect_size < 0.8:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"

        # Print results
        print("\nT-Test Results:")
        print(f"Rush Hour Mean Congestion: {rush_mean:.4f} (n={len(rush_hour_data)})")
        print(f"Non-Rush Hour Mean Congestion: {non_rush_mean:.4f} (n={len(non_rush_hour_data)})")
        print(f"Difference: {rush_mean - non_rush_mean:.4f}")
        print(f"T-statistic: {t_stat:.4f}")
        print(f"P-value: {p_value:.6f}")

        if p_value < 0.05:
            print("Result: Statistically significant difference found")
        else:
            print("Result: No statistically significant difference found")

        print(f"Effect size (Cohen's d): {effect_size:.4f} ({effect_interpretation})")

        # Save results to file
        results_file = os.path.join(output_dir, "rush_hour_ttest_results.txt")
        with open(results_file, 'w') as f:
            f.write("Rush Hour vs. Non-Rush Hour Congestion T-Test Results\n")
            f.write("==================================================\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("Parameters:\n")
            f.write(f"- Rush Hours: {morning_rush} AM and {evening_rush} PM\n")
            f.write(f"- Data Source: {input_file}\n\n")

            f.write("Statistical Results:\n")
            f.write(f"- Rush Hour Mean Congestion: {rush_mean:.4f} (SD={rush_std:.4f}, n={len(rush_hour_data)})\n")
            f.write(
                f"- Non-Rush Hour Mean Congestion: {non_rush_mean:.4f} (SD={non_rush_std:.4f}, n={len(non_rush_hour_data)})\n")
            f.write(f"- Absolute Difference: {abs(rush_mean - non_rush_mean):.4f}\n")
            f.write(f"- Relative Difference: {((rush_mean / non_rush_mean) - 1) * 100:.2f}%\n\n")

            f.write("T-Test Results (Welch's t-test for unequal variances):\n")
            f.write(f"- T-statistic: {t_stat:.4f}\n")
            f.write(f"- P-value: {p_value:.6f}\n")
            f.write(f"- Significance level: 0.05\n")
            f.write(
                f"- Result: {'Statistically significant difference' if p_value < 0.05 else 'No statistically significant difference'}\n\n")

            f.write("Effect Size:\n")
            f.write(f"- Cohen's d: {effect_size:.4f}\n")
            f.write(f"- Interpretation: {effect_interpretation} effect size\n")

        print(f"\nDetailed results saved to {results_file}")

        # Create visualization
        create_visualization(df, rush_hours, output_dir)

        return rush_mean, non_rush_mean, t_stat, p_value, effect_size

    except Exception as e:
        print(f"Error performing t-test: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_visualization(df, rush_hours, output_dir):
    """
    Create visualizations of rush hour vs non-rush hour congestion

    Args:
        df (pandas.DataFrame): Traffic data with hour and congestion information
        rush_hours (list): Hours defined as rush hours
        output_dir (str): Directory to save visualizations
    """
    try:
        # 1. Hourly congestion bar chart
        plt.figure(figsize=(12, 6))

        # Calculate mean congestion by hour
        hourly_congestion = df.groupby('hour')['congestion_factor'].agg(['mean', 'count']).reset_index()

        # Create bar chart with color coding for rush hours
        bar_colors = ['#ff7f0e' if hour in rush_hours else '#1f77b4' for hour in hourly_congestion['hour']]

        bars = plt.bar(hourly_congestion['hour'], hourly_congestion['mean'], color=bar_colors)

        # Add sample size annotations
        for i, bar in enumerate(bars):
            count = hourly_congestion.iloc[i]['count']
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + 0.02,
                     f'n={count}', ha='center', va='bottom', fontsize=8)

        # Highlight rush hour periods
        plt.axhspan(0, plt.ylim()[1], xmin=7 / 24, xmax=10 / 24, alpha=0.1, color='orange')
        plt.axhspan(0, plt.ylim()[1], xmin=16 / 24, xmax=19 / 24, alpha=0.1, color='orange')

        # Reference line at congestion factor = 1 (no congestion)
        plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)

        # Customize the plot
        plt.title('Mean Traffic Congestion by Hour of Day', fontsize=14)
        plt.xlabel('Hour of Day (24-hour format)')
        plt.ylabel('Congestion Factor')
        plt.xticks(range(0, 24))
        plt.grid(axis='y', alpha=0.3)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#ff7f0e', label='Rush Hours'),
            Patch(facecolor='#1f77b4', label='Non-Rush Hours')
        ]
        plt.legend(handles=legend_elements)

        plt.tight_layout()

        # Save the figure
        hourly_plot_path = os.path.join(output_dir, 'hourly_congestion.png')
        plt.savefig(hourly_plot_path)
        print(f"Saved hourly congestion plot to {hourly_plot_path}")

        # 2. Box plot comparison
        plt.figure(figsize=(10, 6))

        # Prepare data for boxplot
        df['time_period'] = 'Non-Rush Hour'
        df.loc[df['hour'].isin(rush_hours), 'time_period'] = 'Rush Hour'

        # Create boxplot
        boxplot_data = [
            df[df['time_period'] == 'Rush Hour']['congestion_factor'],
            df[df['time_period'] == 'Non-Rush Hour']['congestion_factor']
        ]

        plt.boxplot(boxplot_data, labels=['Rush Hour', 'Non-Rush Hour'], patch_artist=True,
                    boxprops=dict(facecolor='lightblue'), medianprops=dict(color='red'))

        # Reference line
        plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)

        # Add sample size annotations
        for i, label in enumerate(['Rush Hour', 'Non-Rush Hour']):
            count = len(df[df['time_period'] == label])
            plt.annotate(f'n={count}', xy=(i + 1, df['congestion_factor'].max() * 0.95),
                         ha='center', va='center', fontsize=10)

        # Customize plot
        plt.title('Congestion Factor Distribution: Rush Hour vs Non-Rush Hour', fontsize=14)
        plt.ylabel('Congestion Factor')
        plt.grid(axis='y', alpha=0.3)

        # Save the figure
        boxplot_path = os.path.join(output_dir, 'congestion_boxplot.png')
        plt.savefig(boxplot_path)
        print(f"Saved congestion boxplot to {boxplot_path}")

    except Exception as e:
        print(f"Error creating visualizations: {e}")


if __name__ == "__main__":
    # Get file path from user or use default
    default_file = "consolidated_traffic.csv"

    # Check if default file exists, otherwise ask for input
    if not os.path.exists(default_file):
        input_file = input(f"Default file '{default_file}' not found. Enter path to traffic data CSV file: ")
    else:
        input_file = default_file

    # Run the analysis
    if os.path.exists(input_file):
        run_rush_hour_ttest(input_file)
    else:
        print(f"Error: File '{input_file}' not found. Please check the file path.")