import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import os
import traceback
import json
import pytz

# Create output directory with today's date
output_dir = f"statistical_tests_{datetime.now().strftime('%Y-%m-%d')}"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")
else:
    print(f"Using existing output directory: {output_dir}")


# Helper function to get path for output files
def get_output_path(filename):
    return os.path.join(output_dir, filename)


# Load consolidated data
try:
    # Try to load CSV version first (faster)
    traffic_df = pd.read_csv('consolidated_traffic.csv')
    air_df = pd.read_csv('consolidated_air_quality.csv')
    print("Loaded CSV data files")
except FileNotFoundError:
    # Fall back to JSON if CSV not found
    try:
        with open('consolidated_traffic.json', 'r') as f:
            traffic_data = json.load(f)
            traffic_df = pd.DataFrame(traffic_data['data'])

        with open('consolidated_air_quality.json', 'r') as f:
            air_data = json.load(f)
            air_df = pd.DataFrame(air_data['data'])
        print("Loaded JSON data files")
    except FileNotFoundError:
        raise FileNotFoundError("Could not find consolidated data files. Run data-consolidation.py first.")

# Check if we already have the PST hour column from the data consolidation
for df in [traffic_df, air_df]:
    # Use the PST-specific columns if they exist
    if 'hour_pst' in df.columns:
        print("Using pre-calculated PST hour information")
        df['hour'] = df['hour_pst']
        df['day_of_week'] = df['day_of_week_pst']
        df['is_weekend'] = df['is_weekend_pst']
    else:
        # If not, we need to handle the timestamps carefully
        print("Converting timestamps to PST")
        # Parse timestamps with UTC=True to handle mixed timezones safely
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

        # Convert to Pacific time
        pacific = pytz.timezone('America/Los_Angeles')
        df['timestamp_pst'] = df['timestamp'].dt.tz_convert(pacific)

        # Extract hour, day of week, and weekend flags
        df['hour'] = df['timestamp_pst'].dt.hour
        df['day_of_week'] = df['timestamp_pst'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# Calculate statistics for PM2.5 data
pm25_mean = air_df['pm2_5'].mean()
pm25_std = air_df['pm2_5'].std()

# Filter out values more than 4 standard deviations from the mean
air_df = air_df[air_df['pm2_5'] <= (pm25_mean + 4 * pm25_std)]

# Alternative approach: use absolute threshold based on EPA standards
# PM2.5 readings above 500 μg/m³ are extremely rare even in severe pollution events
air_df = air_df[air_df['pm2_5'] <= 500]

print(f"Removed extreme PM2.5 outliers")


def rush_hour_analysis():
    """
    Enhanced analysis of air quality differences between rush hour and non-rush hour periods
    with detailed visualization and comprehensive T-test results.
    """
    print("\nRush Hour vs. Non-Rush Hour Analysis:")

    # Define rush hours (morning and evening peak traffic times)
    morning_rush = [7, 8, 9]  # 7-9 AM
    evening_rush = [16, 17, 18]  # 4-6 PM
    rush_hours = morning_rush + evening_rush

    # Create a results file
    results_file = get_output_path('rush_hour_analysis_results.txt')
    with open(results_file, 'w') as f:
        f.write("Rush Hour Air Quality Analysis\n")
        f.write("============================\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Rush Hours Defined As:\n")
        f.write(f"- Morning Rush: {morning_rush[0]}:00-{morning_rush[-1]}:59 (7-10 AM)\n")
        f.write(f"- Evening Rush: {evening_rush[0]}:00-{evening_rush[-1]}:59 (4-7 PM)\n\n")

    # Group data by hour and calculate mean PM2.5 for each hour
    hourly_pm25 = air_df.groupby('hour')['pm2_5'].agg(['mean', 'count', 'std']).reset_index()
    hourly_pm25['is_rush_hour'] = hourly_pm25['hour'].isin(rush_hours)

    # For clearer visualization, also mark morning and evening rush separately
    hourly_pm25['rush_category'] = 'Non-Rush'
    hourly_pm25.loc[hourly_pm25['hour'].isin(morning_rush), 'rush_category'] = 'Morning Rush'
    hourly_pm25.loc[hourly_pm25['hour'].isin(evening_rush), 'rush_category'] = 'Evening Rush'

    # Calculate overall statistics
    rush_pm25 = air_df[air_df['hour'].isin(rush_hours)]['pm2_5']
    non_rush_pm25 = air_df[~air_df['hour'].isin(rush_hours)]['pm2_5']

    # T-test for rush hour vs non-rush hour
    t_stat, p_val = stats.ttest_ind(rush_pm25, non_rush_pm25, equal_var=False)

    # Calculate effect size (Cohen's d)
    effect_size = (rush_pm25.mean() - non_rush_pm25.mean()) / np.sqrt((rush_pm25.var() + non_rush_pm25.var()) / 2)

    # Separate tests for morning rush vs non-rush and evening rush vs non-rush
    morning_rush_pm25 = air_df[air_df['hour'].isin(morning_rush)]['pm2_5']
    evening_rush_pm25 = air_df[air_df['hour'].isin(evening_rush)]['pm2_5']

    morning_t, morning_p = stats.ttest_ind(morning_rush_pm25, non_rush_pm25, equal_var=False)
    evening_t, evening_p = stats.ttest_ind(evening_rush_pm25, non_rush_pm25, equal_var=False)
    morning_vs_evening_t, morning_vs_evening_p = stats.ttest_ind(morning_rush_pm25, evening_rush_pm25, equal_var=False)

    # Calculate morning and evening effect sizes
    morning_effect = (morning_rush_pm25.mean() - non_rush_pm25.mean()) / np.sqrt(
        (morning_rush_pm25.var() + non_rush_pm25.var()) / 2)
    evening_effect = (evening_rush_pm25.mean() - non_rush_pm25.mean()) / np.sqrt(
        (evening_rush_pm25.var() + non_rush_pm25.var()) / 2)

    # Print results
    print(f"Rush Hours PM2.5 (mean): {rush_pm25.mean():.2f} µg/m³ (n={len(rush_pm25)})")
    print(f"Non-Rush Hours PM2.5 (mean): {non_rush_pm25.mean():.2f} µg/m³ (n={len(non_rush_pm25)})")
    print(f"Difference: {rush_pm25.mean() - non_rush_pm25.mean():.2f} µg/m³")
    print(f"T-test: t={t_stat:.4f}, p={p_val:.4f}")
    print(f"Effect size (Cohen's d): {effect_size:.4f}")

    significance = "statistically significant" if p_val < 0.05 else "not statistically significant"
    print(f"The difference is {significance} at the 0.05 level.")

    # More detailed analysis of morning vs evening rush
    print("\nMorning Rush Hours:")
    print(f"  Mean PM2.5: {morning_rush_pm25.mean():.2f} µg/m³ (n={len(morning_rush_pm25)})")
    print(f"  T-test vs Non-Rush: t={morning_t:.4f}, p={morning_p:.4f}")

    print("\nEvening Rush Hours:")
    print(f"  Mean PM2.5: {evening_rush_pm25.mean():.2f} µg/m³ (n={len(evening_rush_pm25)})")
    print(f"  T-test vs Non-Rush: t={evening_t:.4f}, p={evening_p:.4f}")

    print("\nMorning vs Evening Rush:")
    print(f"  T-test: t={morning_vs_evening_t:.4f}, p={morning_vs_evening_p:.4f}")

    # Write results to file
    with open(results_file, 'a') as f:
        f.write("Statistical Results\n")
        f.write("-----------------\n\n")
        f.write(f"All Rush Hours PM2.5 (mean): {rush_pm25.mean():.2f} µg/m³ (n={len(rush_pm25)})\n")
        f.write(f"Non-Rush Hours PM2.5 (mean): {non_rush_pm25.mean():.2f} µg/m³ (n={len(non_rush_pm25)})\n")
        f.write(f"Difference: {rush_pm25.mean() - non_rush_pm25.mean():.2f} µg/m³\n")
        f.write(f"T-test: t={t_stat:.4f}, p={p_val:.4f}\n")
        f.write(f"Effect size (Cohen's d): {effect_size:.4f}\n")
        f.write(f"Interpretation: The difference is {significance} at the 0.05 level.\n\n")

        f.write("Morning Rush Hours:\n")
        f.write(f"  Mean PM2.5: {morning_rush_pm25.mean():.2f} µg/m³ (n={len(morning_rush_pm25)})\n")
        f.write(f"  T-test vs Non-Rush: t={morning_t:.4f}, p={morning_p:.4f}\n")
        f.write(f"  Effect size: {morning_effect:.4f}\n")
        morning_sig = "statistically significant" if morning_p < 0.05 else "not statistically significant"
        f.write(f"  Interpretation: The difference is {morning_sig} at the 0.05 level.\n\n")

        f.write("Evening Rush Hours:\n")
        f.write(f"  Mean PM2.5: {evening_rush_pm25.mean():.2f} µg/m³ (n={len(evening_rush_pm25)})\n")
        f.write(f"  T-test vs Non-Rush: t={evening_t:.4f}, p={evening_p:.4f}\n")
        f.write(f"  Effect size: {evening_effect:.4f}\n")
        evening_sig = "statistically significant" if evening_p < 0.05 else "not statistically significant"
        f.write(f"  Interpretation: The difference is {evening_sig} at the 0.05 level.\n\n")

        f.write("Morning vs Evening Rush:\n")
        f.write(f"  T-test: t={morning_vs_evening_t:.4f}, p={morning_vs_evening_p:.4f}\n")
        morning_evening_sig = "statistically significant" if morning_vs_evening_p < 0.05 else "not statistically significant"
        f.write(f"  Interpretation: The difference is {morning_evening_sig} at the 0.05 level.\n\n")

    # Create enhanced visualizations

    # 1. Hour-by-hour PM2.5 levels with rush hours highlighted
    plt.figure(figsize=(14, 8))

    # Plot bar chart with color-coded rush hour periods
    colors = hourly_pm25['rush_category'].map({
        'Morning Rush': '#FF9999',
        'Evening Rush': '#FF9999',
        'Non-Rush': '#99CCFF'
    })

    # Main bar chart
    bars = plt.bar(hourly_pm25['hour'], hourly_pm25['mean'],
                   yerr=hourly_pm25['std'] / np.sqrt(hourly_pm25['count']),  # Add standard error bars
                   color=colors, edgecolor='black', linewidth=1,
                   error_kw={'capsize': 5, 'capthick': 1, 'ecolor': 'black'})

    # Add count annotations above each bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        count = hourly_pm25.iloc[i]['count']
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.3,
                 f'{count}', ha='center', va='bottom', fontsize=8)

    # Add horizontal lines for overall means
    plt.axhline(y=rush_pm25.mean(), color='red', linestyle='--',
                label=f'Rush Hour Mean: {rush_pm25.mean():.2f} µg/m³')
    plt.axhline(y=non_rush_pm25.mean(), color='blue', linestyle='--',
                label=f'Non-Rush Hour Mean: {non_rush_pm25.mean():.2f} µg/m³')

    # Add annotations for rush hour periods
    plt.axvspan(morning_rush[0] - 0.5, morning_rush[-1] + 0.5, alpha=0.2, color='red')
    plt.axvspan(evening_rush[0] - 0.5, evening_rush[-1] + 0.5, alpha=0.2, color='red')

    plt.text(morning_rush[0] + 1, max(hourly_pm25['mean']) * 0.9, 'Morning\nRush',
             ha='center', va='center', weight='bold', bbox=dict(facecolor='white', alpha=0.7))
    plt.text(evening_rush[0] + 1, max(hourly_pm25['mean']) * 0.9, 'Evening\nRush',
             ha='center', va='center', weight='bold', bbox=dict(facecolor='white', alpha=0.7))

    # Add p-value annotation
    plt.annotate(f'T-test: p={p_val:.4f}{"*" if p_val < 0.05 else " (not significant)"}',
                 xy=(0.5, 0.95), xycoords='axes fraction',
                 ha='center', va='center',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

    # Customize plot
    plt.title('Average PM2.5 Levels by Hour with Rush Hours Highlighted (PST)', fontsize=16)
    plt.xlabel('Hour of Day (24-hour format, PST)', fontsize=14)
    plt.ylabel('PM2.5 (µg/m³)', fontsize=14)
    plt.xticks(range(24), [f'{h:02d}:00' for h in range(24)], rotation=45)
    plt.legend(loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save figure
    hourly_plot_path = get_output_path('pm25_by_hour_with_rush_periods.png')
    plt.savefig(hourly_plot_path, dpi=300)
    print(f"Saved hourly PM2.5 plot to {hourly_plot_path}")

    # 2. Aggregated comparison of rush vs non-rush hours with statistical info
    plt.figure(figsize=(10, 8))

    # Prepare data for categorical plots
    categories = ['Morning Rush', 'Evening Rush', 'All Rush Hours', 'Non-Rush Hours']
    means = [morning_rush_pm25.mean(), evening_rush_pm25.mean(), rush_pm25.mean(), non_rush_pm25.mean()]

    # Calculate standard errors for error bars
    errors = [
        morning_rush_pm25.std() / np.sqrt(len(morning_rush_pm25)),
        evening_rush_pm25.std() / np.sqrt(len(evening_rush_pm25)),
        rush_pm25.std() / np.sqrt(len(rush_pm25)),
        non_rush_pm25.std() / np.sqrt(len(non_rush_pm25))
    ]

    # Sample counts for labels
    counts = [len(morning_rush_pm25), len(evening_rush_pm25), len(rush_pm25), len(non_rush_pm25)]

    # Create bar plot
    bars = plt.bar(categories, means, yerr=errors,
                   color=['#FF9999', '#FF9999', '#FF5555', '#99CCFF'],
                   edgecolor='black', linewidth=1,
                   error_kw={'capsize': 5, 'capthick': 1, 'ecolor': 'black'})

    # Add count annotations above each bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.3,
                 f'n = {counts[i]}', ha='center', va='bottom')

    # Add p-value annotations
    plt.annotate(f'Morning vs Non-Rush: p={morning_p:.4f}{"*" if morning_p < 0.05 else ""}',
                 xy=(0, means[0]), xytext=(0, means[0] + 2),
                 ha='center', arrowprops=dict(arrowstyle='->'))

    plt.annotate(f'Evening vs Non-Rush: p={evening_p:.4f}{"*" if evening_p < 0.05 else ""}',
                 xy=(1, means[1]), xytext=(1, means[1] + 2),
                 ha='center', arrowprops=dict(arrowstyle='->'))

    plt.annotate(f'All Rush vs Non-Rush: p={p_val:.4f}{"*" if p_val < 0.05 else ""}',
                 xy=(2, means[2]), xytext=(2, means[2] + 2),
                 ha='center', arrowprops=dict(arrowstyle='->'))

    # Customize plot
    plt.title('PM2.5 Levels: Rush Hour vs. Non-Rush Hour Comparison (PST)', fontsize=16)
    plt.ylabel('PM2.5 (µg/m³)', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add a note about significance
    sig_note = "* p < 0.05 indicates statistical significance"
    plt.figtext(0.5, 0.01, sig_note, ha='center', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Make room for the note at the bottom

    # Save figure
    comparison_plot_path = get_output_path('rush_hour_comparison_with_stats.png')
    plt.savefig(comparison_plot_path, dpi=300)
    print(f"Saved rush hour comparison plot to {comparison_plot_path}")

    # 3. Boxplot to show distribution and outliers
    plt.figure(figsize=(10, 8))

    # Create dataframe for boxplot
    boxplot_data = []
    for hour in range(24):
        is_rush = hour in rush_hours
        rush_type = "Morning Rush" if hour in morning_rush else "Evening Rush" if hour in evening_rush else "Non-Rush"
        hour_data = air_df[air_df['hour'] == hour]['pm2_5']
        for value in hour_data:
            boxplot_data.append({
                'hour': hour,
                'pm2_5': value,
                'is_rush_hour': is_rush,
                'rush_type': rush_type
            })

    boxplot_df = pd.DataFrame(boxplot_data)

    # Create boxplot with Seaborn
    sns.boxplot(x='hour', y='pm2_5', hue='rush_type',
                data=boxplot_df,
                palette={'Morning Rush': '#FF9999', 'Evening Rush': '#FF9999', 'Non-Rush': '#99CCFF'},
                fliersize=3)

    # Customize plot
    plt.title('PM2.5 Distribution by Hour with Rush Hour Periods (PST)', fontsize=16)
    plt.xlabel('Hour of Day (24-hour format, PST)', fontsize=14)
    plt.ylabel('PM2.5 (µg/m³)', fontsize=14)
    plt.xticks(range(24), [f'{h:02d}:00' for h in range(24)], rotation=45)

    # Add rush hour shading
    plt.axvspan(morning_rush[0] - 0.5, morning_rush[-1] + 0.5, alpha=0.2, color='red')
    plt.axvspan(evening_rush[0] - 0.5, evening_rush[-1] + 0.5, alpha=0.2, color='red')

    plt.tight_layout()

    # Save figure
    boxplot_path = get_output_path('pm25_distribution_by_hour.png')
    plt.savefig(boxplot_path, dpi=300)
    print(f"Saved PM2.5 distribution boxplot to {boxplot_path}")

    return results_file


# Run the enhanced rush hour analysis
try:
    results_file = rush_hour_analysis()
    print(f"\nRush hour analysis complete! Results saved to {results_file}")
    print("Visualizations saved to output directory.")
except Exception as e:
    print(f"Error in rush hour analysis: {str(e)}")
    traceback.print_exc()