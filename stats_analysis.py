import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import pytz
import os
import traceback

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
        print(f"Using pre-calculated PST hour information for {df.shape[0]} records")
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

# Filter out values more than 4 standard deviations from the mean)
air_df = air_df[air_df['pm2_5'] <= (pm25_mean + 4 * pm25_std)]

# Alternative approach: use absolute threshold based on EPA standards
# PM2.5 readings above 500 μg/m³ are extremely rare even in severe pollution events
air_df = air_df[air_df['pm2_5'] <= 500]

print(f"Removed extreme PM2.5 outliers")

# Create a congestion factor (ratio of free flow speed to current speed)
try:
    # Make a copy to avoid chained assignment warnings
    traffic_df_temp = traffic_df.copy()

    # Handle potential zeros in speed column
    traffic_df_temp['speed_safe'] = traffic_df_temp['speed'].replace(0, np.nan)
    traffic_df_temp['congestion_factor'] = traffic_df_temp['free_flow_speed'] / traffic_df_temp['speed_safe']

    # Replace inf values (from div/0) with NaN
    traffic_df_temp['congestion_factor'] = traffic_df_temp['congestion_factor'].replace([np.inf, -np.inf], np.nan)

    # Fill NaNs with 1.0 (no congestion) or you can choose another approach
    traffic_df_temp['congestion_factor'] = traffic_df_temp['congestion_factor'].fillna(1.0)

    # Cap extreme values to avoid outliers dominating analysis
    congestion_cap = 5.0  # 5x normal speed is already severe congestion
    traffic_df_temp['congestion_factor'] = traffic_df_temp['congestion_factor'].clip(upper=congestion_cap)

    # Copy back to original dataframe
    traffic_df = traffic_df_temp

    print(f"Created congestion factor for {len(traffic_df)} traffic records")
except Exception as e:
    print(f"Error creating congestion factor: {str(e)}")
    # Create a default congestion factor of 1.0 if calculation fails
    traffic_df['congestion_factor'] = 1.0


# Function to apply the minimum sample size threshold during EDA
def run_eda():
    """Exploratory Data Analysis with sample size threshold enforcement"""
    print("Exploratory Data Analysis:")
    print("\nTraffic Data Summary:")
    print(traffic_df[['speed', 'free_flow_speed', 'congestion_factor']].describe())

    print("\nAir Quality Data Summary:")
    print(air_df[['pm2_5', 'temperature', 'humidity', 'pressure']].describe())

    # Save summary statistics to files
    traffic_summary = traffic_df[['speed', 'free_flow_speed', 'congestion_factor']].describe()
    air_summary = air_df[['pm2_5', 'temperature', 'humidity', 'pressure']].describe()

    traffic_summary.to_csv(get_output_path('traffic_summary.csv'))
    air_summary.to_csv(get_output_path('air_quality_summary.csv'))

    # Define minimum sample threshold
    MIN_SAMPLES = 500

    # Traffic congestion by hour with sample size threshold
    hourly_congestion_raw = traffic_df.groupby('hour')['congestion_factor'].agg(['mean', 'count']).reset_index()

    # Create a new DataFrame with all hours and apply threshold
    hourly_congestion = pd.DataFrame({'hour': range(24)})
    hourly_congestion = hourly_congestion.merge(hourly_congestion_raw, on='hour', how='left')

    # Fill NaN values with 0 for visualization
    hourly_congestion['mean'] = hourly_congestion['mean'].fillna(0)
    hourly_congestion['count'] = hourly_congestion['count'].fillna(0)

    # Apply the threshold - set mean to 0 for hours with insufficient samples
    hourly_congestion['mean_thresholded'] = hourly_congestion.apply(
        lambda row: row['mean'] if row['count'] >= MIN_SAMPLES else 0, axis=1)

    # Log which hours had insufficient data
    low_sample_hours = hourly_congestion[hourly_congestion['count'] < MIN_SAMPLES]['hour'].tolist()
    if low_sample_hours:
        print(f"Hours with fewer than {MIN_SAMPLES} traffic samples (values set to 0): {low_sample_hours}")

    # Save hourly traffic data with sample counts
    hourly_congestion.to_csv(get_output_path('hourly_traffic_with_counts.csv'), index=False)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(hourly_congestion['hour'], hourly_congestion['mean_thresholded'])
    plt.title('Average Traffic Congestion by Hour (PST)')
    plt.xlabel('Hour of Day (PST)')
    plt.ylabel('Congestion Factor')

    # Add sample count annotations to bars
    for i, bar in enumerate(bars):
        count = hourly_congestion.iloc[i]['count']
        height = bar.get_height()
        # Display the count, or "insufficient data" for low-sample hours
        if count < MIN_SAMPLES:
            plt.text(bar.get_x() + bar.get_width() / 2, 0.02,
                     f"n={int(count)}\n<{MIN_SAMPLES}",
                     ha='center', va='bottom', color='red', fontsize=8, rotation=90)
        else:
            plt.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                     f"n={int(count)}",
                     ha='center', va='bottom', fontsize=8)

    # Highlight rush hour periods
    morning_rush = [7, 8, 9]  # 7-9 AM
    evening_rush = [16, 17, 18]  # 4-6 PM
    plt.axvspan(morning_rush[0] - 0.5, morning_rush[-1] + 0.5, alpha=0.2, color='red')
    plt.axvspan(evening_rush[0] - 0.5, evening_rush[-1] + 0.5, alpha=0.2, color='red')

    plt.text(morning_rush[0] + 1, hourly_congestion['mean_thresholded'].max() * 0.9, 'Morning\nRush',
             ha='center', va='center', weight='bold', bbox=dict(facecolor='white', alpha=0.7))
    plt.text(evening_rush[0] + 1, hourly_congestion['mean_thresholded'].max() * 0.9, 'Evening\nRush',
             ha='center', va='center', weight='bold', bbox=dict(facecolor='white', alpha=0.7))

    plt.xticks(range(24), [f'{h:02d}:00' for h in range(24)], rotation=45)
    plt.tight_layout()
    plt.savefig(get_output_path('traffic_by_hour.png'))

    # PM2.5 by hour with sample size threshold
    hourly_pm25_raw = air_df.groupby('hour')['pm2_5'].agg(['mean', 'count']).reset_index()

    # Create a new DataFrame with all hours and apply threshold
    hourly_pm25 = pd.DataFrame({'hour': range(24)})
    hourly_pm25 = hourly_pm25.merge(hourly_pm25_raw, on='hour', how='left')

    # Fill NaN values with 0 for visualization
    hourly_pm25['mean'] = hourly_pm25['mean'].fillna(0)
    hourly_pm25['count'] = hourly_pm25['count'].fillna(0)

    # Apply the threshold - set mean to 0 for hours with insufficient samples
    hourly_pm25['mean_thresholded'] = hourly_pm25.apply(
        lambda row: row['mean'] if row['count'] >= MIN_SAMPLES else 0, axis=1)

    # Log which hours had insufficient data
    low_sample_hours = hourly_pm25[hourly_pm25['count'] < MIN_SAMPLES]['hour'].tolist()
    if low_sample_hours:
        print(f"Hours with fewer than {MIN_SAMPLES} air quality samples (values set to 0): {low_sample_hours}")

    # Save hourly air quality data with sample counts
    hourly_pm25.to_csv(get_output_path('hourly_pm25_with_counts.csv'), index=False)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(hourly_pm25['hour'], hourly_pm25['mean_thresholded'])
    plt.title('Average PM2.5 by Hour (PST)')
    plt.xlabel('Hour of Day (PST)')
    plt.ylabel('PM2.5')

    # Add sample count annotations to bars
    for i, bar in enumerate(bars):
        count = hourly_pm25.iloc[i]['count']
        height = bar.get_height()
        # Display the count, or "insufficient data" for low-sample hours
        if count < MIN_SAMPLES:
            plt.text(bar.get_x() + bar.get_width() / 2, 0.02,
                     f"n={int(count)}\n<{MIN_SAMPLES}",
                     ha='center', va='bottom', color='red', fontsize=8, rotation=90)
        else:
            plt.text(bar.get_x() + bar.get_width() / 2, height + 0.1,
                     f"n={int(count)}",
                     ha='center', va='bottom', fontsize=8)

    # Highlight rush hour periods
    plt.axvspan(morning_rush[0] - 0.5, morning_rush[-1] + 0.5, alpha=0.2, color='red')
    plt.axvspan(evening_rush[0] - 0.5, evening_rush[-1] + 0.5, alpha=0.2, color='red')

    plt.text(morning_rush[0] + 1, hourly_pm25['mean_thresholded'].max() * 0.9, 'Morning\nRush',
             ha='center', va='center', weight='bold', bbox=dict(facecolor='white', alpha=0.7))
    plt.text(evening_rush[0] + 1, hourly_pm25['mean_thresholded'].max() * 0.9, 'Evening\nRush',
             ha='center', va='center', weight='bold', bbox=dict(facecolor='white', alpha=0.7))

    plt.xticks(range(24), [f'{h:02d}:00' for h in range(24)], rotation=45)
    plt.tight_layout()
    plt.savefig(get_output_path('pm25_by_hour.png'))

    # Return the modified hourly data for further analysis
    return hourly_congestion, hourly_pm25

# 2. Modified Temporal Correlation Analysis to use thresholded data
def temporal_correlation(hourly_congestion, hourly_pm25):
    print("\nTemporal Correlation Analysis:")

    # Merge hourly data using the thresholded values
    hourly_merged = pd.merge(
        hourly_congestion[['hour', 'mean_thresholded', 'count']],
        hourly_pm25[['hour', 'mean_thresholded', 'count']],
        on='hour',
        suffixes=('_traffic', '_pm25')
    )

    # Filter out hours where either dataset has insufficient samples
    MIN_SAMPLES = 500
    hourly_merged_filtered = hourly_merged[
        (hourly_merged['count_traffic'] >= MIN_SAMPLES) &
        (hourly_merged['count_pm25'] >= MIN_SAMPLES)
        ]

    print(f"Hours with sufficient data in both datasets: {len(hourly_merged_filtered)} out of 24")

    # Save merged hourly data
    hourly_merged.to_csv(get_output_path('hourly_merged_data.csv'), index=False)
    hourly_merged_filtered.to_csv(get_output_path('hourly_merged_data_filtered.csv'), index=False)

    # Calculate correlation only if we have enough data points
    if len(hourly_merged_filtered) >= 5:  # Minimum needed for meaningful correlation
        correlation, p_value = stats.pearsonr(
            hourly_merged_filtered['mean_thresholded_traffic'],
            hourly_merged_filtered['mean_thresholded_pm25']
        )
        print(f"Hourly correlation between congestion and PM2.5: r={correlation:.4f}, p={p_value:.4f}")

        # Save correlation results
        with open(get_output_path('temporal_correlation_results.txt'), 'w') as f:
            f.write(f"Hourly correlation between congestion and PM2.5: r={correlation:.4f}, p={p_value:.4f}\n")
            f.write(f"R-squared: {correlation ** 2:.4f}\n")
            f.write(f"Based on {len(hourly_merged_filtered)} hours with sufficient data (≥{MIN_SAMPLES} samples)")

        # Visualize relationship
        plt.figure(figsize=(8, 6))
        plt.scatter(hourly_merged_filtered['mean_thresholded_traffic'],
                    hourly_merged_filtered['mean_thresholded_pm25'])

        # Add hour labels to points
        for idx, row in hourly_merged_filtered.iterrows():
            plt.annotate(f"{int(row['hour']):02d}:00",
                         (row['mean_thresholded_traffic'], row['mean_thresholded_pm25']),
                         textcoords="offset points",
                         xytext=(0, 7),
                         ha='center')

        plt.title(f'PM2.5 vs Traffic Congestion (r={correlation:.4f}, p={p_value:.4f})')
        plt.xlabel('Traffic Congestion Factor')
        plt.ylabel('PM2.5 Level')
        plt.grid(True, alpha=0.3)

        # Add note about threshold
        plt.figtext(0.5, 0.01,
                    f"Note: Only showing hours with ≥{MIN_SAMPLES} samples in both datasets ({len(hourly_merged_filtered)} of 24 hours)",
                    ha='center', fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Make room for the note
        plt.savefig(get_output_path('correlation_plot.png'))

        return correlation, p_value
    else:
        print("Insufficient hours with enough data for meaningful correlation analysis")

        # Save info about the issue
        with open(get_output_path('temporal_correlation_results.txt'), 'w') as f:
            f.write("Insufficient hours with enough data for meaningful correlation analysis\n")
            f.write(f"Only {len(hourly_merged_filtered)} hours had ≥{MIN_SAMPLES} samples in both datasets\n")
            f.write("Need at least 5 valid hours for correlation calculation")

        # Create a simplified visualization showing the data limitation
        plt.figure(figsize=(8, 6))
        plt.scatter(hourly_merged['mean_thresholded_traffic'],
                    hourly_merged['mean_thresholded_pm25'],
                    alpha=0.5, c='lightgray')

        # Mark valid points in a different color
        if not hourly_merged_filtered.empty:
            plt.scatter(hourly_merged_filtered['mean_thresholded_traffic'],
                        hourly_merged_filtered['mean_thresholded_pm25'],
                        c='blue', zorder=5)

        plt.title('PM2.5 vs Traffic Congestion (Insufficient data for correlation)')
        plt.xlabel('Traffic Congestion Factor')
        plt.ylabel('PM2.5 Level')
        plt.grid(True, alpha=0.3)

        plt.figtext(0.5, 0.5,
                    f"Insufficient data for correlation analysis\nOnly {len(hourly_merged_filtered)} hours have ≥{MIN_SAMPLES} samples",
                    ha='center', va='center', fontsize=12,
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

        plt.savefig(get_output_path('correlation_plot_insufficient_data.png'))

        return None, None


# 3. Spatial Analysis
def spatial_analysis():
    print("\nSpatial Analysis:")

    # Group by geographic grid cells
    def assign_grid_cell(lat, lon, grid_size=0.01):
        # Handle NaN values
        if pd.isna(lat) or pd.isna(lon):
            return None

        try:
            # Convert to float and round to create grid cell
            lat_cell = round(float(lat) / grid_size) * grid_size
            lon_cell = round(float(lon) / grid_size) * grid_size
            return f"{lat_cell:.3f}_{lon_cell:.3f}"
        except (ValueError, TypeError):
            print(f"Warning: Could not process coordinate values: lat={lat}, lon={lon}")
            return None

    # For traffic data, use starting point with error handling
    traffic_df['grid_cell'] = traffic_df.apply(
        lambda row: assign_grid_cell(
            row.get('lat_start') if 'lat_start' in row else row.get('lat'),
            row.get('lon_start') if 'lon_start' in row else row.get('lon')
        ), axis=1)

    # For air quality data with error handling
    air_df['grid_cell'] = air_df.apply(
        lambda row: assign_grid_cell(row.get('lat'), row.get('lon')),
        axis=1)

    # Remove rows with None grid cells
    traffic_with_grid = traffic_df.dropna(subset=['grid_cell'])
    air_with_grid = air_df.dropna(subset=['grid_cell'])

    print(f"Traffic data: {len(traffic_df)} rows, {len(traffic_with_grid)} with valid coordinates")
    print(f"Air quality data: {len(air_df)} rows, {len(air_with_grid)} with valid coordinates")

    # Aggregate by grid cell
    traffic_by_cell = traffic_with_grid.groupby('grid_cell')['congestion_factor'].mean().reset_index()
    air_by_cell = air_with_grid.groupby('grid_cell')['pm2_5'].mean().reset_index()

    # Save grid cell data
    traffic_by_cell.to_csv(get_output_path('traffic_by_grid_cell.csv'), index=False)
    air_by_cell.to_csv(get_output_path('air_by_grid_cell.csv'), index=False)

    # Merge data
    spatial_merged = pd.merge(traffic_by_cell, air_by_cell, on='grid_cell', how='inner')
    spatial_merged.to_csv(get_output_path('spatial_merged_data.csv'), index=False)

    # Calculate spatial correlation
    if len(spatial_merged) > 5:  # Ensure we have enough matching cells
        spatial_corr, spatial_p = stats.pearsonr(spatial_merged['congestion_factor'],
                                                 spatial_merged['pm2_5'])
        print(f"Spatial correlation: r={spatial_corr:.4f}, p={spatial_p:.4f}")
        print(f"Matched {len(spatial_merged)} geographic areas")

        # Save correlation results
        with open(get_output_path('spatial_correlation_results.txt'), 'w') as f:
            f.write(f"Spatial correlation: r={spatial_corr:.4f}, p={spatial_p:.4f}\n")
            f.write(f"Matched {len(spatial_merged)} geographic areas\n")
            f.write(f"R-squared: {spatial_corr ** 2:.4f}")

        # Visualize spatial correlation
        plt.figure(figsize=(8, 6))
        plt.scatter(spatial_merged['congestion_factor'], spatial_merged['pm2_5'])
        plt.xlabel('Traffic Congestion Factor')
        plt.ylabel('PM2.5 Level')
        plt.title(f'Spatial Correlation: r={spatial_corr:.4f}, p={spatial_p:.4f}')
        plt.grid(True, alpha=0.3)
        plt.savefig(get_output_path('spatial_correlation.png'))
    else:
        print("Not enough matching geographic areas for meaningful spatial correlation")
        with open(get_output_path('spatial_correlation_results.txt'), 'w') as f:
            f.write("Not enough matching geographic areas for meaningful spatial correlation\n")
            f.write(f"Only found {len(spatial_merged)} matching areas")

    return spatial_merged


# 4. Time-Matched Analysis
def time_matched_analysis():
    print("\nTime-Matched Analysis:")

    try:
        # Round timestamps for matching - using prepared timestamp columns
        if 'timestamp_pst' in traffic_df.columns and 'timestamp_pst' in air_df.columns:
            # Already have timezone-aware timestamps
            traffic_df['hour_timestamp'] = traffic_df['timestamp_pst'].dt.floor('H')
            air_df['hour_timestamp'] = air_df['timestamp_pst'].dt.floor('H')
        else:
            # Use hour column to create timestamps
            base_date = datetime(2025, 3, 1, tzinfo=pytz.timezone('America/Los_Angeles'))

            # Function to create hour timestamp
            def create_hour_ts(hour):
                return base_date.replace(hour=hour)

            traffic_df['hour_timestamp'] = traffic_df['hour'].apply(create_hour_ts)
            air_df['hour_timestamp'] = air_df['hour'].apply(create_hour_ts)

        # Group by hour
        hourly_traffic = traffic_df.groupby('hour')['congestion_factor'].mean().reset_index()
        hourly_air = air_df.groupby('hour')['pm2_5'].mean().reset_index()

        # Save hourly data
        hourly_traffic.to_csv(get_output_path('hourly_traffic.csv'), index=False)
        hourly_air.to_csv(get_output_path('hourly_air.csv'), index=False)

        # Merge on matching hours
        time_matched = pd.merge(hourly_traffic, hourly_air, on='hour')
        time_matched.to_csv(get_output_path('time_matched_data.csv'), index=False)

        # Calculate correlation
        if len(time_matched) > 5:  # Ensure we have enough time points
            time_corr, time_p = stats.pearsonr(time_matched['congestion_factor'], time_matched['pm2_5'])
            print(f"Time-matched correlation: r={time_corr:.4f}, p={time_p:.4f}")
            print(f"Matched {len(time_matched)} time points")

            # Save correlation results
            with open(get_output_path('time_matched_correlation_results.txt'), 'w') as f:
                f.write(f"Time-matched correlation: r={time_corr:.4f}, p={time_p:.4f}\n")
                f.write(f"Matched {len(time_matched)} time points\n")
                f.write(f"R-squared: {time_corr ** 2:.4f}")

            # Visualize time-matched correlation
            plt.figure(figsize=(8, 6))
            plt.scatter(time_matched['congestion_factor'], time_matched['pm2_5'])
            plt.xlabel('Traffic Congestion Factor')
            plt.ylabel('PM2.5 Level')
            plt.title(f'Time-Matched Correlation: r={time_corr:.4f}, p={time_p:.4f}')
            plt.grid(True, alpha=0.3)
            plt.savefig(get_output_path('time_matched_correlation.png'))
        else:
            print("Not enough matching time points for meaningful temporal correlation")
            with open(get_output_path('time_matched_correlation_results.txt'), 'w') as f:
                f.write("Not enough matching time points for meaningful temporal correlation\n")
                f.write(f"Only found {len(time_matched)} matching time points")

        return time_matched
    except Exception as e:
        print(f"Error in time-matched analysis: {str(e)}")
        import traceback
        traceback.print_exc()

        # Log the error to a file
        with open(get_output_path('time_matched_analysis_error.log'), 'w') as f:
            f.write(f"Error in time-matched analysis: {str(e)}\n")
            traceback.print_exc(file=f)

        return None


# 5. Regression Analysis
def regression_analysis(time_matched):
    print("\nRegression Analysis:")

    if time_matched is None or len(time_matched) < 10:
        print("Not enough data points for meaningful regression analysis")
        with open(get_output_path('regression_analysis_results.txt'), 'w') as f:
            f.write("Not enough data points for meaningful regression analysis\n")
            f.write(f"Only found {0 if time_matched is None else len(time_matched)} data points")
        return None

    # Prepare data
    X = time_matched[['congestion_factor']]
    y = time_matched['pm2_5']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print(f"Linear Regression Results:")
    print(f"Coefficient: {model.coef_[0]:.4f}")
    print(f"Intercept: {model.intercept_:.4f}")
    print(f"R² (train): {train_score:.4f}")
    print(f"R² (test): {test_score:.4f}")

    # Save regression results
    with open(get_output_path('regression_analysis_results.txt'), 'w') as f:
        f.write("Linear Regression Results:\n")
        f.write(f"Coefficient: {model.coef_[0]:.4f}\n")
        f.write(f"Intercept: {model.intercept_:.4f}\n")
        f.write(f"R² (train): {train_score:.4f}\n")
        f.write(f"R² (test): {test_score:.4f}\n")
        f.write(f"\nEquation: PM2.5 = {model.coef_[0]:.4f} * Congestion + {model.intercept_:.4f}")

    # Plot regression line
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, alpha=0.5)
    plt.plot(X, model.predict(X), color='red', linewidth=2)
    plt.title('PM2.5 vs Traffic Congestion: Linear Regression')
    plt.xlabel('Traffic Congestion Factor')
    plt.ylabel('PM2.5 Level')
    plt.grid(True, alpha=0.3)
    plt.savefig(get_output_path('regression_plot.png'))

    return model


# 6. Statistical Tests
def statistical_tests():
    print("\nStatistical Tests:")

    # Create a results file
    results_file = get_output_path('statistical_tests_results.txt')
    with open(results_file, 'w') as f:
        f.write("Statistical Tests Results\n")
        f.write("=========================\n\n")

    # Test: Is PM2.5 higher during rush hours?
    rush_hours = [7, 8, 9, 16, 17, 18]  # 7-9 AM and 4-6 PM

    rush_hour_pm25 = air_df[air_df['hour'].isin(rush_hours)]['pm2_5']
    non_rush_hour_pm25 = air_df[~air_df['hour'].isin(rush_hours)]['pm2_5']

    # T-test
    t_stat, p_val = stats.ttest_ind(rush_hour_pm25, non_rush_hour_pm25, equal_var=False)

    print(f"Rush Hours PM2.5 (mean): {rush_hour_pm25.mean():.2f}")
    print(f"Non-Rush Hours PM2.5 (mean): {non_rush_hour_pm25.mean():.2f}")
    print(f"T-test: t={t_stat:.4f}, p={p_val:.4f}")

    # Write results to file
    with open(results_file, 'a') as f:
        f.write("Rush Hour vs Non-Rush Hour PM2.5 Levels\n")
        f.write(f"Rush Hours PM2.5 (mean): {rush_hour_pm25.mean():.2f}\n")
        f.write(f"Non-Rush Hours PM2.5 (mean): {non_rush_hour_pm25.mean():.2f}\n")
        f.write(f"T-test: t={t_stat:.4f}, p={p_val:.4f}\n")
        f.write(
            f"Interpretation: {'Statistically significant difference' if p_val < 0.05 else 'No statistically significant difference'}\n\n")

    # Visualize rush hour vs non-rush hour
    plt.figure(figsize=(8, 6))
    labels = ['Rush Hours', 'Non-Rush Hours']
    values = [rush_hour_pm25.mean(), non_rush_hour_pm25.mean()]

    bars = plt.bar(labels, values)
    plt.title('Average PM2.5 Levels: Rush Hours vs Non-Rush Hours')
    plt.ylabel('PM2.5 Level')

    # Add sample size and p-value annotations
    for i, bar in enumerate(bars):
        height = bar.get_height()
        count = len(rush_hour_pm25) if i == 0 else len(non_rush_hour_pm25)
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.3,
                 f'n = {count}', ha='center', va='bottom')

    plt.figtext(0.5, 0.01, f'T-test: p={p_val:.4f}{"*" if p_val < 0.05 else " (not significant)"}',
                ha='center', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(get_output_path('rush_hour_comparison.png'))

    # Test: Is congestion worse on weekdays?
    try:
        # Check if we have the required columns
        if 'is_weekend' not in traffic_df.columns:
            traffic_df['is_weekend'] = traffic_df['day_of_week'].isin([5, 6]).astype(int)

        weekday_congestion = traffic_df[traffic_df['is_weekend'] == 0]['congestion_factor']
        weekend_congestion = traffic_df[traffic_df['is_weekend'] == 1]['congestion_factor']

        if len(weekday_congestion) > 0 and len(weekend_congestion) > 0:
            t_stat2, p_val2 = stats.ttest_ind(weekday_congestion, weekend_congestion, equal_var=False)

            print(f"\nWeekday Congestion (mean): {weekday_congestion.mean():.2f}")
            print(f"Weekend Congestion (mean): {weekend_congestion.mean():.2f}")
            print(f"T-test: t={t_stat2:.4f}, p={p_val2:.4f}")

            # Write results to file
            with open(results_file, 'a') as f:
                f.write("Weekday vs Weekend Traffic Congestion\n")
                f.write(f"Weekday Congestion (mean): {weekday_congestion.mean():.2f}\n")
                f.write(f"Weekend Congestion (mean): {weekend_congestion.mean():.2f}\n")
                f.write(f"T-test: t={t_stat2:.4f}, p={p_val2:.4f}\n")
                f.write(
                    f"Interpretation: {'Statistically significant difference' if p_val2 < 0.05 else 'No statistically significant difference'}\n\n")

            # Visualize weekday vs weekend
            plt.figure(figsize=(8, 6))
            labels = ['Weekdays', 'Weekends']
            values = [weekday_congestion.mean(), weekend_congestion.mean()]

            bars = plt.bar(labels, values)
            plt.title('Average Traffic Congestion: Weekdays vs Weekends')
            plt.ylabel('Congestion Factor')

            # Add sample size and p-value annotations
            for i, bar in enumerate(bars):
                height = bar.get_height()
                count = len(weekday_congestion) if i == 0 else len(weekend_congestion)
                plt.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                         f'n = {count}', ha='center', va='bottom')

            plt.figtext(0.5, 0.01, f'T-test: p={p_val2:.4f}{"*" if p_val2 < 0.05 else " (not significant)"}',
                        ha='center', fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.savefig(get_output_path('weekday_weekend_comparison.png'))
        else:
            print("Not enough data to compare weekday vs weekend")
            with open(results_file, 'a') as f:
                f.write("Weekday vs Weekend Traffic Congestion\n")
                f.write("Not enough data to compare weekday vs weekend\n\n")
    except Exception as e:
        print(f"Error in weekday/weekend comparison: {str(e)}")
        with open(results_file, 'a') as f:
            f.write("Weekday vs Weekend Traffic Congestion\n")
            f.write(f"Error in analysis: {str(e)}\n\n")

    # Generate a summary of all statistical tests
    with open(results_file, 'a') as f:
        f.write("\nSummary of Statistical Tests\n")
        f.write("===========================\n")
        f.write("1. Rush Hour vs Non-Rush Hour PM2.5: ")
        f.write(f"{'Significant' if p_val < 0.05 else 'Not significant'} (p={p_val:.4f})\n")

        if 'p_val2' in locals():
            f.write("2. Weekday vs Weekend Congestion: ")
            f.write(f"{'Significant' if p_val2 < 0.05 else 'Not significant'} (p={p_val2:.4f})\n")

    print(f"\nStatistical test results saved to {results_file}")
    return results_file


# Run all analyses
try:
    print("Starting analysis...")
    print(f"All output files will be saved to: {output_dir}")

    # Save configuration info
    with open(get_output_path('analysis_info.txt'), 'w') as f:
        f.write(f"Statistical Analysis Run on {datetime.now()}\n")
        f.write(f"Traffic data records: {len(traffic_df)}\n")
        f.write(f"Air quality data records: {len(air_df)}\n")
        f.write(f"Output directory: {output_dir}\n")

    hourly_cong, hourly_pm = run_eda()
    print("EDA completed.")

    temporal_correlation(hourly_cong, hourly_pm)
    print("Temporal correlation completed.")

    try:
        spatial_data = spatial_analysis()
        print("Spatial analysis completed.")
    except Exception as e:
        print(f"Error in spatial analysis: {str(e)}")
        with open(get_output_path('spatial_analysis_error.log'), 'w') as f:
            f.write(f"Error in spatial analysis: {str(e)}\n")
            traceback.print_exc(file=f)

    try:
        time_data = time_matched_analysis()
        print("Time-matched analysis completed.")

        if time_data is not None and len(time_data) > 10:
            regression_analysis(time_data)
            print("Regression analysis completed.")
        else:
            print("Skipping regression analysis due to insufficient matched data.")
            with open(get_output_path('regression_analysis_skipped.log'), 'w') as f:
                f.write("Regression analysis skipped due to insufficient matched data.\n")
    except Exception as e:
        print(f"Error in time analysis: {str(e)}")
        with open(get_output_path('time_analysis_error.log'), 'w') as f:
            f.write(f"Error in time analysis: {str(e)}\n")
            traceback.print_exc(file=f)

    try:
        statistical_tests()
        print("Statistical tests completed.")
    except Exception as e:
        print(f"Error in statistical tests: {str(e)}")
        with open(get_output_path('statistical_tests_error.log'), 'w') as f:
            f.write(f"Error in statistical tests: {str(e)}\n")
            traceback.print_exc(file=f)

    # Create a completion marker file
    with open(get_output_path('analysis_complete.txt'), 'w') as f:
        f.write(f"Analysis completed successfully at {datetime.now()}\n")

    print(f"\nAnalysis completed successfully! All results saved to {output_dir}")
except Exception as e:
    print(f"Error during analysis: {str(e)}")
    with open(get_output_path('analysis_error.log'), 'w') as f:
        f.write(f"Error during analysis: {str(e)}\n")
        traceback.print_exc(file=f)
    import traceback

    traceback.print_exc()
