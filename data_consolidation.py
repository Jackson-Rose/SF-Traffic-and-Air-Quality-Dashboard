import json
import os
import pandas as pd
from glob import glob
from datetime import datetime
import pytz

# Define paths to your data files using recursive search
traffic_files = glob('data/traffic/2025/*/*.json')  # Recursively find all JSON files in month folders
air_quality_files = glob('data/purpleair/2025/*/*.json')  # Recursively find all JSON files in month folders

# Alternative option - try both naming conventions
if not air_quality_files:
    air_quality_files = glob('data/environmental/2025/*/*.json')
    print("Using 'environmental' folder for air quality data")

print(f"Found {len(traffic_files)} traffic files")
print(f"Found {len(air_quality_files)} air quality files")

# Print a few example paths to verify correct matching
if traffic_files:
    print("Example traffic files:")
    for file in traffic_files[:3]:  # Show first 3 examples
        print(f"  - {file}")

if air_quality_files:
    print("Example air quality files:")
    for file in air_quality_files[:3]:  # Show first 3 examples
        print(f"  - {file}")


# Function to consolidate JSON files with timezone conversion
def consolidate_json_files(file_list, output_filename):
    all_data = []

    # Define timezone objects
    utc_tz = pytz.UTC
    pst_tz = pytz.timezone('America/Los_Angeles')

    for file_path in file_list:
        try:
            with open(file_path, 'r') as file:
                json_data = json.load(file)

                # Extract data array and add source filename
                if 'data' in json_data:
                    for item in json_data['data']:
                        # Add source filename to each record
                        item['source_file'] = os.path.basename(file_path)

                        # Convert timestamp from UTC to PST if it exists
                        if 'timestamp' in item:
                            try:
                                # Parse the timestamp string to a datetime object
                                # Handle both formats: with and without timezone info
                                timestamp_str = item['timestamp']
                                if '+00:00' in timestamp_str or 'Z' in timestamp_str:
                                    # Already has UTC timezone info
                                    dt_utc = pd.to_datetime(timestamp_str)
                                else:
                                    # Assume UTC if no timezone info provided
                                    dt_utc = pd.to_datetime(timestamp_str).replace(tzinfo=utc_tz)

                                # Convert to PST
                                dt_pst = dt_utc.astimezone(pst_tz)

                                # Store the PST timestamp
                                item['timestamp'] = dt_pst.isoformat()

                                # Add explicit timezone columns for analysis
                                item['hour_pst'] = dt_pst.hour
                                item['day_of_week_pst'] = dt_pst.dayofweek
                                item['is_weekend_pst'] = 1 if dt_pst.dayofweek >= 5 else 0

                            except Exception as e:
                                print(f"Warning: Error converting timestamp for item: {e}")

                    all_data.extend(json_data['data'])
                else:
                    print(f"Warning: No 'data' field found in {file_path}")

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

    # Create the consolidated data structure
    consolidated = {
        'timestamp': datetime.now().isoformat(),
        'data': all_data
    }

    # Save to a new file
    with open(output_filename, 'w') as outfile:
        json.dump(consolidated, outfile, indent=2)

    print(f"Consolidated {len(all_data)} records from {len(file_list)} files into {output_filename}")

    # Also convert to pandas DataFrame and save as CSV for easier analysis
    df = pd.DataFrame(all_data)
    csv_filename = output_filename.replace('.json', '.csv')
    df.to_csv(csv_filename, index=False)
    print(f"Also saved as CSV: {csv_filename}")

    return df


# Consolidate traffic data
traffic_df = consolidate_json_files(traffic_files, 'consolidated_traffic.json')

# Consolidate air quality data
air_df = consolidate_json_files(air_quality_files, 'consolidated_air_quality.json')

print(f"Traffic data shape: {traffic_df.shape}")
print(f"Air quality data shape: {air_df.shape}")

# Print timezone conversion summary
print("\nTimezone Conversion Summary:")
print("All timestamps have been converted from UTC to PST (America/Los_Angeles)")
print("Added columns: 'hour_pst', 'day_of_week_pst', 'is_weekend_pst' for easier analysis")
print("\nSample PST hours distribution (traffic data):")
if 'hour_pst' in traffic_df.columns:
    hour_counts = traffic_df['hour_pst'].value_counts().sort_index()
    for hour, count in hour_counts.items():
        print(f"  {hour:02d}:00 - {count} records")
