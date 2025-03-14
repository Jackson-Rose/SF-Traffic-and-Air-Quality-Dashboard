import streamlit as st
import asyncio
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, Optional
import os
import json
import shutil
import pytz
from json_handler import JSONEncoder
from source_implementations import TomTomTrafficSource, PurpleAirSource

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class JSONStorageConfig:
    """Configuration for JSON file storage"""

    def __init__(self,
                 base_path: str = "data",
                 max_file_size_mb: int = 10,
                 create_backup: bool = True):
        self.base_path = base_path
        self.max_file_size = max_file_size_mb * 1024 * 1024  # Convert MB to bytes
        self.create_backup = create_backup
        self.logger = logging.getLogger(__name__)

        # Create directory structure
        self._setup_directory_structure()

    def _setup_directory_structure(self):
        """Create necessary directories for data storage"""
        # Main data directories
        os.makedirs(os.path.join(self.base_path, "traffic"), exist_ok=True)
        os.makedirs(os.path.join(self.base_path, "environmental"), exist_ok=True)

        # Backup directory
        if self.create_backup:
            os.makedirs(os.path.join(self.base_path, "backup"), exist_ok=True)


class JSONStorageHandler:
    """Handler for JSON file storage with advanced features"""

    def __init__(self, config: JSONStorageConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def get_storage_path(self, source_name: str, timestamp: datetime) -> str:
        """Generate appropriate storage path based on data source"""
        # Determine data type from source name
        if "traffic" in source_name.lower():
            data_type = "traffic"
        elif "environmental" in source_name.lower() or "purpleair" in source_name.lower():
            data_type = "environmental"
        else:
            data_type = "other"

        # Create yyyy/mm directory structure
        year_month = timestamp.strftime("%Y/%m")
        directory = os.path.join(self.config.base_path, data_type, year_month)
        os.makedirs(directory, exist_ok=True)

        # Generate filename with date
        filename = f"{source_name}_{timestamp.strftime('%Y-%m-%d')}.json"
        return os.path.join(directory, filename)

    async def save_data(self, data: pd.DataFrame, source_name: str):
        """Save data to JSON file with error handling and backup"""
        try:
            timestamp = datetime.now()
            filepath = self.get_storage_path(source_name, timestamp)

            # Prepare data for storage
            data_dict = {
                'timestamp': timestamp.isoformat(),
                'source': source_name,
                'data': data.to_dict('records')
            }

            # Check if file exists and needs rotation
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                if file_size >= self.config.max_file_size:
                    self._rotate_file(filepath)

                # Append to existing file
                with open(filepath, 'r+') as f:
                    try:
                        existing_data = json.load(f)
                        existing_data['data'].extend(data_dict['data'])
                        f.seek(0)
                        json.dump(existing_data, f, indent=2, cls=JSONEncoder)
                    except json.JSONDecodeError:
                        # Backup corrupted file and create new one
                        self._handle_corrupted_file(filepath)
                        with open(filepath, 'w') as new_f:
                            json.dump(data_dict, new_f, indent=2, cls=JSONEncoder)
            else:
                # Create new file
                with open(filepath, 'w') as f:
                    json.dump(data_dict, f, indent=2, cls=JSONEncoder)

            self.logger.info(f"Successfully saved data to {filepath}")
            return filepath

        except Exception as e:
            self.logger.error(f"Error saving data: {e}")
            raise

    def _rotate_file(self, filepath: str):
        """Rotate file when it exceeds size limit"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        directory = os.path.dirname(filepath)
        filename = os.path.basename(filepath)
        base, ext = os.path.splitext(filename)

        # Create rotated filename
        rotated_filepath = os.path.join(directory, f"{base}_{timestamp}{ext}")

        # Move current file to rotated name
        shutil.move(filepath, rotated_filepath)
        self.logger.info(f"Rotated file {filepath} to {rotated_filepath}")

        # Create backup if enabled
        if self.config.create_backup:
            backup_dir = os.path.join(self.config.base_path, "backup")
            backup_filepath = os.path.join(backup_dir, f"{base}_{timestamp}{ext}")
            shutil.copy2(rotated_filepath, backup_filepath)
            self.logger.info(f"Created backup at {backup_filepath}")

    def _handle_corrupted_file(self, filepath: str):
        """Handle corrupted JSON files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        corrupted_filepath = f"{filepath}.corrupted_{timestamp}"

        # Move corrupted file
        shutil.move(filepath, corrupted_filepath)
        self.logger.warning(f"Moved corrupted file to {corrupted_filepath}")

        # Create backup if enabled
        if self.config.create_backup:
            backup_dir = os.path.join(self.config.base_path, "backup")
            backup_filepath = os.path.join(
                backup_dir,
                f"corrupted_{os.path.basename(filepath)}_{timestamp}"
            )
            shutil.copy2(corrupted_filepath, backup_filepath)


class IntegratedDataPipeline:
    def __init__(self, storage_handler: JSONStorageHandler):
        self.storage_handler = storage_handler
        self.data_sources = []
        self.running = False
        self.cache = {}
        self._shutdown_event = asyncio.Event()
        self.tasks = []

        # Add timestamp tracking for data filtering
        self.data_timestamps = {}
        self.last_refresh_time = None

    def add_data_source(self, source):
        """Add a data source to the pipeline"""
        self.data_sources.append(source)
        # Initialize timestamp tracking
        self.data_timestamps[source.name] = []

    async def cleanup(self):
        """Cleanup all resources"""
        logger.info("Cleaning up pipeline resources...")
        self.running = False
        self._shutdown_event.set()

        # Cancel all tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)

        # Cleanup data source sessions
        for source in self.data_sources:
            if hasattr(source, 'session') and source.session and not source.session.closed:
                await source.session.close()

    async def run(self):
        """Run the data pipeline"""
        self.running = True
        try:
            # Create tasks for each data source
            self.tasks = [
                asyncio.create_task(self._collect_and_store(source))
                for source in self.data_sources
            ]

            # Wait for shutdown event
            await self._shutdown_event.wait()

        except Exception as e:
            logger.error(f"Error in pipeline execution: {e}")
        finally:
            await self.cleanup()

    async def _collect_and_store(self, source):
        """Collect and store data from a source"""
        while self.running and not self._shutdown_event.is_set():
            try:
                # Fetch data
                raw_data = await source.fetch_data()
                # Process data
                df = await source.process_data(raw_data)

                # Store data if not empty
                if not df.empty:
                    await source.store_data(df)

                    # Add current timestamp to the tracking list
                    current_time = datetime.now()
                    self.data_timestamps[source.name].append(current_time)

                    # Remove timestamps older than 10 minutes (double our display window)
                    cutoff_time = current_time - timedelta(minutes=10)
                    self.data_timestamps[source.name] = [
                        ts for ts in self.data_timestamps[source.name]
                        if ts >= cutoff_time
                    ]

                    # Update cache with the most recent data
                    self.cache[source.name] = {
                        'timestamp': current_time.isoformat(),
                        'data': df.to_dict('records')
                    }

                # Wait for next collection interval or shutdown
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=source.collection_frequency
                    )
                except asyncio.TimeoutError:
                    continue

            except Exception as e:
                logger.error(f"Error collecting data from {source.name}: {e}")
                if not self._shutdown_event.is_set():
                    await asyncio.sleep(60)  # Wait before retrying

    def get_recent_data(self, source_name: str, minutes: int = 5) -> pd.DataFrame:
        """
        Get data collected within the last specified minutes

        Args:
            source_name: Name of the data source
            minutes: Number of minutes to look back

        Returns:
            DataFrame with filtered recent data
        """
        if source_name not in self.cache:
            logger.warning(f"No cache data found for source: {source_name}")
            return pd.DataFrame()

        # Get the cache data
        data_dict = self.cache.get(source_name, {})
        data_records = data_dict.get('data', [])

        if not data_records:
            logger.warning(f"Empty data records for source: {source_name}")
            return pd.DataFrame()

        # Log the number of records before filtering
        logger.info(f"Retrieved {len(data_records)} records from cache for {source_name}")

        # Convert to DataFrame
        df = pd.DataFrame(data_records)

        # Filter by timestamp if available
        if 'timestamp' in df.columns and len(df) > 0:
            try:
                # Import timezone modules
                import pytz
                from datetime import timezone

                # Convert timestamps to datetime if they're strings
                if isinstance(df['timestamp'].iloc[0], str):
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    logger.info("Converted string timestamps to datetime")

                # Create a timezone-aware cutoff time in UTC to match the data timestamps
                # This handles the case where data timestamps are in UTC (+00:00)
                cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=minutes)
                logger.info(f"Using UTC timezone-aware cutoff time: {cutoff_time}")

                # Debug info before filtering
                min_time = df['timestamp'].min()
                max_time = df['timestamp'].max()
                logger.info(f"Timestamp range: {min_time} to {max_time}")

                # Check if we have any recent data
                recent_df = df[df['timestamp'] >= cutoff_time].copy()

                # If we don't have any recent data, check if all data is old (more than 10 minutes)
                if recent_df.empty:
                    data_age = datetime.now(timezone.utc) - max_time.to_pydatetime()
                    if data_age.total_seconds() > 600:  # 10 minutes in seconds
                        logger.warning(
                            f"All data is old ({data_age.total_seconds() / 60:.1f} minutes) - might need to check collector")
                        # Return the most recent 5 minutes of data even if it's older than our cutoff
                        # Sort by timestamp descending
                        df_sorted = df.sort_values('timestamp', ascending=False)
                        # Take the most recent timestamps that span up to 5 minutes
                        most_recent_time = df_sorted['timestamp'].max()
                        oldest_recent_time = most_recent_time - timedelta(minutes=minutes)
                        recent_df = df_sorted[df_sorted['timestamp'] >= oldest_recent_time]
                        logger.info(
                            f"Returning {len(recent_df)} records from the most recent 5-minute period of available data")
                        return recent_df

                # Return the filtered dataframe
                logger.info(f"Filtered to {len(recent_df)} recent records")
                return recent_df

            except Exception as e:
                logger.error(f"Error filtering data by timestamp: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return df

        logger.info("No timestamp column or empty DataFrame - returning all data")
        return df

    def should_refresh(self, refresh_interval_seconds: int = 60) -> bool:
        """
        Determines if the dashboard should refresh based on the time elapsed
        since the last refresh.
        """
        current_time = datetime.now().timestamp()

        if self.last_refresh_time is None or (current_time - self.last_refresh_time) >= refresh_interval_seconds:
            self.last_refresh_time = current_time
            return True

        return False


class DashboardDataManager:
    """
    Manages data retrieval and filtering for dashboard visualization,
    keeping only recent data within a specified time window.
    """

    def __init__(self, base_path: str = "data", time_window_minutes: int = 5):
        self.base_path = base_path
        self.time_window_minutes = time_window_minutes
        self.last_refresh_time = None

    def get_recent_data(self, source_type: str) -> pd.DataFrame:
        """
        Retrieves and filters data from the last X minutes for a specific source type.

        Args:
            source_type: Either "tomtom_traffic" or "purpleair"

        Returns:
            DataFrame containing only recent data points
        """
        # Determine which directory to search based on source type
        if source_type == "tomtom_traffic":
            data_dir = os.path.join(self.base_path, "traffic")
        elif source_type == "purpleair":
            data_dir = os.path.join(self.base_path, "environmental")
        else:
            raise ValueError(f"Unknown source type: {source_type}")

        # Get current time and calculate the cutoff time
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(minutes=self.time_window_minutes)

        # Find the most recent data files
        today = current_time.strftime("%Y/%m")
        yesterday = (current_time - timedelta(days=1)).strftime("%Y/%m")

        search_paths = [
            os.path.join(data_dir, today, f"{source_type}_{current_time.strftime('%Y-%m-%d')}.json"),
            os.path.join(data_dir, yesterday,
                         f"{source_type}_{(current_time - timedelta(days=1)).strftime('%Y-%m-%d')}.json")
        ]

        # Collect all data entries from the recent files
        all_data = []
        for path in search_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        file_data = json.load(f)
                        all_data.extend(file_data.get('data', []))
                except json.JSONDecodeError:
                    logger.warning(f"Error reading JSON file: {path}")
                    continue

        # Convert to DataFrame
        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)

        # Filter to include only data within the time window
        if 'timestamp' in df.columns and len(df) > 0:
            # Convert timestamp string to datetime objects
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Make cutoff_time timezone-aware if the timestamps are timezone-aware
            if hasattr(df['timestamp'].iloc[0], 'tzinfo') and df['timestamp'].iloc[0].tzinfo is not None:
                cutoff_time = cutoff_time.replace(tzinfo=pytz.UTC)

            # Filter for only recent data
            recent_df = df[df['timestamp'] >= cutoff_time]

            return recent_df

        return df

    def should_refresh(self, refresh_interval_seconds: int = 60) -> bool:
        """
        Determines if the dashboard should refresh based on the time elapsed
        since the last refresh.
        """
        current_time = datetime.now().timestamp()

        if self.last_refresh_time is None or (current_time - self.last_refresh_time) >= refresh_interval_seconds:
            self.last_refresh_time = current_time
            return True

        return False


class DashboardApp:
    def __init__(self):
        self.storage_config = JSONStorageConfig(
            base_path="data",
            max_file_size_mb=10,
            create_backup=True
        )
        self.storage_handler = JSONStorageHandler(self.storage_config)
        self.pipeline = IntegratedDataPipeline(self.storage_handler)
        #self._initialize_data_sources()

        # initialize memory_cache
        self.pipeline.memory_cache = {}
        self._initialize_data_sources()

        # Add data manager for time-filtered data retrieval
        self.data_manager = DashboardDataManager(base_path="data", time_window_minutes=5)

    def _initialize_data_sources(self):
        """Initialize all data sources with API keys from secrets"""
        try:
            # Add TomTom traffic source
            tomtom_source = TomTomTrafficSource(
                api_key=st.secrets["TOMTOM_API_KEY"],
                storage_handler=self.storage_handler
            )
            self.pipeline.add_data_source(tomtom_source)

            # Add PurpleAir source
            purpleair_source = PurpleAirSource(
                api_key=st.secrets["PURPLEAIR_API_KEY"],
                storage_handler=self.storage_handler
            )
            self.pipeline.add_data_source(purpleair_source)

        except Exception as e:
            logger.error(f"Error initializing data sources: {e}")
            raise

    def get_recent_data(self, source_name: str, minutes: int = 5) -> pd.DataFrame:
        """
        Get data from the last X minutes for a specific source

        Args:
            source_name: Name of the data source
            minutes: Number of minutes to look back

        Returns:
            DataFrame with filtered recent data
        """
        # Try getting from pipeline cache first (in-memory data)
        if hasattr(self.pipeline, 'get_recent_data'):
            df = self.pipeline.get_recent_data(source_name, minutes)
            if not df.empty:
                return df

        # Fall back to data manager (file-based data)
        return self.data_manager.get_recent_data(source_name)

    async def run_once(self):
        """Run one cycle of data collection for all sources"""
        for source in self.pipeline.data_sources:
            try:
                # Fetch data
                raw_data = await source.fetch_data()

                # Process data
                df = await source.process_data(raw_data)

                # Store data if not empty
                if not df.empty:
                    await source.store_data(df)

                    # Update regular cache
                    self.pipeline.cache[source.name] = {
                        'timestamp': datetime.now().isoformat(),
                        'data': df.to_dict('records')
                    }

                    # Also update memory_cache if it exists
                    if hasattr(self.pipeline, 'memory_cache'):
                        self.pipeline.memory_cache[source.name] = df
                        logger.info(f"Successfully updated memory cache for {source.name}")

                    logger.info(f"Successfully collected and cached data from {source.name}")
                else:
                    logger.warning(f"No data collected from {source.name}")

            except Exception as e:
                logger.error(f"Error collecting data from {source.name}: {e}")

    def display_dashboard(self):
        """
        Display the dashboard visualization with recent data
        """
        # Create tabs for different visualizations
        traffic_tab, air_quality_tab, combined_tab = st.tabs(["Traffic", "Air Quality", "Combined View"])

        # Get the data
        traffic_data = self.get_recent_data("tomtom_traffic")
        air_data = self.get_recent_data("purpleair")

        with traffic_tab:
            st.subheader("Traffic Conditions")
            if not traffic_data.empty:
                # Create metrics at the top
                col1, col2, col3 = st.columns(3)
                with col1:
                    avg_speed = traffic_data['speed'].mean()
                    st.metric("Average Speed", f"{avg_speed:.1f} mph")
                with col2:
                    # Calculate congestion ratio
                    traffic_data['congestion_ratio'] = traffic_data['free_flow_speed'] / traffic_data['speed']
                    avg_congestion = (traffic_data['congestion_ratio'] - 1).clip(0, 1).mean() * 100
                    st.metric("Average Congestion", f"{avg_congestion:.1f}%")
                with col3:
                    st.metric("Traffic Segments", f"{len(traffic_data)}")

                # Prepare data for the map
                map_data = traffic_data[['lat_start', 'lon_start']].copy()
                map_data.rename(columns={'lat_start': 'lat', 'lon_start': 'lon'}, inplace=True)

                # Display the map
                st.map(map_data)

                # Show road segment types distribution
                if 'id' in traffic_data.columns:
                    road_types = traffic_data['id'].value_counts()
                    st.write("### Road Types Distribution")
                    st.bar_chart(road_types)
            else:
                st.warning("No recent traffic data available")

        with air_quality_tab:
            st.subheader("Air Quality Monitoring")
            if not air_data.empty:
                # Create metrics at the top
                col1, col2, col3 = st.columns(3)
                with col1:
                    avg_pm25 = air_data['pm2_5'].mean()
                    st.metric("Average PM2.5", f"{avg_pm25:.1f} µg/m³")
                with col2:
                    max_pm25 = air_data['pm2_5'].max()
                    st.metric("Max PM2.5", f"{max_pm25:.1f} µg/m³")
                with col3:
                    st.metric("Active Sensors", f"{len(air_data)}")

                # Create air quality map
                st.map(air_data[['lat', 'lon']])

                # Show PM2.5 distribution
                st.write("### PM2.5 Distribution")
                # Create 5 bins for PM2.5 values
                air_data['pm2_5_category'] = pd.cut(
                    air_data['pm2_5'],
                    bins=[0, 12, 35.4, 55.4, 150.4, 500],
                    labels=['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 'Unhealthy', 'Very Unhealthy']
                )
                pm25_dist = air_data['pm2_5_category'].value_counts()
                st.bar_chart(pm25_dist)
            else:
                st.warning("No recent air quality data available")

        with combined_tab:
            st.subheader("Combined Traffic & Air Quality Analysis")
            if not traffic_data.empty and not air_data.empty:
                # Create a combined visualization
                st.write("### Geographic Coverage")

                # Create a DataFrame for the combined map
                combined_map_data = pd.DataFrame()

                # Add traffic data points (start locations)
                traffic_points = traffic_data[['lat_start', 'lon_start']].copy()
                traffic_points.rename(columns={'lat_start': 'lat', 'lon_start': 'lon'}, inplace=True)
                traffic_points['type'] = 'Traffic'

                # Add air quality data points
                air_points = air_data[['lat', 'lon']].copy()
                air_points['type'] = 'Air Quality'

                # Combine them
                combined_map_data = pd.concat([traffic_points, air_points])

                # Show the combined map
                st.map(combined_map_data)

                # Show numeric insights
                st.write("### Data Summary")
                st.write(f"Total Data Points: {len(traffic_data) + len(air_data)}")
                st.write(f"Traffic Segments: {len(traffic_data)}")
                st.write(f"Air Quality Sensors: {len(air_data)}")
            else:
                st.warning("Need both traffic and air quality data for combined view")

    async def run(self):
        """Run the dashboard application"""
        await self.pipeline.run()


async def run_dashboard(dashboard):
    """Run the dashboard with proper asyncio handling"""
    await dashboard.run()


def main():
    st.title("Traffic and Environmental Dashboard")

    # Initialize session state
    if 'dashboard' not in st.session_state:
        st.session_state.dashboard = DashboardApp()

    # Add a stop button
    if st.button('Stop Data Collection'):
        if hasattr(st.session_state.dashboard.pipeline, '_shutdown_event'):
            asyncio.run(st.session_state.dashboard.pipeline.cleanup())
            st.experimental_rerun()

    # Show status
    st.write("Data collection is running...")

    # Get the data counts
    traffic_data = st.session_state.dashboard.get_recent_data("tomtom_traffic")
    air_data = st.session_state.dashboard.get_recent_data("purpleair")

    traffic_count = len(traffic_data) if not traffic_data.empty else 0
    air_count = len(air_data) if not air_data.empty else 0

    # Create progress bars for data collection
    st.write("### Data Collection Progress")
    traffic_progress = min(100, int((traffic_count / 100) * 100))
    air_progress = min(100, int((air_count / 100) * 100))

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Traffic Segments: {traffic_count}/100")
        st.progress(traffic_progress / 100)
    with col2:
        st.write(f"Air Quality Sensors: {air_count}/100")
        st.progress(air_progress / 100)

    # Show maps automatically if enough data points are available
    if traffic_count >= 100 and air_count >= 100:
        st.session_state.dashboard.display_dashboard()
    else:
        # Show a waiting message
        st.info("Maps will appear automatically once sufficient data is collected")
        if traffic_count < 100:
            st.write(f"Waiting for more traffic data...")
        if air_count < 100:
            st.write(f"Waiting for more air quality data...")

    # Run the dashboard
    try:
        asyncio.run(run_dashboard(st.session_state.dashboard))
    except Exception as e:
        logger.error(f"Error running dashboard: {e}")
        st.error(f"An error occurred: {str(e)}")
