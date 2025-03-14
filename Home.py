import streamlit as st
import asyncio
import time
from complete_dashboard import DashboardApp

st.set_page_config(page_title="SF Traffic & Air Quality Monitor", 
                   layout="wide", 
                   page_icon="🌉")

st.title("San Francisco Traffic & Air Quality Monitor")

# Clear cached dashboard app to ensure we get the new version with the updated methods
if 'dashboard_refresh_needed' not in st.session_state:
    # Clear the data collector to ensure we get the updated version
    if 'data_collector' in st.session_state:
        del st.session_state.data_collector
    
    # Set flag to avoid clearing it on every reload
    st.session_state.dashboard_refresh_needed = False

# Initialize the data collection dashboard in session state
if 'data_collector' not in st.session_state:
    st.session_state.data_collector = DashboardApp()

# Initialize last_collection to track when we last collected data
if 'last_collection' not in st.session_state:
    st.session_state.last_collection = None

# Add a control button
if 'collecting' not in st.session_state:
    st.session_state.collecting = False

if st.button('Start Data Collection' if not st.session_state.collecting else 'Stop Data Collection'):
    st.session_state.collecting = not st.session_state.collecting

# Status indicator
if st.session_state.collecting:
    st.success("✅ Data collection is running")

    # Only trigger data collection at intervals to avoid hammering APIs
    current_time = time.time()
    if st.session_state.last_collection is None or (current_time - st.session_state.last_collection) > 60:
        try:
            # Run one data collection cycle
            asyncio.run(st.session_state.data_collector.run_once())
            st.session_state.last_collection = current_time
        except Exception as e:
            st.error(f"Error in data collection: {str(e)}")
            st.session_state.collecting = False
else:
    st.warning("⏸️ Data collection is paused")

# Display cache status
st.subheader("Memory Cache Status")
if 'data_collector' in st.session_state and hasattr(st.session_state.data_collector, 'pipeline'):
    pipeline = st.session_state.data_collector.pipeline

    has_cache = hasattr(pipeline, 'cache')
    st.write(f"Has memory cache: {has_cache}")

    if has_cache:
        cache_keys = list(pipeline.cache.keys())
        st.write(f"Memory cache keys: {cache_keys}")

        # Check for traffic data
        traffic_in_cache = any('traffic' in key.lower() for key in cache_keys)
        st.write(f"Traffic data in memory cache: {traffic_in_cache}")

        # Check for air quality data
        air_quality_in_cache = any('purpleair' in key.lower() for key in cache_keys)
        st.write(f"Air quality data in memory cache: {air_quality_in_cache}")

        # Display the latest data samples if available
        for source_name, cache_data in pipeline.cache.items():
            with st.expander(f"Latest {source_name} data"):
                st.write(f"Last updated: {cache_data['timestamp']}")
                if cache_data['data']:
                    st.dataframe(cache_data['data'][:5])  # Show first 5 rows
                else:
                    st.write("No data available")

# Add navigation instructions
st.markdown("""
## Navigation
Use the sidebar to navigate between pages:
- **Home (Current Page)**: Control data collection
- **Dashboard**: View real-time visualizations with 5-minute data window
""")

# Add information about the data collection
st.markdown("""
## Data Sources
- **TomTom Traffic API**: Real-time traffic flow data
- **PurpleAir API**: Air quality measurements

Data is collected and stored in JSON format in the `data` directory.
""")

# Add a description of the dashboard optimization
st.markdown("""
## Visualization Features
- Maps show only data from the last 5 minutes
- Auto-refresh every minute to keep data current
- Color-coded indicators for traffic congestion and air quality levels
- Data timestamp tracking to ensure freshness
""")