import aiohttp
import asyncio
from datetime import datetime, timezone
import pandas as pd
from typing import Dict
import json
from base_classes import DataSource
import numpy as np


class TomTomTrafficSource(DataSource):
    """Handler for TomTom traffic data with comprehensive SF coverage"""

    def __init__(self, api_key: str, storage_handler):
        super().__init__("tomtom_traffic", api_key, storage_handler)
        self.collection_frequency = 60  # 1 minute

        # San Francisco bounds
        self.sf_bounds = {
            'min_lat': 37.708,  # Southern boundary
            'max_lat': 37.832,  # Northern boundary
            'min_lon': -122.515,  # Western boundary
            'max_lon': -122.355  # Eastern boundary
        }

        # Generate grid of points
        self.sample_points = self._generate_sample_points()
        self.logger.info(f"Generated {len(self.sample_points)} sample points across San Francisco")

    def _generate_sample_points(self) -> list:
        """Generate 500 well-distributed points across San Francisco"""
        # Calculate grid dimensions
        # We'll use a sqrt(500) x sqrt(500) grid and filter out points in the water
        grid_size = int(np.ceil(np.sqrt(500 * 1.5)))  # 1.5x oversampling to account for water

        # Generate lat/lon grid
        lats = np.linspace(self.sf_bounds['min_lat'], self.sf_bounds['max_lat'], grid_size)
        lons = np.linspace(self.sf_bounds['min_lon'], self.sf_bounds['max_lon'], grid_size)

        # Create meshgrid
        lat_grid, lon_grid = np.meshgrid(lats, lons)

        # Convert to list of points
        points = []
        for lat, lon in zip(lat_grid.flatten(), lon_grid.flatten()):
            # Skip points that are likely in the water
            if self._is_likely_land(lat, lon):
                points.append(f"{lat:.6f},{lon:.6f}")

            # Break if we have enough points
            if len(points) >= 500:
                break

        # If we don't have enough points, add some key neighborhood centers
        if len(points) < 500:
            additional_points = self._get_neighborhood_points()
            points.extend(additional_points)

        # Ensure we have exactly 500 points
        return points[:500]

    def _is_likely_land(self, lat: float, lon: float) -> bool:
        """
        Basic check if a point is likely on land in San Francisco.
        This is a simplified check that excludes obvious water areas.
        """
        # Define simplified SF peninsula shape
        if lon < -122.515:  # Too far west (Pacific Ocean)
            return False
        if lon > -122.355:  # Too far east (Bay)
            return False
        if lat < 37.708:  # Too far south
            return False
        if lat > 37.832:  # Too far north
            return False

        # Exclude obvious water areas
        # Golden Gate
        if lat > 37.81 and lon < -122.48:
            return False
        # SF Bay
        if lat > 37.75 and lon > -122.37:
            return False
        # Lower Bay
        if lat < 37.75 and lon > -122.38:
            return False

        return True

    def _get_neighborhood_points(self) -> list:
        """Return additional points for key SF neighborhoods"""
        neighborhoods = [
            # Downtown/Financial District
            (37.7749, -122.4194),
            # Mission District
            (37.7599, -122.4148),
            # North Beach
            (37.8060, -122.4103),
            # Marina District
            (37.8029, -122.4359),
            # Haight-Ashbury
            (37.7692, -122.4481),
            # Castro District
            (37.7609, -122.4350),
            # SoMa
            (37.7785, -122.3975),
            # Nob Hill
            (37.7929, -122.4149),
            # Richmond District
            (37.7789, -122.4750),
            # Sunset District
            (37.7549, -122.4936),
            # Potrero Hill
            (37.7583, -122.4000),
            # Pacific Heights
            (37.7925, -122.4382),
            # Dogpatch
            (37.7648, -122.3883),
            # Hayes Valley
            (37.7759, -122.4245),
            # Lower Haight
            (37.7717, -122.4298)
        ]
        return [f"{lat:.6f},{lon:.6f}" for lat, lon in neighborhoods]

    async def fetch_data(self) -> Dict:
        """Fetch traffic data from TomTom API"""
        if not hasattr(self, 'session') or self.session is None or self.session.closed:
            connector = aiohttp.TCPConnector(ssl=self.ssl_context)
            self.session = aiohttp.ClientSession(connector=connector)

        all_segments = []

        try:
            base_url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
            headers = {
                'Accept': 'application/json'
            }

            # Process points in batches to avoid overloading
            batch_size = 10
            for i in range(0, len(self.sample_points), batch_size):
                batch = self.sample_points[i:i + batch_size]

                # Create tasks for concurrent requests
                tasks = []
                for point in batch:
                    params = {
                        'key': self.api_key,
                        'point': point,
                        'unit': 'MPH',
                        'thickness': 2,
                        'zoom': 14,
                        'format': 'json'
                    }
                    tasks.append(self.session.get(base_url, headers=headers, params=params, timeout=30))

                # Execute batch of requests concurrently
                responses = await asyncio.gather(*tasks, return_exceptions=True)

                # Process responses
                for response in responses:
                    if isinstance(response, Exception):
                        self.logger.error(f"Error in batch request: {response}")
                        continue

                    try:
                        data = await response.json()
                        if 'flowSegmentData' in data:
                            all_segments.append(data)
                    except Exception as e:
                        self.logger.error(f"Error processing response: {e}")
                        continue

                # Small delay between batches to be nice to the API
                await asyncio.sleep(0.1)

            self.logger.info(f"Successfully fetched {len(all_segments)} segments from TomTom API")
            return {'segments': all_segments}

        except Exception as e:
            self.logger.error(f"Error fetching TomTom traffic data: {e}")
            return {'segments': []}

    async def process_data(self, raw_data: Dict) -> pd.DataFrame:
        """Process TomTom traffic data"""
        try:
            processed_segments = []

            for segment_data in raw_data.get('segments', []):
                flow = segment_data.get('flowSegmentData', {})
                if not flow:
                    continue

                coordinates = flow.get('coordinates', {}).get('coordinate', [])
                if not coordinates or len(coordinates) < 2:
                    continue

                # Get first and last points
                first_coord = coordinates[0]
                last_coord = coordinates[-1]

                # Create segment data
                segment = {
                    'id': flow.get('frc'),
                    'road_name': f"FRC: {flow.get('frc')}",
                    'speed': flow.get('currentSpeed'),
                    'free_flow_speed': flow.get('freeFlowSpeed'),
                    'confidence': flow.get('confidence'),
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'source': self.name,
                    'lat_start': first_coord.get('latitude'),
                    'lon_start': first_coord.get('longitude'),
                    'lat_end': last_coord.get('latitude'),
                    'lon_end': last_coord.get('longitude'),
                    'road_closure': flow.get('roadClosure', False),
                    'current_travel_time': flow.get('currentTravelTime'),
                    'free_flow_travel_time': flow.get('freeFlowTravelTime')
                }

                # Validate segment data
                if all(v is not None for v in segment.values()):
                    processed_segments.append(segment)

            # Create DataFrame with all segments
            df = pd.DataFrame(processed_segments)

            # Remove duplicate segments
            df = df.drop_duplicates(subset=['lat_start', 'lon_start', 'lat_end', 'lon_end'])

            self.logger.info(f"Processed {len(df)} unique traffic segments")
            return df

        except Exception as e:
            self.logger.error(f"Error processing TomTom data: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()






class PurpleAirSource(DataSource):
    """Handler for PurpleAir environmental data"""

    def __init__(self, api_key: str, storage_handler):
        super().__init__("purpleair", api_key, storage_handler)
        self.collection_frequency = 900  # 15 minutes

    async def fetch_data(self) -> Dict:
        """Fetch data from PurpleAir API"""
        if not hasattr(self, 'session') or self.session is None or self.session.closed:
            connector = aiohttp.TCPConnector(ssl=self.ssl_context)
            self.session = aiohttp.ClientSession(connector=connector)

        try:
            url = "https://api.purpleair.com/v1/sensors"
            headers = {
                "X-API-Key": self.api_key
            }

            # Request fields in specific order
            params = {
                'fields': 'latitude,longitude,pm2.5_atm,temperature,humidity,pressure',
                'location_type': 0,  # outdoor sensors
                'nwlng': -122.5,  # San Francisco bounding box
                'nwlat': 37.8,
                'selng': -122.35,
                'selat': 37.7
            }

            self.logger.warning(f"Requesting PurpleAir data with fields: {params['fields']}")

            async with self.session.get(url, headers=headers, params=params, timeout=30) as response:
                response.raise_for_status()
                data = await response.json()

                # Debug logging
                if 'fields' in data and 'data' in data and data['data']:
                    self.logger.warning(f"API Response fields: {data['fields']}")
                    self.logger.warning(f"First sensor raw data: {data['data'][0]}")
                return data

        except Exception as e:
            self.logger.error(f"Error fetching PurpleAir data: {e}")
            return {}

    async def process_data(self, raw_data: Dict) -> pd.DataFrame:
        """Process PurpleAir data"""
        if not raw_data or 'data' not in raw_data or 'fields' not in raw_data:
            self.logger.error("Missing required data in PurpleAir response")
            return pd.DataFrame()

        try:
            # Get and verify field mappings
            fields = raw_data['fields']
            self.logger.warning(f"Field order in response: {fields}")

            # Create direct index mapping
            field_map = {
                'latitude': fields.index('latitude'),
                'longitude': fields.index('longitude'),
                'pm2.5': fields.index('pm2.5_atm'),
                'temperature': fields.index('temperature'),
                'humidity': fields.index('humidity'),
                'pressure': fields.index('pressure')
            }

            self.logger.warning(f"Field mapping: {field_map}")

            # Define PM2.5 filtering parameters
            max_pm25_threshold = 500  # Maximum reasonable PM2.5 value

            # List of sensor IDs to exclude (known malfunctioning sensors)
            excluded_sensor_ids = ['4746252160', '4746262784']
            suspected_bad_sensors = []  # Will track sensors with consistently extreme values

            self.logger.info(f"Filtering out {len(excluded_sensor_ids)} known malfunctioning sensors")

            # First pass - collect all raw readings to identify statistical outliers
            all_pm25_values = []
            all_sensors = []

            for sensor in raw_data['data']:
                try:
                    if sensor[field_map['pm2.5']] is not None:
                        pm25 = float(sensor[field_map['pm2.5']])
                        all_pm25_values.append(pm25)
                        all_sensors.append(sensor)
                except (ValueError, TypeError):
                    continue

            # Calculate statistical thresholds for outlier detection
            if all_pm25_values:
                mean_pm25 = sum(all_pm25_values) / len(all_pm25_values)
                median_pm25 = sorted(all_pm25_values)[len(all_pm25_values) // 2]

                # Calculate IQR for outlier detection
                q1 = np.percentile(all_pm25_values, 25)
                q3 = np.percentile(all_pm25_values, 75)
                iqr = q3 - q1
                upper_bound = q3 + 3 * iqr  # 3*IQR above Q3 is considered extreme

                self.logger.info(f"PM2.5 statistics - Mean: {mean_pm25:.2f}, Median: {median_pm25:.2f}")
                self.logger.info(f"PM2.5 outlier threshold: {upper_bound:.2f} (Q3 + 3*IQR)")

                # Use upper_bound but cap it at max_pm25_threshold
                outlier_threshold = min(upper_bound, max_pm25_threshold)
            else:
                outlier_threshold = max_pm25_threshold

            self.logger.info(f"Using PM2.5 outlier threshold: {outlier_threshold:.2f}")

            # Second pass - process sensors with filtering
            processed_data = []
            filtered_count = 0
            outlier_count = 0

            for sensor in all_sensors:
                self.logger.debug(f"Processing raw sensor data: {sensor}")
                try:
                    # Generate sensor ID
                    sensor_id = str(id(sensor))

                    # Skip excluded sensors (known malfunctioning ones)
                    if sensor_id in excluded_sensor_ids or sensor_id in suspected_bad_sensors:
                        filtered_count += 1
                        self.logger.info(f"Filtered out sensor ID: {sensor_id}")
                        continue

                    # Get PM2.5 value and check if it's an outlier
                    pm25 = float(sensor[field_map['pm2.5']])
                    if pm25 > outlier_threshold:
                        outlier_count += 1
                        self.logger.warning(f"Filtering outlier PM2.5 value: {pm25} from sensor ID: {sensor_id}")
                        # Instead of skipping entirely, we can cap the value at the threshold
                        pm25 = outlier_threshold

                    processed_sensor = {
                        'sensor_id': sensor_id,
                        'lat': float(sensor[field_map['latitude']]),
                        'lon': float(sensor[field_map['longitude']]),
                        'pm2_5': pm25,  # Use filtered PM2.5 value
                        'temperature': float(sensor[field_map['temperature']]),
                        'humidity': float(sensor[field_map['humidity']]),
                        'pressure': float(sensor[field_map['pressure']]),
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'source': self.name
                    }

                    self.logger.debug(f"Processed sensor: {json.dumps(processed_sensor, indent=2)}")

                    # Validate data for geographic and temperature validity
                    if (-90 <= processed_sensor['lat'] <= 90 and
                            -180 <= processed_sensor['lon'] <= 180 and
                            -100 <= processed_sensor['temperature'] <= 150):  # Fahrenheit range
                        processed_data.append(processed_sensor)
                    else:
                        self.logger.warning(f"Invalid data detected: lat={processed_sensor['lat']}, "
                                            f"lon={processed_sensor['lon']}, "
                                            f"pm2.5={processed_sensor['pm2_5']}, "
                                            f"temp={processed_sensor['temperature']}")

                except Exception as e:
                    self.logger.error(f"Error processing sensor: {e}")
                    continue

            df = pd.DataFrame(processed_data)

            # Log filtering results
            self.logger.info(f"Filtered out {filtered_count} sensors with known issues")
            self.logger.info(f"Capped {outlier_count} outlier PM2.5 readings")
            self.logger.info(f"Processed {len(df)} valid sensors after filtering")

            # Log final processed data sample
            if not df.empty:
                self.logger.warning(f"Final processed data sample:\n{df.head(1).to_dict('records')}")

            return df

        except Exception as e:
            self.logger.error(f"Error processing PurpleAir data: {e}")
            self.logger.error(f"Raw data structure: {json.dumps(raw_data.keys())}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()