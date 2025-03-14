import logging
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict
import aiohttp
import ssl
import certifi


class DataSource(ABC):
    """Base class for all data sources with storage integration"""

    def __init__(self, name: str, api_key: str, storage_handler):
        self.name = name
        self.api_key = api_key
        self.storage_handler = storage_handler
        self.logger = logging.getLogger(self.__class__.__name__)
        self.session: aiohttp.ClientSession = None

        # Create SSL context using certifi
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())
        self.conn_kwargs = {
            "ssl": self.ssl_context,
            "verify_ssl": True
        }

    async def cleanup(self):
        """Cleanup resources"""
        if self.session and not self.session.closed:
            await self.session.close()

    async def store_data(self, data: pd.DataFrame):
        """Store the processed data"""
        try:
            await self.storage_handler.save_data(data, self.name)
        except Exception as e:
            self.logger.error(f"Error storing data for {self.name}: {e}")
            raise

    @abstractmethod
    async def fetch_data(self) -> Dict:
        """Fetch data from the source"""
        pass

    @abstractmethod
    async def process_data(self, raw_data: Dict) -> pd.DataFrame:
        """Process the raw data into a standardized format"""
        pass