import json
from datetime import datetime
import pandas as pd

class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle Timestamp objects"""
    def default(self, obj):
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        return super().default(obj)