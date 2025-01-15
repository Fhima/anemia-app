import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')
    ROBOFLOW_API_URL = os.getenv('ROBOFLOW_API_URL')
    
    @staticmethod
    def validate_config():
        """Validate that all required configuration variables are set"""
        if not all([Config.ROBOFLOW_API_KEY, Config.ROBOFLOW_API_URL]):
            raise ValueError("Missing required environment variables. Please check your .env file.")
