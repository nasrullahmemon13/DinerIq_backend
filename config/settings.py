import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

                      
APP_NAME = "DineIQ Analytics"
APP_VERSION = "1.0.0"
DEBUG = os.getenv("DINEIQ_DEBUG", "False").lower() in ("true", "1", "yes")

              
DEFAULT_API_PORT = int(os.getenv("DINEIQ_PORT", "8010"))
PORT_RANGE = (8010, 8030)

                
SECRET_KEY = os.getenv("SECRET_KEY", "dineiq-super-secret-key-production-2026")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24            

                     
DATA_DIR = BASE_DIR / "raw_data"
PROCESSED_DIR = BASE_DIR / "processed_data"
PARQUET_DIR = BASE_DIR / "parquet_data"
STAGING_DIR = BASE_DIR / "data_staging"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

                        
DATABASE_PATH = BASE_DIR / "database" / "dineiq.db"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATABASE_PATH.as_posix()}")
