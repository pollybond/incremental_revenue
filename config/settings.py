import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

DB_CONFIG = {
    "host": os.getenv("GP_HOST"),
    "port": int(os.getenv("GP_PORT", 5432)),
    "dbname": os.getenv("GP_DB"),
    "user": os.getenv("GP_USER"),
    "password": os.getenv("GP_PASSWORD"),
}

if not all(DB_CONFIG.values()):
    missing = [k for k, v in DB_CONFIG.items() if not v]
    raise EnvironmentError(f"Missing DB config vars in .env: {missing}")

RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))
PRE_PERIOD_DAYS = int(os.getenv("PRE_PERIOD_DAYS", 30))
POST_PERIOD_DAYS = int(os.getenv("POST_PERIOD_DAYS", 30))