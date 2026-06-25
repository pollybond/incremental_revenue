import os
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=dotenv_path)

from config.promo_types import PromoMechanicType, MECHANIC_DESCRIPTIONS

__all__ = [
    "PromoMechanicType",
    "MECHANIC_DESCRIPTIONS"
]