import os
from dotenv import load_dotenv

load_dotenv()

MODEL: str = os.getenv("MODEL") or ""
