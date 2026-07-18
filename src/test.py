import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")

print("Đường dẫn .env:", ENV_FILE)
print("File .env tồn tại:", ENV_FILE.exists())
print("Đã đọc API key:", bool(GEMINI_API_KEY))
print("Model:", GEMINI_MODEL)