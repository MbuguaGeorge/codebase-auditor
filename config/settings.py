import os
from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("PROVIDER", "openai")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

MODEL = os.getenv("MODEL", "gpt-4o")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./traces.db")
