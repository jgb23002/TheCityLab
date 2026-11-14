"""Load local environment settings and resolve project paths."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

COSTAR_DATA_DIR = PROJECT_ROOT / "data" / "costar" / "data"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
USE_CACHE_ONLY = os.getenv("USE_CACHE_ONLY", "false").strip().lower() == "true"

VEXCEL_BASE_URL = "https://api.vexcelgroup.com/v2"
VEXCEL_API_KEY = os.getenv("VEXCEL_API_KEY")
MATTERPORT_GRAPHQL_URL = "https://api.matterport.com/api/models/graph"
MP_TOKEN_ID = os.getenv("MP_TOKEN_ID")
MP_TOKEN_SECRET = os.getenv("MP_TOKEN_SECRET")
MP_SDK_KEY = os.getenv("MP_SDK_KEY")
AWS_REGION = os.getenv("AWS_REGION")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").strip().lower() == "true"
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")
NREL_API_KEY = os.getenv("NREL_API_KEY")
LLM_PROVIDER = os.getenv("LLM_PROVIDER")
LLM_MODEL = os.getenv("LLM_MODEL")
VISION_MODEL = os.getenv("VISION_MODEL")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# TODO: Validate required settings when their integrations are implemented.
