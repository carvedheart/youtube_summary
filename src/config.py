"""Configuration settings for the YouTube processor."""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys and tokens
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# MongoDB Configuration
MONGO_CONFIG = {
    "connection_string": os.getenv("MONGO_URI"),
    "database_name": os.getenv("MONGO_DB", "youtube"),
    "collection_name": os.getenv("MONGO_COLLECTION", "test")
}

# Whisper model configuration
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "tiny")  # Thay đổi từ base sang tiny
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float32")

# Processing configuration
MAX_VIDEOS_PER_PLAYLIST = int(os.getenv("MAX_VIDEOS_PER_PLAYLIST", "30"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))
SUMMARY_MAX_LENGTH = int(os.getenv("SUMMARY_MAX_LENGTH", "150"))
SUMMARY_MIN_LENGTH = int(os.getenv("SUMMARY_MIN_LENGTH", "30"))

