# YouTube Processor

A comprehensive tool for processing YouTube videos and playlists, generating summaries, and analyzing content.

## Features

- Download and process YouTube videos and playlists
- Generate summaries using state-of-the-art NLP models
- Support for both Vietnamese and English content
- Transcribe audio using Whisper model
- Calculate quality metrics (BERTScore, ROUGE)
- Store results in MongoDB

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/youtube_processor.git
cd youtube_processor

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Usage

### Command Line

```bash
# Process a YouTube playlist
youtube-processor "https://www.youtube.com/playlist?list=PLAYLIST_ID" --max-videos 10
```

### As a Python Module

```python
from src.main import process_youtube_playlist

# Process a playlist
results = process_youtube_playlist("https://www.youtube.com/playlist?list=PLAYLIST_ID", max_videos=10)
```

## Configuration

Create a `.env` file in the root directory with the following variables:

```
YOUTUBE_API_KEY=your_youtube_api_key
HF_TOKEN=your_huggingface_token
MONGO_URI=your_mongodb_connection_string
MONGO_DB=youtube
MONGO_COLLECTION=summaries
```

## Project Structure

```
youtube_processor/
│
├── src/                           # Source code
│   ├── main.py                    # Entry point
│   ├── config.py                  # Configuration
│   ├── models/                    # ML models
│   ├── processors/                # Processing logic
│   ├── utils/                     # Utility functions
│   └── storage/                   # Data storage
│
├── notebooks/                     # Jupyter notebooks
├── tests/                         # Unit tests
├── requirements.txt               # Dependencies
└── README.md                      # Documentation
```

## License

MIT