"""Update yt-dlp to the latest version."""

import subprocess
import sys

def update_yt_dlp():
    """Update yt-dlp to the latest version."""
    print("Updating yt-dlp to the latest version...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "yt-dlp"])
        print("yt-dlp updated successfully!")
    except Exception as e:
        print(f"Error updating yt-dlp: {str(e)}")

if __name__ == "__main__":
    update_yt_dlp()