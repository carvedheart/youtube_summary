"""Run YouTube processor using official YouTube API with OAuth 2.0."""

import os
import argparse
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from src.config import MAX_VIDEOS_PER_PLAYLIST
from src.processors.video_processor_api import process_video_api, get_youtube_service
from src.storage.mongodb_handler_pandas import upload_to_mongodb_pandas
from src.utils.youtube_utils import get_video_id

def get_playlist_videos(playlist_id, max_videos=MAX_VIDEOS_PER_PLAYLIST):
    """Get videos from a playlist using YouTube API."""
    youtube = get_youtube_service()
    
    # Get playlist details
    playlist_response = youtube.playlists().list(
        part="snippet",
        id=playlist_id
    ).execute()
    
    if not playlist_response["items"]:
        print(f"Playlist {playlist_id} not found")
        return None, []
    
    playlist_title = playlist_response["items"][0]["snippet"]["title"]
    
    # Get playlist items
    video_ids = []
    next_page_token = None
    
    while len(video_ids) < max_videos:
        playlist_items_response = youtube.playlistItems().list(
            part="snippet",
            playlistId=playlist_id,
            maxResults=50,
            pageToken=next_page_token
        ).execute()
        
        for item in playlist_items_response["items"]:
            video_ids.append(item["snippet"]["resourceId"]["videoId"])
            if len(video_ids) >= max_videos:
                break
        
        next_page_token = playlist_items_response.get("nextPageToken")
        if not next_page_token:
            break
    
    return playlist_title, video_ids[:max_videos]

def extract_playlist_id(url):
    """Extract playlist ID from URL."""
    import re
    
    # Try to match playlist ID
    playlist_id_match = re.search(r"list=([a-zA-Z0-9_-]+)", url)
    if playlist_id_match:
        return playlist_id_match.group(1)
    
    return None

def process_youtube_playlist_api(url, max_videos=MAX_VIDEOS_PER_PLAYLIST):
    """Process a YouTube playlist using YouTube API."""
    # Create directories
    os.makedirs("output", exist_ok=True)
    
    try:
        # Extract playlist ID
        playlist_id = extract_playlist_id(url)
        if not playlist_id:
            print("[ERROR] Invalid playlist URL")
            return None
        
        # Get playlist videos
        playlist_title, video_ids = get_playlist_videos(playlist_id, max_videos)
        if not video_ids:
            print("[ERROR] No videos found in playlist")
            return None
        
        print(f"Playlist: {playlist_title}")
        print(f"Found {len(video_ids)} videos")
        
        # Process videos
        results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(process_video_api, video_id): video_id for video_id in video_ids}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing videos"):
                video_id = futures[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        print(f"Processed: {result['Title']}")
                except Exception as e:
                    print(f"Error processing video {video_id}: {str(e)}")
        
        if not results:
            print("[WARN] No video processed successfully")
            return None
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save results
        output_path = "output/youtube_summary_results_api.csv"
        df.to_csv(output_path, index=False)
        print(f"\nProcessed {len(df)} videos successfully")
        print(f"Results saved to {output_path}")
        
        # Upload to MongoDB
        try:
            upload_to_mongodb_pandas(df)
        except Exception as e:
            print(f"Error uploading to MongoDB: {str(e)}")
        
        return df
    
    except Exception as e:
        print(f"Error processing playlist: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main entry point for the command line."""
    parser = argparse.ArgumentParser(description="Process YouTube playlists using API")
    parser.add_argument("url", help="YouTube playlist URL")
    parser.add_argument("--max-videos", type=int, default=MAX_VIDEOS_PER_PLAYLIST, 
                        help="Maximum number of videos to process")
    
    args = parser.parse_args()
    process_youtube_playlist_api(args.url, args.max_videos)

if __name__ == "__main__":
    main()
