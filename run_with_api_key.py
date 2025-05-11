"""Run YouTube processor using only YouTube API Key without OAuth."""

import os
import argparse
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import xml.etree.ElementTree
import xml.parsers.expat

from src.config import YOUTUBE_API_KEY, MAX_VIDEOS_PER_PLAYLIST
from src.models.summarization_model import summarize_text
from src.utils.metrics import compute_bertscore, compute_rouge
from src.utils.youtube_utils import get_video_id
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

def get_youtube_service():
    """Initialize the YouTube API service with API key only."""
    return build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

def get_playlist_videos(playlist_id, max_videos=MAX_VIDEOS_PER_PLAYLIST):
    """Get videos from a playlist using YouTube API with API key."""
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
            if item["snippet"]["resourceId"]["kind"] == "youtube#video":
                video_ids.append(item["snippet"]["resourceId"]["videoId"])
                if len(video_ids) >= max_videos:
                    break
        
        next_page_token = playlist_items_response.get("nextPageToken")
        if not next_page_token:
            break
    
    return playlist_title, video_ids

def extract_playlist_id(url):
    """Extract playlist ID from URL."""
    import re
    
    # Try to match playlist ID
    playlist_id_match = re.search(r"list=([a-zA-Z0-9_-]+)", url)
    if playlist_id_match:
        return playlist_id_match.group(1)
    
    return None

def get_video_details(video_id):
    """Get video details using YouTube API with API key."""
    try:
        youtube = get_youtube_service()
        
        # Get video details
        video_response = youtube.videos().list(
            part="snippet,contentDetails,statistics",
            id=video_id
        ).execute()
        
        if not video_response["items"]:
            print(f"Video {video_id} not found")
            return None
        
        video = video_response["items"][0]
        
        # Extract relevant information
        return {
            "Title": video["snippet"]["title"],
            "URL": f"https://www.youtube.com/watch?v={video_id}",
            "Duration": video["contentDetails"]["duration"],
            "Author": video["snippet"]["channelTitle"],
            "Views": video["statistics"].get("viewCount", "0"),
            "PublishedAt": video["snippet"]["publishedAt"]
        }
    
    except Exception as e:
        print(f"Error getting video details: {str(e)}")
        return None

def get_video_captions(video_id):
    """Get captions using YouTube Transcript API."""
    try:
        print(f"Getting captions for video {video_id}")
        
        # Get available transcripts
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Try to find English or Vietnamese transcript
        transcript = None
        language = None
        
        # Try manually created transcripts first
        for lang_code in ['en', 'vi']:
            try:
                transcript = transcript_list.find_transcript([lang_code])
                language = lang_code
                break
            except NoTranscriptFound:
                continue
        
        # If no manual transcript, try auto-generated
        if not transcript:
            try:
                # Get all available transcripts
                available_transcripts = list(transcript_list)
                
                # Try to find English or Vietnamese
                for t in available_transcripts:
                    if t.language_code in ['en', 'vi']:
                        transcript = t
                        language = t.language_code
                        break
                
                # If still no transcript, use the first available and translate to English
                if not transcript and available_transcripts:
                    transcript = available_transcripts[0]
                    transcript = transcript.translate('en')
                    language = 'en'
            except Exception as e:
                print(f"Error finding transcript: {str(e)}")
        
        if not transcript:
            return None, None
        
        # Get transcript data
        try:
            transcript_data = transcript.fetch()
        except (xml.etree.ElementTree.ParseError, xml.parsers.expat.ExpatError) as e:
            print(f"XML parsing error for video {video_id}: {str(e)}")
            return None, None
        
        # Extract text
        try:
            if isinstance(transcript_data, list):
                text = " ".join([entry.get('text', '') for entry in transcript_data])
            else:
                if hasattr(transcript_data, 'text'):
                    text = transcript_data.text
                elif hasattr(transcript_data, 'get_text'):
                    text = transcript_data.get_text()
                elif hasattr(transcript_data, '__str__'):
                    text = str(transcript_data)
                else:
                    print(f"Unknown transcript format: {type(transcript_data)}")
                    return None, None
        except Exception as e:
            print(f"Error extracting text from transcript: {str(e)}")
            return None, None
        
        return text, language
    
    except (TranscriptsDisabled, NoTranscriptFound) as e:
        print(f"No transcripts available for video {video_id}: {str(e)}")
        return None, None
    except Exception as e:
        print(f"Error with YouTube Transcript API: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def process_video(video_id):
    """Process a single YouTube video using API Key."""
    try:
        # Get video details
        print(f"Getting details for video {video_id}...")
        video_info = get_video_details(video_id)
        if not video_info:
            print(f"Could not get details for video {video_id}")
            return None
            
        # Get captions
        print(f"Getting captions for video: {video_info['Title']}...")
        captions, language = get_video_captions(video_id)
        
        if not captions:
            print(f"No captions found for video {video_id}")
            return None
            
        print(f"Processing video: {video_info['Title']} (language: {language})")
        
        # Preprocess captions to reduce length if too long
        if len(captions) > 10000:
            print(f"Captions too long ({len(captions)} chars), truncating...")
            # Keep first and last parts of the transcript for better context
            captions = captions[:5000] + " ... " + captions[-5000:]
        
        # Generate summary with improved timeout handling
        print(f"Generating summary for: {video_info['Title']}...")
        try:
            # First try with a shorter timeout
            summary = summarize_with_improved_timeout(captions, language, video_info["Title"])
            
            if not summary or summary.startswith("Summary of") and "auto-generated" in summary:
                print(f"Failed to generate summary for {video_info['Title']}, trying chunked approach")
                # Try chunked approach if the first attempt failed
                summary = summarize_long_text(captions, language, video_info["Title"])
        except Exception as e:
            print(f"Error generating summary: {str(e)}")
            summary = f"Summary of {video_info['Title']} (auto-generated due to error)"
        
        # Calculate metrics with simplified approach for speed
        print(f"Calculating metrics for: {video_info['Title']}...")
        try:
            # Use a simpler/faster metric calculation if full metrics are too slow
            P, R, F1 = compute_bertscore(summary, captions[:1000], lang=language)  # Use shorter text
            rouge_scores = compute_rouge(summary, captions[:1000])  # Use shorter text
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
            P, R, F1 = 0, 0, 0
            rouge_scores = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
        
        print(f"Completed processing: {video_info['Title']}")
        
        # Create result
        result = {
            "Title": video_info["Title"],
            "URL": video_info["URL"],
            "Duration": video_info["Duration"],
            "Author": video_info["Author"],
            "Views": video_info["Views"],
            "Summary": summary,
            "BERTScore_F1": F1,
            "ROUGE-1": rouge_scores["rouge1"],
            "ROUGE-2": rouge_scores["rouge2"],
            "ROUGE-L": rouge_scores["rougeL"],
            "Language": language,
            "Source": "captions"
        }
        
        return result
    
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def summarize_with_improved_timeout(text, lang, title, timeout=90):
    """Summarize text with improved timeout handling."""
    from concurrent.futures import ThreadPoolExecutor
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(summarize_text, text, lang)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            print(f"Summary generation timed out for {title}")
            return f"Summary of {title} (auto-generated due to timeout)"

def summarize_long_text(text, lang, title, chunk_size=2000, overlap=200):
    """Summarize long text by breaking it into chunks."""
    from src.models.summarization_model import summarize_text
    
    # If text is short enough, try direct summarization
    if len(text) < chunk_size:
        try:
            return summarize_text(text, lang)
        except Exception as e:
            print(f"Error in direct summarization: {str(e)}")
            return f"Summary of {title} (auto-generated due to error)"
    
    # Break text into chunks with overlap
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if len(chunk) > 200:  # Only include chunks with substantial content
            chunks.append(chunk)
    
    # Summarize each chunk
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        try:
            print(f"Summarizing chunk {i+1}/{len(chunks)} for {title}")
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(summarize_text, chunk, lang)
                chunk_summary = future.result(timeout=30)  # Shorter timeout per chunk
                if chunk_summary and not chunk_summary.startswith("Could not generate summary"):
                    chunk_summaries.append(chunk_summary)
        except Exception as e:
            print(f"Error summarizing chunk {i+1}: {str(e)}")
            continue
    
    # If we have chunk summaries, combine them and create a final summary
    if chunk_summaries:
        combined_summary = " ".join(chunk_summaries)
        
        # If combined summary is still too long, summarize it again
        if len(combined_summary) > 3000:
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(summarize_text, combined_summary, lang)
                    final_summary = future.result(timeout=45)
                    if final_summary and not final_summary.startswith("Could not generate summary"):
                        return final_summary
            except Exception as e:
                print(f"Error in final summarization: {str(e)}")
                return combined_summary[:3000] + "..."
        
        return combined_summary
    
    return f"Summary of {title} (auto-generated due to processing error)"

def process_youtube_playlist_api_key(url, max_videos=MAX_VIDEOS_PER_PLAYLIST, progress_callback=None, processed_videos=None):
    """Process a YouTube playlist using YouTube API with API key."""
    # Create directories
    os.makedirs("output", exist_ok=True)
    
    try:
        # Extract playlist ID
        playlist_id = extract_playlist_id(url)
        if not playlist_id:
            if progress_callback:
                progress_callback(0, 1, "[ERROR] Invalid playlist URL")
            print("[ERROR] Invalid playlist URL")
            return None
        
        # Get playlist videos - get more videos than requested to have backups
        if progress_callback:
            progress_callback(0.05, 1, "Fetching playlist information...", None)
            
        playlist_title, all_video_ids = get_playlist_videos(playlist_id, max_videos * 2)
        if not all_video_ids:
            if progress_callback:
                progress_callback(0.1, 1, "[ERROR] No videos found in playlist", None)
            print("[ERROR] No videos found in playlist")
            return None
        
        print(f"Playlist: {playlist_title}")
        print(f"Found {len(all_video_ids)} videos, will process {max_videos}")
        
        # Don't send "Found X videos" message to UI
        # Instead, just update the progress silently
        if progress_callback:
            progress_callback(0.1, 1, "", None)
        
        # Process videos with improved parallelism and ensure we get the requested number
        results = []
        processed_count = 0
        video_index = 0
        
        # Keep processing videos until we have the requested number or run out of videos
        while processed_count < max_videos and video_index < len(all_video_ids):
            # Get the next batch of videos to process
            batch_size = min(4, max_videos - processed_count)  # Process up to 4 at a time for better UI updates
            current_batch = all_video_ids[video_index:video_index + batch_size]
            video_index += batch_size
            
            # Process the current batch
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = {executor.submit(process_video, video_id): video_id for video_id in current_batch}
                
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing videos (Batch {processed_count//4 + 1})"):
                    video_id = futures[future]
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                            processed_count += 1
                            print(f"Processed: {result['Title']} ({processed_count}/{max_videos})")
                            
                            # Update progress and pass the result
                            if progress_callback:
                                progress_value = 0.1 + 0.8 * (processed_count / max_videos)
                                progress_callback(
                                    progress_value, 
                                    1, 
                                    f"Processed {processed_count}/{max_videos} videos: {result['Title']}",
                                    result
                                )
                        else:
                            print(f"Failed to process video {video_id}, will try another video")
                            # Don't send failure messages to UI
                    except Exception as e:
                        print(f"Error processing video {video_id}: {str(e)}")
                        # Don't send error messages to UI
            
            print(f"Processed {processed_count}/{max_videos} videos so far")
        
        if not results:
            if progress_callback:
                progress_callback(1, 1, "No video processed successfully", None)
            print("[WARN] No video processed successfully")
            return None
        
        print(f"Successfully processed {len(results)}/{max_videos} videos")
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save results
        if progress_callback:
            progress_callback(0.95, 1, "Saving results...", None)
            
        output_path = "output/youtube_summary_results_api_key.csv"
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
        
        # Try to upload to MongoDB if configured
        try:
            if progress_callback:
                progress_callback(0.98, 1, "Uploading to MongoDB...", None)
                
            from src.storage.mongodb_handler_pandas import upload_to_mongodb_pandas
            upload_to_mongodb_pandas(df)
            print("Results uploaded to MongoDB")
        except Exception as e:
            print(f"Error uploading to MongoDB: {str(e)}")
        
        if progress_callback:
            progress_callback(1, 1, "Processing complete!", None)
            
        return df
    
    except Exception as e:
        print(f"Error processing playlist: {str(e)}")
        import traceback
        traceback.print_exc()
        if progress_callback:
            progress_callback(1, 1, f"Error: {str(e)}", None)
        return None

def main():
    """Main entry point for the command line."""
    parser = argparse.ArgumentParser(description="Process YouTube playlists using API Key")
    parser.add_argument("url", help="YouTube playlist URL")
    parser.add_argument("--max-videos", type=int, default=MAX_VIDEOS_PER_PLAYLIST, 
                        help="Maximum number of videos to process")
    
    args = parser.parse_args()
    process_youtube_playlist_api_key(args.url, args.max_videos)

if __name__ == "__main__":
    main()




