"""Streamlit app for YouTube Processor demo."""

import streamlit as st
import pandas as pd
import os
import time
from PIL import Image
import base64
from io import BytesIO

from src.main import process_youtube_playlist
from run_with_api_key import process_youtube_playlist_api_key
from simple_processor import process_youtube_playlist_simple

# Set page config
st.set_page_config(
    page_title="YouTube Processor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF0000;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #505050;
        margin-bottom: 20px;
    }
    .result-container {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .video-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1E3A8A;
    }
    .metric-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 10px;
    }
    .metric-box {
        background-color: #ffffff;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        flex: 1;
        min-width: 100px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #4B5563;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #6B7280;
    }
    .processing-video-title {
        font-size: 1.4rem;
        font-weight: bold;
        color: #FF0000;
        background-color: #FFF9C4;
        padding: 8px 12px;
        border-radius: 5px;
        margin: 10px 0;
        border-left: 5px solid #FFC107;
        display: inline-block;
    }
    .progress-info {
        background-color: #E3F2FD;
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
        border: 1px solid #BBDEFB;
    }
    .final-results-table {
        margin-top: 30px;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stDataFrame {
        padding: 0 !important;
    }
    .average-scores {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        margin: 20px 0;
        border: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def display_results(df):
    """Display processing results in a nice format."""
    if df is None or len(df) == 0:
        st.error("No results to display")
        return
    
    # Display summary stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Videos Processed", len(df))
    with col2:
        languages = df["Language"].value_counts().to_dict()
        lang_str = ", ".join([f"{k}: {v}" for k, v in languages.items()])
        st.metric("Languages", lang_str)
    with col3:
        avg_bertscore = df["BERTScore_F1"].mean()
        st.metric("Avg. BERTScore", f"{avg_bertscore:.2f}")
    
    # Display each video result
    for _, row in df.iterrows():
        with st.expander(f"📹 {row['Title']}"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**URL:** [{row['URL']}]({row['URL']})")
                st.markdown(f"**Author:** {row['Author']}")
                st.markdown(f"**Duration:** {row['Duration']}")
                st.markdown(f"**Views:** {row['Views']}")
                st.markdown(f"**Language:** {row['Language']}")
                
                st.markdown("### Summary")
                st.markdown(row["Summary"])
            
            with col2:
                st.markdown("### Metrics")
                st.markdown(f"**BERTScore F1:** {row['BERTScore_F1']:.4f}")
                st.markdown(f"**ROUGE-1:** {row['ROUGE-1']:.4f}")
                st.markdown(f"**ROUGE-2:** {row['ROUGE-2']:.4f}")
                st.markdown(f"**ROUGE-L:** {row['ROUGE-L']:.4f}")

# Main app
def main():
    st.markdown("<h1 class='main-header'>YouTube Processor</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Process YouTube playlists and generate summaries</p>", unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.image("https://img.icons8.com/color/96/000000/youtube-play.png", width=100)
    st.sidebar.title("Settings")

    max_videos = st.sidebar.slider(
        "Maximum Videos to Process",
        min_value=1,
        max_value=30,
        value=5,
        help="Maximum number of videos to process from the playlist"
    )

    # Main content
    playlist_url = st.text_input("Enter YouTube Playlist URL", 
                                placeholder="https://www.youtube.com/playlist?list=...")

    process_button = st.button("Process Playlist", type="primary")

    if process_button and playlist_url:
        # Create progress tracking elements
        progress_bar = st.progress(0)
        status_container = st.container()
        video_info = st.empty()
        
        # Create a container for displaying results in real-time
        results_container = st.container()
        with results_container:
            st.subheader("Processed Videos")
            processed_videos_container = st.empty()
        
        # Create a container for final results table
        final_results_container = st.container()
        
        # Initialize a list to store processed videos
        processed_videos = []
        
        # Create a callback function to update progress and display results
        def update_progress(current, total, message, result=None):
            progress_bar.progress(current)
            
            # Only display processing information for successfully processed videos
            if "Processed" in message and "videos:" in message and result is not None:
                video_title = message.split("videos:")[1].strip()
                video_count = message.split("Processed")[1].split("videos:")[0].strip()
                
                # Display with enhanced styling
                video_info.markdown(f"""
                <div class="progress-info">
                    <div>{message.split('videos:')[0]} videos</div>
                    <div class="processing-video-title">{video_title}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Add result to processed videos and update display
                processed_videos.append(result)
                display_processed_videos(processed_videos)
                
            elif "Processing complete" in message:
                # Show completion message
                status_container.text(message)
                # Display final results table only when processing is complete
                display_final_results_table(processed_videos)
            elif "Saving results" in message or "Uploading to MongoDB" in message:
                # Only show other completion messages
                status_container.text(message)
            # Skip displaying "Found X videos" and error messages on UI
        
        # Function to display processed videos in real-time
        def display_processed_videos(videos):
            if not videos:
                processed_videos_container.info("No videos processed yet.")
                return
            
            # Create a temporary DataFrame
            import pandas as pd
            temp_df = pd.DataFrame(videos)
            
            # Display each video
            content = ""
            for i, row in temp_df.iterrows():
                content += f"""
                <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 15px; border-left: 5px solid #1E88E5;">
                    <h3 style="color: #1E3A8A;">{i+1}. {row['Title']}</h3>
                    <p><strong>Author:</strong> {row['Author']} | <strong>Language:</strong> {row['Language']} | <strong>BERTScore:</strong> {row['BERTScore_F1']:.4f}</p>
                    <details>
                        <summary style="cursor: pointer; color: #2962FF; font-weight: bold;">View Summary</summary>
                        <div style="padding: 10px; background-color: white; border-radius: 5px; margin-top: 10px;">
                            {row['Summary']}
                        </div>
                    </details>
                </div>
                """
            
            processed_videos_container.markdown(content, unsafe_allow_html=True)
        
        # Function to display final results table
        def display_final_results_table(videos):
            if not videos:
                return
            
            with final_results_container:
                st.markdown("### Final Results Table")
                
                # Create DataFrame
                import pandas as pd
                df = pd.DataFrame(videos)
                
                # Select only the columns we want to display
                display_columns = ["Title", "URL", "Duration", "Author", "Views", 
                                  "BERTScore_F1", "ROUGE-1", "ROUGE-2", "ROUGE-L", 
                                  "Language", "Source"]
                
                # Make sure all columns exist
                for col in display_columns:
                    if col not in df.columns:
                        df[col] = "N/A"
                
                # Select and display the table
                display_df = df[display_columns]
                
                # Calculate average scores
                avg_bertscore = df["BERTScore_F1"].mean()
                avg_rouge1 = df["ROUGE-1"].mean()
                avg_rouge2 = df["ROUGE-2"].mean()
                avg_rougeL = df["ROUGE-L"].mean()
                
                # Display average scores
                st.markdown("#### Average Scores")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Avg. BERTScore", f"{avg_bertscore:.4f}")
                with col2:
                    st.metric("Avg. ROUGE-1", f"{avg_rouge1:.4f}")
                with col3:
                    st.metric("Avg. ROUGE-2", f"{avg_rouge2:.4f}")
                with col4:
                    st.metric("Avg. ROUGE-L", f"{avg_rougeL:.4f}")
                
                # Apply styling to the DataFrame
                styled_df = display_df.style.format({
                    "BERTScore_F1": "{:.4f}",
                    "ROUGE-1": "{:.4f}",
                    "ROUGE-2": "{:.4f}",
                    "ROUGE-L": "{:.4f}"
                })
                
                # Add background gradient to score columns
                styled_df = styled_df.background_gradient(
                    subset=["BERTScore_F1", "ROUGE-1", "ROUGE-2", "ROUGE-L"],
                    cmap="YlGnBu"
                )
                
                # Display the table
                st.dataframe(styled_df, use_container_width=True)
        
        with st.spinner("Processing playlist... This may take a few minutes."):
            try:
                # Process with API Key and pass the callback
                results = process_youtube_playlist_api_key(playlist_url, max_videos, update_progress, processed_videos)
                
                # Display final results
                if results is not None:
                    st.success(f"Successfully processed {len(results)} videos!")
                    # Final display is already handled by the callback
                else:
                    st.error("Failed to process playlist. Check the console for errors.")
            
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.exception(e)

if __name__ == "__main__":
    main()













