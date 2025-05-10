"""Main entry point for the YouTube processor application."""

import os
from pyspark.sql import SparkSession
from pyspark import SparkConf
from multiprocessing import cpu_count
import pandas as pd

from src.processors.playlist_processor import process_playlist
from src.storage.mongodb_handler import upload_to_mongodb

def create_spark_session():
    """Create and configure a Spark session."""
    conf = SparkConf() \
        .set("spark.executor.memory", "12g") \
        .set("spark.driver.memory", "12g") \
        .set("spark.driver.maxResultSize", "6g") \
        .set("spark.sql.shuffle.partitions", str(cpu_count() * 2)) \
        .set("spark.default.parallelism", str(cpu_count() * 2)) \
        .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .set("spark.executor.extraJavaOptions", "-XX:+UseG1GC -XX:+UseCompressedOops") \
        .set("spark.driver.extraJavaOptions", "-XX:+UseG1GC -XX:+UseCompressedOops") \
        .set("spark.memory.fraction", "0.8") \
        .set("spark.memory.storageFraction", "0.3") \
        .set("spark.python.worker.memory", "1g") \
        .set("spark.locality.wait", "0")
    
    return SparkSession.builder \
        .appName("YouTubeProcessing") \
        .config(conf=conf) \
        .getOrCreate()

def process_youtube_playlist(url, max_videos=30):
    """Process a YouTube playlist and store results."""
    # Create directories
    os.makedirs("cache", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    
    # Initialize Spark
    spark = create_spark_session()
    
    try:
        # Process playlist
        df = process_playlist(url, spark, max_videos=max_videos)
        
        if df:
            # Convert to pandas for display
            pandas_df = df.toPandas()
            print(f"Processed {len(pandas_df)} videos successfully")
            
            # Save results
            parquet_path = "output/youtube_summary_results"
            df.write.partitionBy("Language") \
                .mode("overwrite") \
                .parquet(parquet_path)
            
            # Upload to MongoDB
            upload_to_mongodb(parquet_path)
            
            return pandas_df
        else:
            print("Không xử lý được video nào trong playlist.")
            return None
    
    except Exception as e:
        print(f"Lỗi khi xử lý playlist: {str(e)}")
        return None
    
    finally:
        spark.stop()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process YouTube playlists")
    parser.add_argument("url", help="YouTube playlist URL")
    parser.add_argument("--max-videos", type=int, default=30, 
                        help="Maximum number of videos to process")
    
    args = parser.parse_args()
    process_youtube_playlist(args.url, args.max_videos)