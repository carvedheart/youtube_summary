"""MongoDB interactions for storing and retrieving data."""

import os
import json
import pandas as pd
from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError
from tqdm import tqdm
from src.config import MONGO_CONFIG
from src.utils.data_utils import read_parquet_to_pandas

def get_mongodb_client():
    """Get MongoDB client using configuration."""
    return MongoClient(MONGO_CONFIG["connection_string"])

def get_collection():
    """Get MongoDB collection using configuration."""
    client = get_mongodb_client()
    db = client[MONGO_CONFIG["database_name"]]
    return db[MONGO_CONFIG["collection_name"]]

def upload_to_mongodb(parquet_path):
    """
    Upload data from Parquet files to MongoDB.
    
    Args:
        parquet_path: Path to Parquet file or directory
    """
    try:
        print("🔄 Đang kết nối MongoDB...")
        client = get_mongodb_client()
        db = client[MONGO_CONFIG["database_name"]]
        collection = db[MONGO_CONFIG["collection_name"]]
        client.admin.command('ping')
        print(f"✅ Đã kết nối tới collection: {MONGO_CONFIG['collection_name']}")
        
        # Read Parquet data
        print(f"📖 Đang đọc file {parquet_path}...")
        df = read_parquet_to_pandas(parquet_path)
        
        if df.empty:
            print("No data to upload to MongoDB")
            return
            
        # Convert to records (list of dictionaries)
        records = json.loads(df.to_json(orient='records'))
        total_records = len(records)
        print(f"🔼 Đang upload {total_records} bản ghi...")
        
        # Insert records in batches
        batch_size = 50
        for i in tqdm(range(0, total_records, batch_size), desc="Uploading"):
            batch = records[i:i + batch_size]
            collection.insert_many(batch)
            
        print(f"🎉 Hoàn thành! Đã upload {total_records} bản ghi vào MongoDB")
        
    except Exception as e:
        print(f"❌ Error uploading to MongoDB: {str(e)}")
    finally:
        if 'client' in locals():
            client.close()

def fetch_from_mongodb(query=None, limit=100):
    """
    Fetch data from MongoDB.
    
    Args:
        query: MongoDB query (optional)
        limit: Maximum number of records to fetch
        
    Returns:
        Pandas DataFrame
    """
    client = None
    try:
        client = get_mongodb_client()
        collection = get_collection()
        
        if query is None:
            query = {}
            
        cursor = collection.find(query).limit(limit)
        data = list(cursor)
        
        if not data:
            return pd.DataFrame()
            
        return pd.DataFrame(data)
        
    except Exception as e:
        print(f"❌ Error fetching from MongoDB: {str(e)}")
        return pd.DataFrame()
    finally:
        if client:
            client.close()

def update_mongodb_record(filter_query, update_data):
    """
    Update a record in MongoDB.
    
    Args:
        filter_query: Query to find the record
        update_data: Data to update
        
    Returns:
        Number of records updated
    """
    client = None
    try:
        client = get_mongodb_client()
        collection = get_collection()
        result = collection.update_one(filter_query, {"$set": update_data})
        return result.modified_count
    except Exception as e:
        print(f"❌ Error updating MongoDB record: {str(e)}")
        return 0
    finally:
        if client:
            client.close()

def bulk_update_mongodb(updates):
    """
    Perform bulk updates to MongoDB.
    
    Args:
        updates: List of (filter_query, update_data) tuples
        
    Returns:
        Number of records updated
    """
    client = None
    try:
        client = get_mongodb_client()
        collection = get_collection()
        
        operations = [
            UpdateOne(filter_query, {"$set": update_data})
            for filter_query, update_data in updates
        ]
        
        if not operations:
            return 0
            
        result = collection.bulk_write(operations)
        return result.modified_count
    except BulkWriteError as bwe:
        print(f"❌ Bulk write error: {bwe.details}")
        return bwe.details.get('nModified', 0)
    except Exception as e:
        print(f"❌ Error performing bulk update: {str(e)}")
        return 0
    finally:
        if client:
            client.close()

def delete_from_mongodb(filter_query):
    """
    Delete records from MongoDB.
    
    Args:
        filter_query: Query to find records to delete
        
    Returns:
        Number of records deleted
    """
    client = None
    try:
        client = get_mongodb_client()
        collection = get_collection()
        result = collection.delete_many(filter_query)
        return result.deleted_count
    except Exception as e:
        print(f"❌ Error deleting from MongoDB: {str(e)}")
        return 0
    finally:
        if client:
            client.close()
