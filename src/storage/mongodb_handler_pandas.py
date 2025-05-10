"""MongoDB handler for pandas DataFrames."""

from pymongo import MongoClient
import pandas as pd
import os
from src.config import MONGO_CONFIG

def upload_to_mongodb_pandas(df):
    """Upload pandas DataFrame to MongoDB."""
    try:
        print("Connecting to MongoDB...")
        client = MongoClient(MONGO_CONFIG["connection_string"])
        db = client[MONGO_CONFIG["database_name"]]
        collection = db[MONGO_CONFIG["collection_name"]]
        
        # Convert DataFrame to records
        records = df.to_dict('records')
        
        # Insert records
        if records:
            collection.insert_many(records)
            print(f"Uploaded {len(records)} records to MongoDB")
        
        client.close()
        return True
    except Exception as e:
        print(f"Error uploading to MongoDB: {str(e)}")
        return False

def fetch_from_mongodb_pandas(limit=100):
    """Fetch data from MongoDB and return as pandas DataFrame."""
    try:
        client = MongoClient(MONGO_CONFIG["connection_string"])
        db = client[MONGO_CONFIG["database_name"]]
        collection = db[MONGO_CONFIG["collection_name"]]
        
        # Fetch records
        cursor = collection.find().limit(limit)
        records = list(cursor)
        
        # Convert to DataFrame
        df = pd.DataFrame(records)
        
        client.close()
        return df
    except Exception as e:
        print(f"Error fetching from MongoDB: {str(e)}")
        return pd.DataFrame()