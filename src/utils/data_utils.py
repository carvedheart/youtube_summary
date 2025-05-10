"""Data handling utilities."""

import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyspark.sql import SparkSession

def read_parquet_to_pandas(path):
    """
    Read Parquet file(s) into a Pandas DataFrame.
    
    Args:
        path: Path to Parquet file or directory
        
    Returns:
        Pandas DataFrame
    """
    if os.path.isdir(path):
        # Read all parquet files in directory
        dfs = []
        for file in os.listdir(path):
            if file.endswith('.parquet'):
                file_path = os.path.join(path, file)
                dfs.append(pd.read_parquet(file_path))
        
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame()
    else:
        # Read single parquet file
        return pd.read_parquet(path)

def save_to_parquet(df, path, partition_cols=None):
    """
    Save DataFrame to Parquet format.
    
    Args:
        df: Pandas DataFrame
        path: Output path
        partition_cols: List of columns to partition by
    """
    table = pa.Table.from_pandas(df)
    
    if partition_cols:
        pq.write_to_dataset(
            table,
            root_path=path,
            partition_cols=partition_cols
        )
    else:
        pq.write_table(table, path)

def spark_to_pandas(spark_df):
    """
    Convert Spark DataFrame to Pandas DataFrame.
    
    Args:
        spark_df: Spark DataFrame
        
    Returns:
        Pandas DataFrame
    """
    return spark_df.toPandas()

def pandas_to_spark(pandas_df, spark=None):
    """
    Convert Pandas DataFrame to Spark DataFrame.
    
    Args:
        pandas_df: Pandas DataFrame
        spark: SparkSession (optional)
        
    Returns:
        Spark DataFrame
    """
    if spark is None:
        spark = SparkSession.builder.getOrCreate()
    
    return spark.createDataFrame(pandas_df)