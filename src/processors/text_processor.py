"""Text preprocessing utilities for NLP tasks."""

from pyspark.sql import DataFrame
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml import Pipeline

def normalize_text(text):
    """Normalize text by converting to lowercase and removing punctuation."""
    import re
    import string
    
    if not text:
        return ""
        
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    return text

# Create UDF for Spark
normalize_udf = udf(normalize_text, StringType())

def preprocess_text(text_df: DataFrame) -> DataFrame:
    """
    Preprocess text data for NLP tasks.
    
    Args:
        text_df: DataFrame with a 'text' column
        
    Returns:
        DataFrame with preprocessed text features
    """
    # Normalize text
    text_df = text_df.withColumn("text", normalize_udf(col("text")))
    
    # Create ML pipeline
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    tf = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=5000)
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    
    # Apply pipeline
    pipeline = Pipeline(stages=[tokenizer, remover, tf, idf])
    return pipeline.fit(text_df).transform(text_df)

def split_text_into_chunks(text, max_tokens=500):
    """
    Split text into chunks that don't exceed max_tokens.
    
    Args:
        text: Text to split
        max_tokens: Maximum number of tokens per chunk
        
    Returns:
        List of text chunks
    """
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    chunks = []
    current_chunk = ""
    current_token_count = 0
    
    for sentence in sentences:
        # Approximate token count (words + some overhead)
        sentence_token_count = len(sentence.split()) + 2
        
        if current_token_count + sentence_token_count <= max_tokens:
            current_chunk += sentence + ". "
            current_token_count += sentence_token_count
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence + ". "
            current_token_count = sentence_token_count
            
    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks