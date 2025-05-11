"""Metrics for evaluating summarization quality."""

def compute_bertscore(candidate, reference, lang="en"):
    """
    Compute BERTScore for summarization evaluation.
    
    Args:
        candidate: Candidate summary text
        reference: Reference text
        lang: Language code ('en' or 'vi')
        
    Returns:
        Tuple of (Precision, Recall, F1)
    """
    if not candidate or not reference:
        return 0.0, 0.0, 0.0
    
    try:
        # Import here to avoid loading models unnecessarily
        from bert_score import score as bertscore
        import torch
        
        # Ensure inputs are strings
        if not isinstance(candidate, str):
            candidate = str(candidate)
        if not isinstance(reference, str):
            reference = str(reference)
            
        # Truncate inputs if they are too long to avoid memory issues
        max_length = 512
        if len(candidate.split()) > max_length:
            candidate = " ".join(candidate.split()[:max_length])
        if len(reference.split()) > max_length:
            reference = " ".join(reference.split()[:max_length])
        
        # For Vietnamese, we'll use XLM-RoBERTa which has better multilingual support
        if lang == "vi":
            model_type = "xlm-roberta-base"
        else:
            model_type = "bert-base-uncased"
        
        # Compute BERTScore
        try:
            P, R, F1 = bertscore(
                [candidate], 
                [reference], 
                lang=lang,
                model_type=model_type,
                verbose=False
            )
            
            # Convert tensor to float if needed
            if hasattr(P, 'item'):
                P = P.item()
            if hasattr(R, 'item'):
                R = R.item()
            if hasattr(F1, 'item'):
                F1 = F1.item()
                
            # If F1 is 0 for Vietnamese, try with a different model
            if F1 == 0 and lang == 'vi':
                print("BERTScore returned 0 for Vietnamese, trying with XLM-RoBERTa-Large...")
                P, R, F1 = bertscore(
                    [candidate], 
                    [reference], 
                    model_type="xlm-roberta-large",
                    verbose=False
                )
                
                if hasattr(F1, 'item'):
                    F1 = F1.item()
            
            return float(P), float(R), float(F1)
        
        except Exception as e:
            print(f"Error in BERTScore calculation: {str(e)}")
            # Fall through to the fallback method
            raise e
            
    except Exception as e:
        print(f"Error computing BERTScore: {str(e)}")
        
        # Fallback to simple lexical overlap as a basic metric
        try:
            # Simple word overlap for Vietnamese
            if lang == 'vi':
                # For Vietnamese, we'll use character-level n-grams instead of words
                # since Vietnamese words can be composed of multiple syllables
                def get_character_ngrams(text, n=3):
                    return [text[i:i+n] for i in range(len(text) - n + 1)]
                
                candidate_ngrams = set(get_character_ngrams(candidate.lower()))
                reference_ngrams = set(get_character_ngrams(reference.lower()))
                
                if not candidate_ngrams or not reference_ngrams:
                    return 0.0, 0.0, 0.0
                    
                common_ngrams = candidate_ngrams.intersection(reference_ngrams)
                precision = len(common_ngrams) / len(candidate_ngrams) if candidate_ngrams else 0
                recall = len(common_ngrams) / len(reference_ngrams) if reference_ngrams else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                return precision, recall, f1
            else:
                # Word-level overlap for English
                candidate_words = set(candidate.lower().split())
                reference_words = set(reference.lower().split())
                
                if not candidate_words or not reference_words:
                    return 0.0, 0.0, 0.0
                    
                common_words = candidate_words.intersection(reference_words)
                precision = len(common_words) / len(candidate_words) if candidate_words else 0
                recall = len(common_words) / len(reference_words) if reference_words else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                return precision, recall, f1
        except Exception as fallback_error:
            print(f"Fallback method also failed: {str(fallback_error)}")
            return 0.0, 0.0, 0.0

def compute_rouge(candidate, reference):
    """
    Compute ROUGE scores for summarization evaluation.
    
    Args:
        candidate: Candidate summary text
        reference: Reference text
        
    Returns:
        Dictionary with ROUGE-1, ROUGE-2, and ROUGE-L scores
    """
    if not candidate or not reference:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, candidate)
        
        return {
            "rouge1": scores['rouge1'].fmeasure,
            "rouge2": scores['rouge2'].fmeasure,
            "rougeL": scores['rougeL'].fmeasure
        }
    except Exception as e:
        print(f"Error computing ROUGE scores: {str(e)}")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
