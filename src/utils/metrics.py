"""Metrics for evaluating summarization quality."""

from bert_score import score as bertscore
from rouge_score import rouge_scorer

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
        
    # Use appropriate model based on language
    model_type = "vinai/phobert-base" if lang == "vi" else "bert-base-uncased"
    
    try:
        P, R, F1 = bertscore(
            [candidate], 
            [reference], 
            lang=lang,
            model_type=model_type,
            verbose=False
        )
        
        return P.item(), R.item(), F1.item()
    except Exception as e:
        print(f"Error computing BERTScore: {str(e)}")
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