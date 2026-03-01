# JSON export functions for recommendations and evaluation results.

import json
from typing import List
from collabrec.models import Recommendation
from collabrec.evaluation import EvaluationResult

def export_recommendations_json(user_id: str, recs: List[Recommendation],
                                method: str, similarity: str, filepath: str) -> None:
    """
    Write recommendations to a JSON file.

    The JSON structure includes metadata and a list of recommendations
    with rank, item_id, predicted_score, and confidence.

    Parameters:
        user_id: target user
        recs: list of Recommendation objects
        method: "item" or "user"
        similarity: "cosine" or "pearson"
        filepath: output path
    """
    payload = {
        "user_id": user_id,
        "method": method,
        "similarity": similarity,
        "top_n": len(recs),
        "recommendations": [r.to_dict() for r in recs],
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def export_evaluation_json(result: EvaluationResult, eval_mode: str,
                           method: str, similarity: str, filepath: str) -> None:
    """
    Write evaluation results to a JSON file.

    Parameters:
        result: EvaluationResult object
        eval_mode: "split" or "loo"
        method: "item" or "user"
        similarity: "cosine" or "pearson"
        filepath: output path
    """
    payload = {
        "eval_mode": eval_mode,
        "method": method,
        "similarity": similarity,
        **result.to_dict(),
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2) 