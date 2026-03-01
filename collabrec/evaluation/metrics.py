# Evaluation metrics: RMSE, MAE, coverage, and a container for results.

import math
from typing import List, Optional, Tuple, Dict
from collabrec.models import BaseRecommender
from collabrec.data import RatingMatrix
from collabrec.constants import DEFAULT_TOP_N


def rmse(predictions: List[Tuple[float, float]]) -> float:
    """
    Compute Root Mean Squared Error.

    Parameters:
        predictions: list of (predicted, actual) pairs

    Returns:
        RMSE value, or NaN if the list is empty.
    """
    if not predictions:
        return float("nan")
    mse = sum((p - a) ** 2 for p, a in predictions) / len(predictions)
    return math.sqrt(mse)

def mae(predictions: List[Tuple[float, float]]) -> float:
    """
    Compute Mean Absolute Error.

    Parameters:
        predictions: list of (predicted, actual) pairs

    Returns:
        MAE value, or NaN if the list is empty.
    """
    if not predictions:
        return float("nan")
    return sum(abs(p - a) for p, a in predictions) / len(predictions)

def coverage(model: BaseRecommender, matrix: RatingMatrix,
             top_n: int = DEFAULT_TOP_N) -> float:
    """
    Compute recommendation coverage: the fraction of users for whom the model
    can generate at least one recommendation.

    Parameters:
        model: a fitted recommender
        matrix: the dataset (users are taken from here)
        top_n: number of recommendations to request (only 1 is actually requested to test coverage)

    Returns:
        fraction of users who receive at least one recommendation (between 0 and 1)
    """
    users = matrix.users
    can_rec = 0
    for uid in users:
        try:
            recs = model.recommend(uid, top_n=1)
            if recs:
                can_rec += 1
        except Exception:
            pass   # any error means the model could not recommend for this user
    return can_rec / len(users) if users else 0.0

class EvaluationResult:
    """
    Container for evaluation metrics and metadata.

    Attributes:
        rmse_value (float): RMSE
        mae_value (float): MAE
        num_predictions (int): number of successful predictions made
        num_total (int): total number of test cases attempted
        coverage_value (float, optional): coverage fraction
        elapsed_seconds (float): wall‑clock time taken for the evaluation
    """

    def __init__(self, rmse_value: float, mae_value: float,
                 num_predictions: int, num_total: int,
                 coverage_value: Optional[float] = None,
                 elapsed_seconds: float = 0.0) -> None:
        self.rmse_value = rmse_value
        self.mae_value = mae_value
        self.num_predictions = num_predictions
        self.num_total = num_total
        self.coverage_value = coverage_value
        self.elapsed_seconds = elapsed_seconds

    def to_dict(self) -> Dict:
        """Convert the result to a dictionary, suitable for JSON export."""
        d = {
            "rmse": round(self.rmse_value, 6) if not math.isnan(self.rmse_value) else None,
            "mae": round(self.mae_value, 6) if not math.isnan(self.mae_value) else None,
            "predictions_made": self.num_predictions,
            "predictions_attempted": self.num_total,
            "prediction_rate": round(self.num_predictions / self.num_total, 4) if self.num_total else None,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }
        if self.coverage_value is not None:
            d["coverage"] = round(self.coverage_value, 4)
        return d

    def __repr__(self) -> str:
        return (f"EvaluationResult(RMSE={self.rmse_value:.4f}, MAE={self.mae_value:.4f}, "
                f"predictions={self.num_predictions}/{self.num_total})")