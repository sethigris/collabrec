# This file defines the base class that all recommenders inherit from.
# It sets up common attributes and methods, and specifies what every recommender must implement.


from typing import Dict, List, Optional, Tuple
from collabrec.data import RatingMatrix, SimilarityCache, ModelError, UserNotFoundError
from collabrec.constants import DEFAULT_SIMILARITY, DEFAULT_MIN_COMMON, DEFAULT_NEIGHBOUR_K, DEFAULT_TOP_N
from collabrec.similarity import select_similarity_function

class Recommendation:
    """
    This class holds one recommendation: an item, its predicted score, and a confidence value.
    The rank indicates its position in the list (1 = best).

    Attributes:
        item_id: the identifier of the recommended item
        predicted_score: the estimated rating as a float
        confidence: a number between 0 and 1 indicating how reliable this prediction is
        rank: the 1‑based position in the recommendation list
    """
    __slots__ = ("item_id", "predicted_score", "confidence", "rank")

    def __init__(self, item_id: str, predicted_score: float,
                 confidence: float = 1.0, rank: int = 0) -> None:
        self.item_id = item_id
        self.predicted_score = predicted_score
        self.confidence = max(0.0, min(1.0, confidence))
        self.rank = rank

    def to_dict(self) -> Dict:
        """Convert the recommendation to a dictionary, suitable for JSON export."""
        return {
            "rank": self.rank,
            "item_id": self.item_id,
            "predicted_score": round(self.predicted_score, 4),
            "confidence": round(self.confidence, 4),
        }

    def __repr__(self) -> str:
        """A string representation for debugging purposes."""
        return (f"Recommendation(rank={self.rank}, item={self.item_id!r}, "
                f"score={self.predicted_score:.3f}, conf={self.confidence:.2f})")

class BaseRecommender:
    """
    This is the abstract base class for all recommenders.
    It holds common settings like similarity metric, minimum common ratings, and number of neighbours (K).
    It also includes a cache for similarity scores.

    Subclasses must implement fit(), predict(), and recommend().
    """

    def __init__(self, similarity: str = DEFAULT_SIMILARITY,
                 min_common: int = DEFAULT_MIN_COMMON,
                 neighbour_k: int = DEFAULT_NEIGHBOUR_K) -> None:
        """
        Initialise a recommender with the given parameters.

        Parameters:
            similarity: name of the similarity metric ("cosine" or "pearson")
            min_common: minimum number of co‑rated items/users for a similarity score to be considered valid
            neighbour_k: number of neighbours to consider when making a prediction (K)
        """
        self.similarity_name = similarity
        self.min_common = min_common
        self.neighbour_k = neighbour_k
        self.sim_fn = select_similarity_function(similarity)
        self._matrix: Optional[RatingMatrix] = None
        self._sim_cache = SimilarityCache()   # cache for pairwise similarities
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Public interface that subclasses must override
    # ------------------------------------------------------------------

    def fit(self, matrix: RatingMatrix) -> "BaseRecommender":
        """
        Train the model on the given rating matrix.
        This method must be called before predict() or recommend().

        Parameters:
            matrix: the training data

        Returns:
            self (to allow method chaining)
        """
        raise NotImplementedError

    def predict(self, user_id: str, item_id: str) -> float:
        """
        Predict the rating that a user would give to an item.

        Parameters:
            user_id: target user
            item_id: target item

        Returns:
            estimated rating as a float
        """
        raise NotImplementedError

    def recommend(self, user_id: str, top_n: int = DEFAULT_TOP_N,
                  exclude_seen: bool = True) -> List[Recommendation]:
        """
        Generate top‑N recommendations for a user.

        Parameters:
            user_id: target user
            top_n: number of recommendations to return
            exclude_seen: if True, items the user has already rated are excluded

        Returns:
            list of Recommendation objects, sorted by predicted score descending
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared helper methods
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        """Raise ModelError if the model has not been fitted."""
        if not self._is_fitted:
            raise ModelError("Model has not been fitted. Call fit() first.")

    def _check_user(self, user_id: str) -> None:
        """Raise UserNotFoundError if the user does not exist in the training data."""
        if not self._matrix.has_user(user_id):
            raise UserNotFoundError(f"User {user_id!r} not found.")

    @staticmethod
    def _rank_recommendations(scored_items: List[Tuple[str, float, float]],
                               top_n: int) -> List[Recommendation]:
        """
        Convert a list of (item_id, score, confidence) into a ranked list of Recommendations.
        Items are sorted by score descending, then by item_id ascending to break ties deterministically.
        Rank numbers are assigned starting from 1.

        Parameters:
            scored_items: list of tuples (item_id, score, confidence)
            top_n: maximum number to return

        Returns:
            list of Recommendation
        """
        scored_items.sort(key=lambda x: (-x[1], x[0]))
        result = []
        for rank, (iid, score, conf) in enumerate(scored_items[:top_n], start=1):
            result.append(Recommendation(iid, score, conf, rank))
        return result

    # ------------------------------------------------------------------
    # Optional methods for similarity exploration
    # ------------------------------------------------------------------

    def get_similar_items(self, item_id: str, top_n: int = DEFAULT_TOP_N) -> List[Tuple[str, float]]:
        """
        Return the top‑N items most similar to the given item.

        Parameters:
            item_id: target item
            top_n: number of neighbours

        Returns:
            list of (neighbour_item_id, similarity_score)
        """
        raise NotImplementedError

    def get_similar_users(self, user_id: str, top_n: int = DEFAULT_TOP_N) -> List[Tuple[str, float]]:
        """
        Return the top‑N users most similar to the given user.

        Parameters:
            user_id: target user
            top_n: number of neighbours

        Returns:
            list of (neighbour_user_id, similarity_score)
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        fitted = "fitted" if self._is_fitted else "unfitted"
        return (f"{self.__class__.__name__}(similarity={self.similarity_name!r}, "
                f"k={self.neighbour_k}, min_common={self.min_common}, {fitted})")