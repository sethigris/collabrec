# Factory function to construct a recommender based on the chosen method.

from collabrec.constants import METHOD_ITEM, METHOD_USER, DEFAULT_METHOD, DEFAULT_SIMILARITY, DEFAULT_MIN_COMMON, DEFAULT_NEIGHBOUR_K
from collabrec.models.base import BaseRecommender
from collabrec.models.item_based import ItemBasedCF
from collabrec.models.user_based import UserBasedCF


def build_model(method: str = DEFAULT_METHOD, similarity: str = DEFAULT_SIMILARITY,
                min_common: int = DEFAULT_MIN_COMMON,
                neighbour_k: int = DEFAULT_NEIGHBOUR_K,
                use_adjusted_cosine: bool = False) -> BaseRecommender:
    """
    Create an unfitted recommender instance based on the method name.

    Parameters:
        method: "item" or "user"
        similarity: "cosine" or "pearson"
        min_common: minimum co‑rated items/users for similarity
        neighbour_k: number of neighbours to consider
        use_adjusted_cosine: only relevant when method="item" and similarity="cosine"

    Returns:
        An unfitted BaseRecommender subclass.

    Raises:
        ValueError if method is unknown.
    """
    method = method.lower().strip()
    if method == METHOD_ITEM:
        return ItemBasedCF(similarity, min_common, neighbour_k, use_adjusted_cosine)
    if method == METHOD_USER:
        return UserBasedCF(similarity, min_common, neighbour_k)
    raise ValueError(f"Unknown method: {method}")