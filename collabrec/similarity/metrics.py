# This file contains the actual similarity measures.
# These functions compare two users or two items based on their rating vectors.

import math
from typing import Dict
from collabrec.similarity.core import dot_product, vector_norm
from collabrec.constants import DEFAULT_MIN_COMMON

def cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float],
                      min_common: int = DEFAULT_MIN_COMMON) -> float:
    """
    Compute the cosine similarity between two vectors.
    Cosine similarity is the dot product divided by the product of the lengths.
    It returns a value between -1 and 1. A value close to 1 means the vectors point in the same direction.

    Only the common keys are used for the dot product, but the full lengths of each vector are used in the denominator.

    Parameters:
        vec_a, vec_b: the vectors to compare (dictionaries)
        min_common: the minimum number of common keys required. If there are fewer, the result is 0.

    Returns:
        a similarity score between -1 and 1, or 0 if not enough common keys or if a vector has zero length.
    """
    common = set(vec_a.keys()) & set(vec_b.keys())
    if len(common) < min_common:
        return 0.0
    dp = sum(vec_a[k] * vec_b[k] for k in common)
    norm_a = vector_norm(vec_a)
    norm_b = vector_norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dp / (norm_a * norm_b)

def pearson_correlation(vec_a: Dict[str, float], vec_b: Dict[str, float],
                        min_common: int = DEFAULT_MIN_COMMON) -> float:
    """
    Compute the Pearson correlation between two vectors.
    This is like cosine similarity, but first the mean of each vector over the common keys is subtracted.
    It measures linear relationship while ignoring differences in average levels.

    The formula is the standard Pearson correlation coefficient.

    Parameters:
        vec_a, vec_b: the vectors to compare
        min_common: minimum number of common keys required

    Returns:
        correlation between -1 and 1, or 0 if not enough common keys.
    """
    common = set(vec_a.keys()) & set(vec_b.keys())
    n = len(common)
    if n < min_common:
        return 0.0
    vals_a = [vec_a[k] for k in common]
    vals_b = [vec_b[k] for k in common]
    mean_a = sum(vals_a) / n
    mean_b = sum(vals_b) / n
    num = sum((a - mean_a) * (b - mean_b) for a, b in zip(vals_a, vals_b))
    denom_a = math.sqrt(sum((a - mean_a) ** 2 for a in vals_a))
    denom_b = math.sqrt(sum((b - mean_b) ** 2 for b in vals_b))
    denom = denom_a * denom_b
    return num / denom if denom != 0.0 else 0.0

def adjusted_cosine_similarity(item_a: str, item_b: str, matrix,
                               min_common: int = DEFAULT_MIN_COMMON) -> float:
    """
    Compute the adjusted cosine similarity between two items.
    This is a special version used in item‑based collaborative filtering.
    For each user who rated both items, the user's average rating is subtracted from each rating.
    This removes the bias of users who tend to rate everything high or low.

    The formula is the same as cosine similarity, but applied to the centered ratings (rating minus user mean).

    Parameters:
        item_a, item_b: the items to compare
        matrix: the RatingMatrix that contains the ratings (needed to get user means)
        min_common: minimum number of users who rated both items

    Returns:
        adjusted cosine similarity between -1 and 1.
    """
    ratings_a = matrix.get_item_ratings(item_a)   # dictionary: user -> rating
    ratings_b = matrix.get_item_ratings(item_b)
    common_users = set(ratings_a.keys()) & set(ratings_b.keys())

    if len(common_users) < min_common:
        return 0.0

    num = 0.0
    sum_sq_a = 0.0
    sum_sq_b = 0.0

    for u in common_users:
        u_mean = matrix.user_mean(u)
        if u_mean is None:
            continue   # should not happen, but safety check
        diff_a = ratings_a[u] - u_mean
        diff_b = ratings_b[u] - u_mean
        num += diff_a * diff_b
        sum_sq_a += diff_a ** 2
        sum_sq_b += diff_b ** 2

    denom = math.sqrt(sum_sq_a) * math.sqrt(sum_sq_b)
    return num / denom if denom != 0.0 else 0.0