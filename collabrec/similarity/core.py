# This file contains basic mathematical functions for working with sparse vectors.
# A sparse vector is represented as a dictionary where keys are dimensions (like item ids) and values are numbers (like ratings).

import math
from typing import Dict

def dot_product(a: Dict[str, float], b: Dict[str, float]) -> float:
    """
    Compute the dot product of two sparse vectors.
    The dot product is the sum of products of values that share the same key.
    Only keys that exist in both vectors contribute.
    To improve efficiency, the smaller dictionary is looped over.
    """
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b[k] for k, v in a.items() if k in b)

def vector_norm(v: Dict[str, float]) -> float:
    """
    Calculate the Euclidean length (L2 norm) of a sparse vector.
    This is the square root of the sum of squares of all values in the vector.
    """
    return math.sqrt(sum(x * x for x in v.values()))