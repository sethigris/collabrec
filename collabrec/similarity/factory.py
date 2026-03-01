# This file provides a simple way to obtain the right similarity function by name.


from collabrec.similarity.metrics import cosine_similarity, pearson_correlation
from collabrec.constants import SIMILARITY_COSINE, SIMILARITY_PEARSON

def select_similarity_function(name: str):
    """
    Take a name like "cosine" or "pearson" and return the corresponding similarity function.
    If the name is not recognised, a ValueError is raised.
    """
    name = name.lower().strip()
    if name == SIMILARITY_COSINE:
        return cosine_similarity
    if name == SIMILARITY_PEARSON:
        return pearson_correlation
    raise ValueError(f"Unknown similarity metric: {name}")