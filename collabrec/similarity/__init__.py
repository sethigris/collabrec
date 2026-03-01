# This package provides functions to compare users or items.
# It includes core mathematical operations, the actual similarity metrics, and a factory to select the right one.


from collabrec.similarity.core import dot_product, vector_norm
from collabrec.similarity.metrics import cosine_similarity, pearson_correlation, adjusted_cosine_similarity
from collabrec.similarity.factory import select_similarity_function