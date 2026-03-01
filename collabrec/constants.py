# These are the default settings for the whole program.
# They are defined here so they are easy to change and keep consistent.

# DEFAULT_TOP_N is the number of recommendations given by default.
DEFAULT_TOP_N = 10

# DEFAULT_MIN_COMMON is the minimum number of ratings two users or two items must share
# before their similarity score is considered meaningful. The default is 2.
DEFAULT_MIN_COMMON = 2

# DEFAULT_SIMILARITY is the method used to compare users or items.
# The default is cosine similarity.
DEFAULT_SIMILARITY = "cosine"

# DEFAULT_METHOD determines whether item‑based or user‑based collaborative filtering is used.
# The default is item‑based.
DEFAULT_METHOD = "item"

# DEFAULT_NEIGHBOUR_K is how many similar users or items are considered when making a prediction.
# The letter K stands for the number of neighbours. The default is 20.
DEFAULT_NEIGHBOUR_K = 20

# DEFAULT_RANDOM_SEED is a number used to make random choices repeatable.
# Using the same seed gives the same "random" split every time.
DEFAULT_RANDOM_SEED = 42

# DEFAULT_EVAL_SPLIT is the fraction of data kept for training during a random split.
# The default is 0.8, meaning 80% for training and 20% for testing.
DEFAULT_EVAL_SPLIT = 0.8

# These are the names for the two similarity metrics that are supported.
SIMILARITY_COSINE = "cosine"
SIMILARITY_PEARSON = "pearson"

# These are the names for the two recommendation methods.
METHOD_ITEM = "item"
METHOD_USER = "user"

# MISSING is a special value used to mean "no rating exists here".
# It is represented by None.
MISSING = None