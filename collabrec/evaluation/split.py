# Functions to split a rating matrix into training and test sets.

import random
from typing import List, Tuple
from collabrec.data import RatingMatrix, Rating
from collabrec.constants import DEFAULT_EVAL_SPLIT, DEFAULT_RANDOM_SEED


def random_train_test_split(matrix: RatingMatrix, train_fraction: float = DEFAULT_EVAL_SPLIT,
                            seed: int = DEFAULT_RANDOM_SEED) -> Tuple[RatingMatrix, RatingMatrix]:
    """
    Randomly split ratings into training and test sets.

    Each rating is independently assigned to train with probability train_fraction,
    and to test with probability 1 - train_fraction. The split is deterministic given the seed.

    A guarantee is provided: every user has at least one rating in the training set.
    If a user appears only in the test set, one of their test ratings is moved to training.

    Parameters:
        matrix: the full RatingMatrix
        train_fraction: fraction of ratings to use for training (must be between 0 and 1)
        seed: random seed for reproducibility

    Returns:
        (train_matrix, test_matrix)
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0,1), got {train_fraction}")

    rng = random.Random(seed)
    train = RatingMatrix()
    test = RatingMatrix()

    all_ratings = list(matrix.iter_ratings())
    rng.shuffle(all_ratings)

    for r in all_ratings:
        if rng.random() < train_fraction:
            train.add_rating(r.user_id, r.item_id, r.rating)
        else:
            test.add_rating(r.user_id, r.item_id, r.rating)

    # Ensure every user has at least one training rating
    for user_id in test.users:
        if not train.has_user(user_id):
            user_test_ratings = list(test.get_user_ratings(user_id).items())
            if user_test_ratings:
                iid, rating = user_test_ratings[0]
                train.add_rating(user_id, iid, rating)
                test.remove_rating(user_id, iid)

    return train, test

def leave_one_out_split(matrix: RatingMatrix,
                        seed: int = DEFAULT_RANDOM_SEED) -> List[Tuple[RatingMatrix, Rating]]:
    """
    Generate Leave‑One‑Out splits.

    For each user with at least 2 ratings, one rating is randomly withheld as the test item,
    and the remaining ratings form the training set.

    Parameters:
        matrix: the full RatingMatrix
        seed: random seed for selecting the withheld rating

    Returns:
        list of (train_matrix, held_out_Rating) tuples, one per eligible user
    """
    rng = random.Random(seed)
    splits = []
    for user_id in matrix.users:
        user_ratings = list(matrix.get_user_ratings(user_id).items())
        if len(user_ratings) < 2:
            continue   # need at least one for training + one for testing
        rng.shuffle(user_ratings)
        held_iid, held_r = user_ratings[0]
        held_out = Rating(user_id, held_iid, held_r)
        train = matrix.copy()
        train.remove_rating(user_id, held_iid)
        splits.append((train, held_out))
    return splits