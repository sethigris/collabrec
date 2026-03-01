# High‑level evaluation runners that coordinate splits, model training, and metric computation.

import time
from collabrec.data import RatingMatrix
from collabrec.constants import DEFAULT_EVAL_SPLIT, DEFAULT_RANDOM_SEED
from collabrec.models import build_model
from collabrec.evaluation.split import random_train_test_split, leave_one_out_split
from collabrec.evaluation.metrics import rmse, mae, EvaluationResult


def evaluate_random_split(matrix: RatingMatrix, method: str = "item",
                          similarity: str = "cosine", min_common: int = 2,
                          neighbour_k: int = 20, train_fraction: float = DEFAULT_EVAL_SPLIT,
                          seed: int = DEFAULT_RANDOM_SEED,
                          verbose: bool = True) -> EvaluationResult:
    """
    Evaluate a model using a random train/test split.

    The model is trained on train_fraction of the ratings and evaluated on the rest.
    RMSE and MAE are computed over the test set.

    Parameters:
        matrix: full dataset
        method: "item" or "user"
        similarity: "cosine" or "pearson"
        min_common: minimum co‑rated items/users for similarity
        neighbour_k: neighbourhood size
        train_fraction: fraction of ratings used for training
        seed: random seed for split reproducibility
        verbose: if True, print progress information

    Returns:
        EvaluationResult
    """
    if verbose:
        print("\n" + "=" * 72)
        print(f"  Evaluation — Random Split (train={train_fraction:.0%})")
        print("=" * 72)

    t0 = time.time()
    train, test = random_train_test_split(matrix, train_fraction, seed)
    if verbose:
        print(f"  Train ratings: {train.num_ratings}  Test ratings: {test.num_ratings}")

    model = build_model(method, similarity, min_common, neighbour_k)
    model.fit(train)

    preds = []
    total = 0
    for r in test.iter_ratings():
        total += 1
        if not train.has_user(r.user_id):
            continue   # cold‑start user – cannot predict
        preds.append((model.predict(r.user_id, r.item_id), r.rating))

    elapsed = time.time() - t0
    result = EvaluationResult(rmse(preds), mae(preds), len(preds), total,
                              elapsed_seconds=elapsed)

    if verbose:
        print(f"\n  RMSE        : {result.rmse_value:.4f}")
        print(f"  MAE         : {result.mae_value:.4f}")
        print(f"  Predictions : {result.num_predictions}/{result.num_total}")
        print(f"  Time        : {elapsed:.2f}s")
    return result

def evaluate_leave_one_out(matrix: RatingMatrix, method: str = "item",
                           similarity: str = "cosine", min_common: int = 2,
                           neighbour_k: int = 20, seed: int = DEFAULT_RANDOM_SEED,
                           verbose: bool = True) -> EvaluationResult:
    """
    Evaluate using Leave‑One‑Out cross‑validation.

    For each user with at least 2 ratings, one rating is withheld and predicted
    by the model trained on all other ratings. RMSE and MAE are computed over all
    withheld predictions.

    Parameters:
        matrix: full dataset
        method: "item" or "user"
        similarity: "cosine" or "pearson"
        min_common: minimum co‑rated items/users for similarity
        neighbour_k: neighbourhood size
        seed: random seed for withheld rating selection
        verbose: if True, print progress information

    Returns:
        EvaluationResult
    """
    if verbose:
        print("\n" + "=" * 72)
        print("  Evaluation — Leave‑One‑Out")
        print("=" * 72)

    t0 = time.time()
    splits = leave_one_out_split(matrix, seed)
    if verbose:
        print(f"  LOO splits: {len(splits)}")

    preds = []
    total = len(splits)

    for idx, (train, held) in enumerate(splits):
        model = build_model(method, similarity, min_common, neighbour_k)
        model.fit(train)
        preds.append((model.predict(held.user_id, held.item_id), held.rating))
        if verbose and (idx + 1) % max(1, total // 10) == 0:
            print(f"\r  Progress: {(idx+1)/total:.0%}", end="")
    if verbose:
        print()

    elapsed = time.time() - t0
    result = EvaluationResult(rmse(preds), mae(preds), len(preds), total,
                              elapsed_seconds=elapsed)
    if verbose:
        print(f"\n  RMSE        : {result.rmse_value:.4f}")
        print(f"  MAE         : {result.mae_value:.4f}")
        print(f"  Predictions : {result.num_predictions}/{result.num_total}")
        print(f"  Time        : {elapsed:.2f}s")
    return result