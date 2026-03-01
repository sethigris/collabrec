# Pretty‑printing functions for CLI output.

from typing import List, Tuple
from collabrec.models import Recommendation


_TERM_WIDTH = 72   # assumed terminal width of 72 characters

def print_header(title: str) -> None:
    """Print a centred header with surrounding equals signs."""
    print()
    print("=" * _TERM_WIDTH)
    pad = (_TERM_WIDTH - len(title) - 2) // 2
    print(" " * pad + f" {title} ")
    print("=" * _TERM_WIDTH)

def print_subheader(title: str) -> None:
    """Print a sub‑header (used inside a section)."""
    print(f"  ── {title}")

def kv(key: str, value) -> None:
    """Print a key‑value pair with fixed column width."""
    print(f"  {key:<30} {value}")

def print_recommendations(user_id: str, recs: List[Recommendation],
                          method: str, similarity: str) -> None:
    """
    Print a formatted list of recommendations.

    A visual confidence bar of 10 characters is included.
    """
    print_header(f"Top-{len(recs)} Recommendations for '{user_id}'")
    print(f"  Method     : {method}")
    print(f"  Similarity : {similarity}")
    print()
    print("  Rank   Item                           Pred. Score   Confidence")
    print("  " + "─" * 60)
    for rec in recs:
        conf_bar = "█" * int(rec.confidence * 10) + "░" * (10 - int(rec.confidence * 10))
        print(f"  {rec.rank:<6} {rec.item_id:<30} {rec.predicted_score:<14.3f} {conf_bar} {rec.confidence:.0%}")
    print()

def print_similar_items(item_id: str, neighbours: List[Tuple[str, float]]) -> None:
    """Print a list of items similar to the given item."""
    print_header(f"Items Similar to '{item_id}'")
    print("  Rank   Item                           Similarity")
    print("  " + "─" * 50)
    for rank, (nid, sim) in enumerate(neighbours, start=1):
        bar = "█" * int(abs(sim) * 20)   # visual bar up to 20 characters
        print(f"  {rank:<6} {nid:<30} {sim:+.4f} {bar}")
    print()

def print_similar_users(user_id: str, neighbours: List[Tuple[str, float]]) -> None:
    """Print a list of users similar to the given user."""
    print_header(f"Users Similar to '{user_id}'")
    print("  Rank   User                           Similarity")
    print("  " + "─" * 50)
    for rank, (nid, sim) in enumerate(neighbours, start=1):
        bar = "█" * int(abs(sim) * 20)
        print(f"  {rank:<6} {nid:<30} {sim:+.4f} {bar}")
    print()

def print_evaluation(result) -> None:
    """Print evaluation metrics in a tidy format."""
    print_header("Evaluation Results")
    kv("RMSE", f"{result.rmse_value:.4f}")
    kv("MAE", f"{result.mae_value:.4f}")
    kv("Predictions made", result.num_predictions)
    kv("Predictions attempted", result.num_total)
    if result.num_total:
        kv("Prediction rate", f"{result.num_predictions/result.num_total:.1%}")
    if result.coverage_value is not None:
        kv("Coverage", f"{result.coverage_value:.1%}")
    kv("Elapsed", f"{result.elapsed_seconds:.2f}s")
    print()