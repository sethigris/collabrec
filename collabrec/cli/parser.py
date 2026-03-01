# Argument parser for the CLI – defines all command‑line options.


import argparse
import textwrap
from collabrec.constants import DEFAULT_TOP_N, DEFAULT_METHOD, DEFAULT_SIMILARITY, DEFAULT_MIN_COMMON, DEFAULT_NEIGHBOUR_K, DEFAULT_RANDOM_SEED, DEFAULT_EVAL_SPLIT

__version__ = "1.0.0"

def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build and return the argument parser for the CollabRec CLI.

    The parser is organised into groups for data input, recommendation parameters,
    model configuration, evaluation, exploration, and output options.

    Returns:
        argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="recommender",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            CollabRec — Deterministic Collaborative Filtering Recommender
            ─────────────────────────────────────────────────────────────
            Generate personalised item recommendations using user‑based or
            item‑based collaborative filtering with cosine or Pearson
            similarity. Optionally evaluate accuracy (RMSE / MAE)."""),
        epilog=textwrap.dedent("""\
            Examples:
              # Item‑based, top‑5 for user 'alice'
              python -m cli.main --data ratings.csv --user alice --top 5

              # User‑based with Pearson similarity
              python -m cli.main --data ratings.csv --user bob \\
                  --method user --similarity pearson --top 10

              # Show dataset statistics
              python -m cli.main --data ratings.csv --stats

              # Evaluate on an 80/20 split
              python -m cli.main --data ratings.csv --eval --eval-split 0.8

              # Generate a sample CSV for testing
              python -m cli.main --generate-sample sample.csv

              # Find items similar to a given item
              python -m cli.main --data ratings.csv --similar-items inception""")
    )

    # Data group – input file and format
    data_group = parser.add_argument_group("Data")
    data_group.add_argument("--data", "-d", metavar="FILE",
                            help="Path to ratings CSV file (required unless --generate-sample is used)")
    data_group.add_argument("--delimiter", default=",", metavar="CHAR",
                            help="CSV field delimiter (default: ',')")
    data_group.add_argument("--encoding", default="utf-8", metavar="ENC",
                            help="CSV file encoding (default: utf-8)")
    data_group.add_argument("--generate-sample", metavar="FILE",
                            help="Generate a sample ratings CSV and exit")

    # Recommendation group – what to recommend
    rec_group = parser.add_argument_group("Recommendation")
    rec_group.add_argument("--user", "-u", metavar="USER_ID",
                           help="User ID for whom to generate recommendations")
    rec_group.add_argument("--top", "-n", type=int, default=DEFAULT_TOP_N, metavar="N",
                           help=f"Number of recommendations (default: {DEFAULT_TOP_N})")
    rec_group.add_argument("--include-seen", action="store_true",
                           help="Include already‑rated items in recommendations")

    # Model group – algorithm configuration
    model_group = parser.add_argument_group("Model")
    model_group.add_argument("--method", "-m", choices=["item", "user"], default=DEFAULT_METHOD,
                             help=f"CF method: 'item' or 'user' (default: {DEFAULT_METHOD})")
    model_group.add_argument("--similarity", "-s", choices=["cosine", "pearson"], default=DEFAULT_SIMILARITY,
                             help=f"Similarity metric (default: {DEFAULT_SIMILARITY})")
    model_group.add_argument("--k", type=int, default=DEFAULT_NEIGHBOUR_K, metavar="K",
                             help=f"Number of neighbours (default: {DEFAULT_NEIGHBOUR_K})")
    model_group.add_argument("--min-common", type=int, default=DEFAULT_MIN_COMMON, metavar="N",
                             help=f"Min co‑rated items for similarity (default: {DEFAULT_MIN_COMMON})")
    model_group.add_argument("--adjusted-cosine", action="store_true",
                             help="Use adjusted cosine (item‑based only; subtracts user mean)")

    # Evaluation group – performance assessment
    eval_group = parser.add_argument_group("Evaluation")
    eval_group.add_argument("--eval", action="store_true",
                            help="Run evaluation mode (compute RMSE / MAE)")
    eval_group.add_argument("--eval-mode", choices=["split", "loo"], default="split",
                            help="Evaluation strategy: 'split' (random) or 'loo' (leave‑one‑out)")
    eval_group.add_argument("--eval-split", type=float, default=DEFAULT_EVAL_SPLIT, metavar="FRAC",
                            help=f"Train fraction for random split (default: {DEFAULT_EVAL_SPLIT})")
    eval_group.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED, metavar="INT",
                            help=f"Random seed (default: {DEFAULT_RANDOM_SEED})")

    # Exploration group – similarity queries and predictions
    explore_group = parser.add_argument_group("Exploration")
    explore_group.add_argument("--stats", action="store_true",
                               help="Print dataset statistics and exit")
    explore_group.add_argument("--similar-items", metavar="ITEM_ID",
                               help="List items most similar to ITEM_ID (item‑based only)")
    explore_group.add_argument("--similar-users", metavar="USER_ID",
                               help="List users most similar to USER_ID (user‑based only)")
    explore_group.add_argument("--predict", nargs=2, metavar=("USER_ID", "ITEM_ID"),
                               help="Predict a single user‑item rating")

    # Output group – JSON export and verbosity
    out_group = parser.add_argument_group("Output")
    out_group.add_argument("--json", metavar="FILE",
                           help="Export results to a JSON file")
    out_group.add_argument("--quiet", "-q", action="store_true",
                           help="Suppress informational output")
    out_group.add_argument("--version", action="version", version=f"CollabRec {__version__}")

    return parser