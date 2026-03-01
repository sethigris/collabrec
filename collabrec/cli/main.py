# Main CLI entry point – parses arguments and dispatches to the appropriate functions.

import sys
import time
from typing import Optional, List

from collabrec.constants import DEFAULT_METHOD, DEFAULT_SIMILARITY
from collabrec.data import load_ratings_csv, generate_sample_csv, UserNotFoundError, InsufficientDataError
from collabrec.models import build_model, ItemBasedCF, UserBasedCF
from collabrec.evaluation import evaluate_random_split, evaluate_leave_one_out, DatasetStats
from collabrec.cli.parser import build_arg_parser
from collabrec.cli.printers import print_recommendations, print_similar_items, print_similar_users, print_evaluation
from collabrec.cli.export import export_recommendations_json, export_evaluation_json



def main(argv: Optional[List[str]] = None) -> int:
    """
    Main function – parses arguments and executes the requested command.

    Returns:
        exit code (0 on success, 1 on error)
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # ----------------------------------------------------------------------
    # Generate sample CSV and exit
    # ----------------------------------------------------------------------
    if args.generate_sample:
        generate_sample_csv(args.generate_sample)
        print(f"Sample CSV written to: {args.generate_sample}")
        return 0

    # ----------------------------------------------------------------------
    # Load data (required for all other modes)
    # ----------------------------------------------------------------------
    if not args.data:
        print("ERROR: --data is required. Use --generate-sample to create test data.", file=sys.stderr)
        return 1

    try:
        matrix = load_ratings_csv(args.data, args.delimiter, args.encoding)
    except Exception as e:
        print(f"ERROR loading data: {e}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"  Loaded {matrix.num_ratings} ratings | {matrix.num_users} users | {matrix.num_items} items")

    # ----------------------------------------------------------------------
    # Statistics mode
    # ----------------------------------------------------------------------
    if args.stats:
        DatasetStats(matrix).print_summary()
        return 0

    # ----------------------------------------------------------------------
    # Evaluation mode
    # ----------------------------------------------------------------------
    if args.eval:
        if args.eval_mode == "loo":
            result = evaluate_leave_one_out(
                matrix, method=args.method, similarity=args.similarity,
                min_common=args.min_common, neighbour_k=args.k,
                seed=args.seed, verbose=not args.quiet
            )
        else:
            result = evaluate_random_split(
                matrix, method=args.method, similarity=args.similarity,
                min_common=args.min_common, neighbour_k=args.k,
                train_fraction=args.eval_split, seed=args.seed,
                verbose=not args.quiet
            )
        if not args.quiet:
            print_evaluation(result)
        if args.json:
            export_evaluation_json(result, args.eval_mode, args.method, args.similarity, args.json)
            print(f"  Results exported to: {args.json}")
        return 0

    # ----------------------------------------------------------------------
    # Fit model (for other modes)
    # ----------------------------------------------------------------------
    model = build_model(
        method=args.method, similarity=args.similarity,
        min_common=args.min_common, neighbour_k=args.k,
        use_adjusted_cosine=args.adjusted_cosine
    )
    t0 = time.time()
    model.fit(matrix)
    elapsed = time.time() - t0
    if not args.quiet:
        print(f"  Model trained in {elapsed:.3f}s | {model}")

    # ----------------------------------------------------------------------
    # Single prediction mode
    # ----------------------------------------------------------------------
    if args.predict:
        uid, iid = args.predict
        try:
            score = model.predict(uid, iid)
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
        print(f"\n  Predicted rating for user={uid!r}, item={iid!r}: {score:.4f}")
        return 0

    # ----------------------------------------------------------------------
    # Similar items mode (may need to switch to item‑based model)
    # ----------------------------------------------------------------------
    if args.similar_items:
        if args.method != "item":
            print("WARNING: --similar-items is most useful with --method item. Switching to item‑based CF.",
                  file=sys.stderr)
            model = ItemBasedCF(similarity=args.similarity,
                                min_common=args.min_common,
                                neighbour_k=args.k,
                                use_adjusted_cosine=args.adjusted_cosine)
            model.fit(matrix)
        try:
            neighbours = model.get_similar_items(args.similar_items, top_n=args.top)
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
        if not neighbours:
            print(f"No similar items found for {args.similar_items!r}. Try lowering --min-common.")
            return 0
        print_similar_items(args.similar_items, neighbours)
        return 0

    # ----------------------------------------------------------------------
    # Similar users mode (may need to switch to user‑based model)
    # ----------------------------------------------------------------------
    if args.similar_users:
        if args.method != "user":
            print("WARNING: --similar-users is most useful with --method user. Switching to user‑based CF.",
                  file=sys.stderr)
            model = UserBasedCF(similarity=args.similarity,
                                min_common=args.min_common,
                                neighbour_k=args.k)
            model.fit(matrix)
        try:
            neighbours = model.get_similar_users(args.similar_users, top_n=args.top)
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
        if not neighbours:
            print(f"No similar users found for {args.similar_users!r}. Try lowering --min-common.")
            return 0
        print_similar_users(args.similar_users, neighbours)
        return 0

    # ----------------------------------------------------------------------
    # Default: generate recommendations for a user
    # ----------------------------------------------------------------------
    if not args.user:
        parser.print_help()
        return 0

    try:
        recs = model.recommend(args.user, top_n=args.top, exclude_seen=not args.include_seen)
    except (UserNotFoundError, InsufficientDataError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    if not args.quiet:
        print_recommendations(args.user, recs, args.method, args.similarity)

    if args.json:
        export_recommendations_json(args.user, recs, args.method, args.similarity, args.json)
        if not args.quiet:
            print(f"  Recommendations exported to: {args.json}")

    return 0

if __name__ == "__main__":
    sys.exit(main())