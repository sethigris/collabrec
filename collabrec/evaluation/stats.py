# Dataset statistics – compute and display descriptive statistics of a RatingMatrix.


import collections
import math
from typing import Dict, List, Tuple
from collabrec.data import RatingMatrix


def _mean(values):
    """Arithmetic mean (internal helper)."""
    return sum(values) / len(values)

def _std(values):
    """Population standard deviation (internal helper)."""
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / len(values))

class DatasetStats:
    """
    Compute and display descriptive statistics for a RatingMatrix.

    Provides methods for rating distribution, per‑user and per‑item counts,
    top‑rated items, most active users, and a formatted summary.
    """

    def __init__(self, matrix: RatingMatrix) -> None:
        self.matrix = matrix

    def rating_distribution(self) -> Dict[float, int]:
        """Count how many times each rating value appears."""
        dist = collections.Counter()
        for r in self.matrix.iter_ratings():
            dist[r.rating] += 1
        return dict(sorted(dist.items()))

    def ratings_per_user(self) -> Dict[str, int]:
        """Return a dictionary mapping user_id to the number of ratings they made."""
        return {uid: len(self.matrix.get_user_ratings(uid)) for uid in self.matrix.users}

    def ratings_per_item(self) -> Dict[str, int]:
        """Return a dictionary mapping item_id to the number of ratings it received."""
        return {iid: len(self.matrix.get_item_ratings(iid)) for iid in self.matrix.items}

    def top_rated_items(self, n: int = 10) -> List[Tuple[str, float, int]]:
        """
        Return the top‑n items by average rating. Only items with at least 2 ratings are considered.

        Returns:
            list of (item_id, average_rating, number_of_ratings) sorted by average descending.
        """
        result = []
        for iid in self.matrix.items:
            ratings = list(self.matrix.get_item_ratings(iid).values())
            if len(ratings) >= 2:
                result.append((iid, _mean(ratings), len(ratings)))
        result.sort(key=lambda x: (-x[1], x[0]))
        return result[:n]

    def most_active_users(self, n: int = 10) -> List[Tuple[str, int]]:
        """
        Return the top‑n users by number of ratings.

        Returns:
            list of (user_id, number_of_ratings) sorted descending.
        """
        counts = [(uid, len(self.matrix.get_user_ratings(uid))) for uid in self.matrix.users]
        counts.sort(key=lambda x: (-x[1], x[0]))
        return counts[:n]

    def summary(self) -> Dict:
        """Return a dictionary with overall summary statistics."""
        all_ratings = [r.rating for r in self.matrix.iter_ratings()]
        per_user = list(self.ratings_per_user().values())
        per_item = list(self.ratings_per_item().values())
        return {
            "num_users": self.matrix.num_users,
            "num_items": self.matrix.num_items,
            "num_ratings": self.matrix.num_ratings,
            "density": round(self.matrix.density, 6),
            "global_mean_rating": round(_mean(all_ratings), 4) if all_ratings else None,
            "global_min_rating": min(all_ratings) if all_ratings else None,
            "global_max_rating": max(all_ratings) if all_ratings else None,
            "global_std_rating": round(_std(all_ratings), 4) if all_ratings else None,
            "mean_ratings_per_user": round(_mean(per_user), 2) if per_user else None,
            "mean_ratings_per_item": round(_mean(per_item), 2) if per_item else None,
        }

    def print_summary(self) -> None:
        """Print a formatted summary of the dataset to the console."""
        s = self.summary()
        print("\n" + "=" * 72)
        print("  Dataset Statistics")
        print("=" * 72)
        print(f"  Users                       {s['num_users']}")
        print(f"  Items                       {s['num_items']}")
        print(f"  Ratings                     {s['num_ratings']}")
        print(f"  Density                     {s['density']:.4f}")
        print(f"  Global Mean Rating          {s['global_mean_rating']}")
        print(f"  Global Min Rating           {s['global_min_rating']}")
        print(f"  Global Max Rating           {s['global_max_rating']}")
        print(f"  Global Std Rating           {s['global_std_rating']}")
        print(f"  Mean ratings/user           {s['mean_ratings_per_user']}")
        print(f"  Mean ratings/item           {s['mean_ratings_per_item']}")
        print("\n  ── Rating Distribution")
        dist = self.rating_distribution()
        total = sum(dist.values())
        for val, cnt in sorted(dist.items()):
            bar = "█" * int(cnt / total * 40)
            print(f"    {val:5.1f} | {bar:<40} {cnt:>5} ({cnt/total:.1%})")
        print("\n  ── Most Active Users (top 5)")
        for uid, cnt in self.most_active_users(5):
            print(f"    {uid:<20} {cnt} ratings")
        print("\n  ── Top Rated Items (top 5, min 2 ratings)")
        for iid, avg, cnt in self.top_rated_items(5):
            stars = "★" * round(avg) + "☆" * (5 - round(avg))
            print(f"    {iid:<25} avg={avg:.2f} {stars} ({cnt} ratings)")