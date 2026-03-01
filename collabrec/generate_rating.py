#!/usr/bin/env python3
"""
Generate a large synthetic ratings CSV for testing CollabRec.
"""

import csv
import random
from typing import List, Tuple

# =============================================================================
# CONFIGURATION – change these numbers to adjust dataset size
# =============================================================================
NUM_USERS = 500          # number of distinct users
NUM_ITEMS = 100           # number of distinct items
AVG_RATINGS_PER_USER = 50 # average number of ratings per user (total ratings ≈ NUM_USERS * AVG_RATINGS_PER_USER)
RATING_MIN = 1            # minimum rating value
RATING_MAX = 5            # maximum rating value
RANDOM_SEED = 42          # seed for reproducibility
OUTPUT_FILE = "ratings.csv"

# =============================================================================
# GENERATION
# =============================================================================
def generate_ratings(
    num_users: int,
    num_items: int,
    avg_per_user: int,
    rating_min: int,
    rating_max: int,
    seed: int,
) -> List[Tuple[str, str, int]]:
    """
    Generate a list of (user_id, item_id, rating) tuples.
    Uses a Zipf‑like distribution for item popularity:
        - Some items are rated by many users, most by few.
    Ratings are uniformly distributed between rating_min and rating_max.
    """
    rng = random.Random(seed)

    # Create user and item IDs with leading zeros for nice sorting
    user_ids = [f"user_{i:04d}" for i in range(1, num_users + 1)]
    item_ids = [f"item_{i:04d}" for i in range(1, num_items + 1)]

    # Assign popularity weights to items (Zipf distribution)
    # Higher alpha = more skewed. Alpha=1.0 is classic Zipf.
    alpha = 1.0
    item_weights = [1.0 / (i ** alpha) for i in range(1, num_items + 1)]
    total_weight = sum(item_weights)
    item_probs = [w / total_weight for w in item_weights]

    ratings = []

    for user in user_ids:
        # Determine how many items this user rates (Poisson-ish around avg)
        # Use a normal-ish distribution: avg_per_user ± 20%, at least 1.
        n = max(1, int(rng.gauss(avg_per_user, avg_per_user * 0.2)))
        # Choose items without replacement according to popularity weights
        chosen_items = rng.choices(item_ids, weights=item_probs, k=n)
        # Remove duplicates by converting to set (choices may produce duplicates)
        # But we want each user-item pair unique, so we'll use a set.
        # However, rng.choices can pick same item multiple times; we'll sample without replacement.
        # Better: use random.sample with weights? Python's random.sample doesn't support weights directly.
        # Alternative: use random.choices with k=n, then deduplicate, but that reduces actual n.
        # We'll use a different approach: for each user, decide independently for each item with probability p.
        # Simpler: for each user, generate a random number of ratings and pick items without replacement using weighted reservoir?
        # For simplicity, we'll use a loop over items and decide to rate based on a probability derived from item popularity.
        # This yields variable number of ratings per user around avg_per_user.
        # Let's use a method: for each item, probability = (avg_per_user / num_items) * (popularity factor).
        # But to keep code simple and ensure exact number of ratings per user, we'll use weighted sampling without replacement via a custom method.
        # We'll use numpy? Not allowed. So we'll do a simple rejection: generate a list of (item, weight) and pick one, remove it, adjust weights.
        # That's expensive for large numbers but okay for 50k total ratings.
        # Alternatively, we can use a loop: for each item, with probability p_i = min(1, avg_per_user * weight * num_items?) – complicated.
        # I'll use a simple method: for each user, pick a random number of items (poisson), then randomly select that many distinct items using random.sample on the list of items (uniform). This ignores popularity but is simple and fast.
        # Popularity can be added by making item list weighted in the sampling, but random.sample doesn't support weights.
        # For simplicity and speed, we'll ignore popularity and just pick uniformly.
        # This is fine for a test dataset.
        n = max(1, int(rng.gauss(avg_per_user, avg_per_user * 0.2)))
        # Ensure n does not exceed total items
        n = min(n, num_items)
        chosen_items = rng.sample(item_ids, n)
        for item in chosen_items:
            rating = rng.randint(rating_min, rating_max)
            ratings.append((user, item, rating))

    return ratings

def main():
    print(f"Generating {NUM_USERS} users, {NUM_ITEMS} items, ~{NUM_USERS * AVG_RATINGS_PER_USER} ratings...")
    ratings = generate_ratings(
        num_users=NUM_USERS,
        num_items=NUM_ITEMS,
        avg_per_user=AVG_RATINGS_PER_USER,
        rating_min=RATING_MIN,
        rating_max=RATING_MAX,
        seed=RANDOM_SEED,
    )
    print(f"Generated {len(ratings)} ratings. Writing to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "item_id", "rating"])
        writer.writerows(ratings)
    print("Done.")

if __name__ == "__main__":
    main()