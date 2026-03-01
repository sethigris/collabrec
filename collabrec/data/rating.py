# This file contains the main classes that store ratings in memory.
# It includes a Rating class for a single rating, a RatingMatrix that holds all ratings,
# and a SimilarityCache to remember similarity scores so they do not need to be recomputed.

import collections
import math
from typing import Dict, FrozenSet, Iterator, List, Optional, Set
from collabrec.constants import MISSING

def _mean(values: List[float]) -> float:
    """A helper function to calculate the average of a list of numbers."""
    return sum(values) / len(values)

class Rating:
    """
    This class represents one rating given by one user to one item.
    It contains three pieces of information:
        - user_id: the identifier of the user (extra spaces are removed)
        - item_id: the identifier of the item (extra spaces are removed)
        - rating: the numeric score as a floating point number

    Two ratings are considered equal if they have the same user and item
    and almost the same rating (a small tolerance is used for floating point comparison).
    This allows ratings to be used in sets or as dictionary keys.
    """

    __slots__ = ("user_id", "item_id", "rating")  # Using slots reduces memory usage.

    def __init__(self, user_id: str, item_id: str, rating: float) -> None:
        """When a new Rating is created, the user, item, and rating are stored after cleaning."""
        self.user_id = str(user_id).strip()
        self.item_id = str(item_id).strip()
        self.rating  = float(rating)

    def __repr__(self) -> str:
        """This method controls what is shown when a Rating object is printed."""
        return f"Rating(user={self.user_id!r}, item={self.item_id!r}, r={self.rating})"

    def __eq__(self, other: object) -> bool:
        """Two ratings are equal if they have the same user, same item, and nearly the same rating."""
        if not isinstance(other, Rating):
            return NotImplemented
        return (self.user_id == other.user_id and self.item_id == other.item_id
                and math.isclose(self.rating, other.rating))

    def __hash__(self) -> int:
        """The hash is based on the user and item only, because the pair (user,item) is unique."""
        return hash((self.user_id, self.item_id))


class RatingMatrix:
    """
    This is the main data structure. It is a sparse matrix that stores ratings in two directions:
        - from user to items: to quickly get all ratings of a user
        - from item to users: to quickly get all ratings of an item

    Internally it uses dictionaries of dictionaries. The outer dictionary has user identifiers as keys,
    and the inner dictionary maps item identifiers to ratings. The same rating is also stored in the
    item‑to‑user dictionary.

    All changes (adding or removing a rating) update both dictionaries so they stay synchronized.
    """

    def __init__(self) -> None:
        """Start with empty dictionaries and a counter for the total number of ratings."""
        self._user_item: Dict[str, Dict[str, float]] = collections.defaultdict(dict)
        self._item_user: Dict[str, Dict[str, float]] = collections.defaultdict(dict)
        self._rating_count: int = 0   # This counter avoids having to count every time.

    # ------------------------------------------------------------------
    # Modifying the matrix – adding or removing ratings
    # ------------------------------------------------------------------

    def add_rating(self, user_id: str, item_id: str, rating: float) -> None:
        """
        Add a new rating or replace an existing one.
        The user and item identifiers are stripped of extra spaces, and the rating is converted to float.
        If this (user,item) pair is new, the rating counter is increased.
        Then the rating is stored in both dictionaries.
        """
        user_id = str(user_id).strip()
        item_id = str(item_id).strip()
        rating  = float(rating)
        # Check if this is a new pair
        if user_id not in self._user_item or item_id not in self._user_item[user_id]:
            self._rating_count += 1
        self._user_item[user_id][item_id] = rating
        self._item_user[item_id][user_id] = rating

    def remove_rating(self, user_id: str, item_id: str) -> None:
        """
        If a rating exists, remove it from both dictionaries.
        Decrease the rating counter.
        After removal, if a user or item has no ratings left, the empty dictionary is deleted.
        """
        if user_id in self._user_item and item_id in self._user_item[user_id]:
            del self._user_item[user_id][item_id]
            del self._item_user[item_id][user_id]
            self._rating_count -= 1
            # Clean up empty entries
            if not self._user_item[user_id]:
                del self._user_item[user_id]
            if not self._item_user[item_id]:
                del self._item_user[item_id]

    # ------------------------------------------------------------------
    # Retrieving data – single ratings, all ratings of a user, all ratings of an item
    # ------------------------------------------------------------------

    def get_rating(self, user_id: str, item_id: str) -> Optional[float]:
        """Look up a specific rating. Returns None if it does not exist."""
        return self._user_item.get(user_id, {}).get(item_id, MISSING)

    def get_user_ratings(self, user_id: str) -> Dict[str, float]:
        """Return all ratings made by a user as a dictionary mapping item to rating. If the user is unknown, return an empty dict."""
        return dict(self._user_item.get(user_id, {}))

    def get_item_ratings(self, item_id: str) -> Dict[str, float]:
        """Return all ratings given to an item as a dictionary mapping user to rating. If the item is unknown, return an empty dict."""
        return dict(self._item_user.get(item_id, {}))

    # ------------------------------------------------------------------
    # Membership checks
    # ------------------------------------------------------------------

    def has_user(self, user_id: str) -> bool:
        """Check if a user exists in the matrix."""
        return user_id in self._user_item

    def has_item(self, item_id: str) -> bool:
        """Check if an item exists in the matrix."""
        return item_id in self._item_user

    def has_rating(self, user_id: str, item_id: str) -> bool:
        """Check if a specific (user,item) rating exists."""
        return item_id in self._user_item.get(user_id, {})

    # ------------------------------------------------------------------
    # Iteration – all methods return items in sorted order for reproducibility
    # ------------------------------------------------------------------

    def iter_ratings(self) -> Iterator[Rating]:
        """
        Yield every rating in the matrix.
        Users are sorted alphabetically, and for each user their items are sorted alphabetically.
        This ensures a consistent order.
        """
        for user_id in sorted(self._user_item):
            for item_id in sorted(self._user_item[user_id]):
                yield Rating(user_id, item_id, self._user_item[user_id][item_id])

    def iter_users(self) -> Iterator[str]:
        """Yield all user identifiers in alphabetical order."""
        return iter(sorted(self._user_item.keys()))

    def iter_items(self) -> Iterator[str]:
        """Yield all item identifiers in alphabetical order."""
        return iter(sorted(self._item_user.keys()))

    # ------------------------------------------------------------------
    # Averages – user mean, item mean, global mean
    # ------------------------------------------------------------------

    def user_mean(self, user_id: str) -> Optional[float]:
        """Calculate the average rating of a user. Returns None if the user has no ratings."""
        ratings = list(self._user_item.get(user_id, {}).values())
        return _mean(ratings) if ratings else None

    def item_mean(self, item_id: str) -> Optional[float]:
        """Calculate the average rating of an item. Returns None if the item has no ratings."""
        ratings = list(self._item_user.get(item_id, {}).values())
        return _mean(ratings) if ratings else None

    def global_mean(self) -> Optional[float]:
        """Calculate the overall average of all ratings in the matrix. Returns None if there are no ratings."""
        all_ratings = [r for item_ratings in self._item_user.values() for r in item_ratings.values()]
        return _mean(all_ratings) if all_ratings else None

    # ------------------------------------------------------------------
    # Properties – provide quick summary information
    # ------------------------------------------------------------------

    @property
    def users(self) -> List[str]:
        """Return a sorted list of all user identifiers."""
        return sorted(self._user_item.keys())

    @property
    def items(self) -> List[str]:
        """Return a sorted list of all item identifiers."""
        return sorted(self._item_user.keys())

    @property
    def num_users(self) -> int:
        """The number of distinct users."""
        return len(self._user_item)

    @property
    def num_items(self) -> int:
        """The number of distinct items."""
        return len(self._item_user)

    @property
    def num_ratings(self) -> int:
        """The total number of ratings stored."""
        return self._rating_count

    @property
    def density(self) -> float:
        """
        The density is the fraction of possible user‑item pairs that actually have a rating.
        It is calculated as (number of ratings) divided by (number of users times number of items).
        A low density means the matrix is very sparse, which is common in recommendation systems.
        """
        total = self.num_users * self.num_items
        return self._rating_count / total if total else 0.0

    # ------------------------------------------------------------------
    # Copy and conversion
    # ------------------------------------------------------------------

    def copy(self) -> "RatingMatrix":
        """Create a deep copy of the matrix – a new matrix containing the same ratings."""
        new = RatingMatrix()
        for r in self.iter_ratings():
            new.add_rating(r.user_id, r.item_id, r.rating)
        return new

    def to_list(self) -> List[Rating]:
        """Convert all ratings to a list. The order is the same as iter_ratings()."""
        return list(self.iter_ratings())

    def __len__(self) -> int:
        """Return the number of ratings (same as num_ratings)."""
        return self._rating_count

    def __repr__(self) -> str:
        """A short description showing the number of users, items, ratings, and the density."""
        return (f"RatingMatrix(users={self.num_users}, items={self.num_items}, "
                f"ratings={self.num_ratings}, density={self.density:.4f})")


class SimilarityCache:
    """
    A cache for pairwise similarity scores.
    Calculating similarities can be expensive, so results are stored and reused.

    Each pair is stored only once using a frozenset of the two identifiers as the key.
    This means the same key works for (A,B) and (B,A).
    The cache also tracks hits and misses to measure its effectiveness.
    """

    def __init__(self) -> None:
        """Start with an empty cache and zero hit/miss counters."""
        self._cache: Dict[FrozenSet, float] = {}
        self._hits = 0
        self._misses = 0

    def get(self, id_a: str, id_b: str) -> Optional[float]:
        """
        Retrieve the similarity between id_a and id_b from the cache.
        If the pair is found, a hit is counted and the value is returned.
        If not, a miss is counted and None is returned.
        """
        key = frozenset([id_a, id_b])
        val = self._cache.get(key)
        if val is None:
            self._misses += 1
        else:
            self._hits += 1
        return val

    def set(self, id_a: str, id_b: str, value: float) -> None:
        """Store the similarity value for the pair (id_a, id_b)."""
        self._cache[frozenset([id_a, id_b])] = value

    def clear(self) -> None:
        """Empty the cache and reset the hit and miss counters to zero."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def hit_rate(self) -> float:
        """The proportion of get operations that found a value in the cache (hits divided by total attempts)."""
        total = self._hits + self._misses
        return self._hits / total if total else 0.0

    def __len__(self) -> int:
        """The number of distinct pairs stored in the cache."""
        return len(self._cache)