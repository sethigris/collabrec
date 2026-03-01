# Item‑based collaborative filtering recommender.

import itertools
from typing import Dict, List, Tuple
from collabrec.data import RatingMatrix
from collabrec.data.exceptions import InsufficientDataError
from collabrec.constants import DEFAULT_SIMILARITY, DEFAULT_MIN_COMMON, DEFAULT_NEIGHBOUR_K, DEFAULT_TOP_N
from collabrec.similarity import adjusted_cosine_similarity
from collabrec.models.base import BaseRecommender, Recommendation



class ItemBasedCF(BaseRecommender):
    """
    Item‑Based Collaborative Filtering Recommender.

    How it works:
        1. For each pair of items, compute their similarity using all users who rated both.
        2. Store for each item a list of its most similar neighbours.
        3. To predict user u's rating for item i:
           - Find the k items most similar to i that u has already rated.
           - Compute a weighted average of u's ratings on those items, weighted by similarity.
        4. To recommend for user u:
           - Enumerate all items u has NOT rated.
           - Predict a rating for each.
           - Return the top‑N highest predicted.

    An extra option is available:
        - use_adjusted_cosine: if True and similarity="cosine", use adjusted cosine (subtract user mean) instead of raw cosine.
    """

    def __init__(self, similarity: str = DEFAULT_SIMILARITY,
                 min_common: int = DEFAULT_MIN_COMMON,
                 neighbour_k: int = DEFAULT_NEIGHBOUR_K,
                 use_adjusted_cosine: bool = False) -> None:
        """
        Initialise an item‑based CF model.

        Parameters:
            similarity: "cosine" or "pearson"
            min_common: minimum number of users who must have rated both items for the similarity to be considered valid
            neighbour_k: number of similar items to consider during prediction (K)
            use_adjusted_cosine: only relevant when similarity="cosine"; if True, uses adjusted cosine (subtracts user mean)
        """
        super().__init__(similarity, min_common, neighbour_k)
        self.use_adjusted_cosine = use_adjusted_cosine
        self._item_similarities: Dict[str, List[Tuple[str, float]]] = {}

    def fit(self, matrix: RatingMatrix) -> "ItemBasedCF":
        """
        Pre‑compute all pairwise item similarities and store sorted neighbour lists.

        The time complexity is O(I^2 * U) where I is the number of items and U is the number of users.
        For large datasets, approximate methods would be needed, but this implementation is exact.

        Parameters:
            matrix: training data

        Returns:
            self
        """
        self._matrix = matrix
        self._sim_cache.clear()
        self._item_similarities = {}

        items = matrix.items
        # Pre‑fetch item vectors for faster access
        item_vectors = {iid: matrix.get_item_ratings(iid) for iid in items}

        # Compute pairwise similarities
        for a, b in itertools.combinations(items, 2):
            if self.use_adjusted_cosine and self.similarity_name == "cosine":
                sim = adjusted_cosine_similarity(a, b, matrix, self.min_common)
            else:
                sim = self.sim_fn(item_vectors[a], item_vectors[b], self.min_common)
            if sim != 0.0:   # store only non‑zero similarities to save space
                self._sim_cache.set(a, b, sim)

        # Build sorted neighbour lists for each item
        for iid in items:
            neighbours = []
            for other in items:
                if other == iid:
                    continue
                s = self._sim_cache.get(iid, other)
                if s is not None and s > 0.0:
                    neighbours.append((other, s))
            # Sort: highest similarity first, break ties by item id for determinism
            neighbours.sort(key=lambda x: (-x[1], x[0]))
            self._item_similarities[iid] = neighbours

        self._is_fitted = True
        return self

    def predict(self, user_id: str, item_id: str) -> float:
        """
        Predict the rating using a weighted average of the k most similar items that the user has rated.

        If no similar rated items are found, fallback to the user's mean rating, then the global mean, then 0.0.

        Parameters:
            user_id: target user
            item_id: target item

        Returns:
            predicted rating
        """
        self._check_fitted()
        fallback = (self._matrix.user_mean(user_id) or self._matrix.global_mean() or 0.0)

        # If either the user or the item is unknown, a personalised prediction is not possible.
        if not self._matrix.has_item(item_id) or not self._matrix.has_user(user_id):
            return fallback

        user_ratings = self._matrix.get_user_ratings(user_id)
        neighbours = self._item_similarities.get(item_id, [])

        weighted = 0.0
        sim_sum = 0.0
        count = 0
        for nid, sim in neighbours:
            if count >= self.neighbour_k:
                break
            if nid in user_ratings:
                weighted += sim * user_ratings[nid]
                sim_sum += abs(sim)
                count += 1

        if sim_sum == 0.0 or count == 0:
            return fallback
        return weighted / sim_sum

    def recommend(self, user_id: str, top_n: int = DEFAULT_TOP_N,
                  exclude_seen: bool = True) -> List[Recommendation]:
        """
        Generate top‑N recommendations for a user.

        Steps:
            - Identify candidate items (all items minus those already rated, if exclude_seen is True)
            - Predict a rating for each candidate
            - Compute confidence as the fraction of the k neighbours that were actually used
            - Rank and return the top_n

        Parameters:
            user_id: target user
            top_n: number of recommendations
            exclude_seen: if True, exclude items the user has already rated

        Returns:
            list of Recommendation

        Raises:
            UserNotFoundError if the user is unknown
            InsufficientDataError if there are no candidate items or no predictions could be made
        """
        self._check_fitted()
        self._check_user(user_id)

        seen = set(self._matrix.get_user_ratings(user_id).keys())
        if exclude_seen:
            candidates = [i for i in self._matrix.items if i not in seen]
        else:
            candidates = self._matrix.items

        if not candidates:
            raise InsufficientDataError(f"User {user_id!r} has rated all items; nothing to recommend.")

        scored = []
        for iid in candidates:
            score = self.predict(user_id, iid)
            neighbours = self._item_similarities.get(iid, [])
            useful = sum(1 for nid, _ in neighbours[:self.neighbour_k] if nid in seen)
            conf = useful / self.neighbour_k if self.neighbour_k else 0.0
            scored.append((iid, score, conf))

        if not scored:
            raise InsufficientDataError(f"No scorable items for user {user_id!r}.")
        return self._rank_recommendations(scored, top_n)

    def get_similar_items(self, item_id: str, top_n: int = DEFAULT_TOP_N) -> List[Tuple[str, float]]:
        """
        Return the top‑N items most similar to item_id.

        Parameters:
            item_id: target item
            top_n: number of neighbours

        Returns:
            list of (neighbour_item_id, similarity)
        """
        self._check_fitted()
        return self._item_similarities.get(item_id, [])[:top_n]

    def get_similar_users(self, user_id: str, top_n: int = DEFAULT_TOP_N) -> List[Tuple[str, float]]:
        """Not supported by item‑based CF."""
        raise NotImplementedError("ItemBasedCF does not compute user similarities.")