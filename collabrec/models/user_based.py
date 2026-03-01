# User‑based collaborative filtering recommender.

import itertools
from typing import Dict, List, Tuple
from collabrec.data import RatingMatrix
from collabrec.data.exceptions import InsufficientDataError
from collabrec.constants import DEFAULT_SIMILARITY, DEFAULT_MIN_COMMON, DEFAULT_NEIGHBOUR_K, DEFAULT_TOP_N
from collabrec.models.base import BaseRecommender, Recommendation


class UserBasedCF(BaseRecommender):
    """
    User‑Based Collaborative Filtering Recommender.

    How it works:
        1. For each pair of users, compute their similarity using items they both have rated.
        2. Store for each user a list of its most similar neighbours.
        3. To predict user u's rating for item i:
           - Find the k users most similar to u who have also rated i.
           - Compute a bias‑corrected weighted average:
                predicted = r̄_u + [ Σ sim(u,v) * (r_v,i - r̄_v) ] / Σ |sim(u,v)|
        4. To recommend for user u:
           - Enumerate all items u has NOT rated.
           - Predict a rating for each.
           - Return the top‑N highest predicted.
    """

    def __init__(self, similarity: str = DEFAULT_SIMILARITY,
                 min_common: int = DEFAULT_MIN_COMMON,
                 neighbour_k: int = DEFAULT_NEIGHBOUR_K) -> None:
        """
        Initialise a user‑based CF model.

        Parameters:
            similarity: "cosine" or "pearson"
            min_common: minimum number of items two users must have both rated for their similarity to be considered valid
            neighbour_k: number of similar users to consider during prediction (K)
        """
        super().__init__(similarity, min_common, neighbour_k)
        self._user_similarities: Dict[str, List[Tuple[str, float]]] = {}

    def fit(self, matrix: RatingMatrix) -> "UserBasedCF":
        """
        Pre‑compute all pairwise user similarities and store sorted neighbour lists.

        The time complexity is O(U^2 * I) where U is the number of users and I is the number of items.

        Parameters:
            matrix: training data

        Returns:
            self
        """
        self._matrix = matrix
        self._sim_cache.clear()
        self._user_similarities = {}

        users = matrix.users
        # Pre‑fetch user vectors
        user_vectors = {uid: matrix.get_user_ratings(uid) for uid in users}

        # Compute pairwise similarities
        for a, b in itertools.combinations(users, 2):
            sim = self.sim_fn(user_vectors[a], user_vectors[b], self.min_common)
            if sim != 0.0:
                self._sim_cache.set(a, b, sim)

        # Build sorted neighbour lists for each user
        for uid in users:
            neighbours = []
            for other in users:
                if other == uid:
                    continue
                s = self._sim_cache.get(uid, other)
                if s is not None and s > 0.0:
                    neighbours.append((other, s))
            neighbours.sort(key=lambda x: (-x[1], x[0]))
            self._user_similarities[uid] = neighbours

        self._is_fitted = True
        return self

    def predict(self, user_id: str, item_id: str) -> float:
        """
        Predict the rating using a bias‑corrected weighted average of neighbours.

        If no similar users have rated the item, fallback to the user's mean, then global mean, then 0.0.

        Parameters:
            user_id: target user
            item_id: target item

        Returns:
            predicted rating
        """
        self._check_fitted()
        user_mean = self._matrix.user_mean(user_id)
        fallback = user_mean or self._matrix.global_mean() or 0.0

        if not self._matrix.has_user(user_id):
            return fallback

        user_mean = user_mean or 0.0
        neighbours = self._user_similarities.get(user_id, [])

        weighted = 0.0
        sim_sum = 0.0
        count = 0
        for nid, sim in neighbours:
            if count >= self.neighbour_k:
                break
            nrating = self._matrix.get_rating(nid, item_id)
            if nrating is not None:
                nmean = self._matrix.user_mean(nid) or 0.0
                weighted += sim * (nrating - nmean)
                sim_sum += abs(sim)
                count += 1

        if sim_sum == 0.0 or count == 0:
            return fallback
        return user_mean + (weighted / sim_sum)

    def recommend(self, user_id: str, top_n: int = DEFAULT_TOP_N,
                  exclude_seen: bool = True) -> List[Recommendation]:
        """
        Generate top‑N recommendations for a user.

        Steps:
            - Identify candidate items (unrated)
            - Predict rating for each
            - Confidence = (number of neighbours who rated this item) / neighbour_k
            - Rank and return top_n

        Parameters:
            user_id: target user
            top_n: number of recommendations
            exclude_seen: if True, exclude already‑rated items

        Returns:
            list of Recommendation

        Raises:
            UserNotFoundError if user unknown
            InsufficientDataError if no candidate items or no predictions possible
        """
        self._check_fitted()
        self._check_user(user_id)

        seen = set(self._matrix.get_user_ratings(user_id).keys())
        if exclude_seen:
            candidates = [i for i in self._matrix.items if i not in seen]
        else:
            candidates = self._matrix.items

        if not candidates:
            raise InsufficientDataError(f"User {user_id!r} has rated all items.")

        scored = []
        for iid in candidates:
            score = self.predict(user_id, iid)
            neighbours = self._user_similarities.get(user_id, [])
            useful = sum(1 for nid, _ in neighbours[:self.neighbour_k]
                         if self._matrix.has_rating(nid, iid))
            conf = useful / self.neighbour_k if self.neighbour_k else 0.0
            scored.append((iid, score, conf))

        if not scored:
            raise InsufficientDataError(f"No scorable items for user {user_id!r}.")
        return self._rank_recommendations(scored, top_n)

    def get_similar_users(self, user_id: str, top_n: int = DEFAULT_TOP_N) -> List[Tuple[str, float]]:
        """
        Return the top‑N users most similar to user_id.

        Parameters:
            user_id: target user
            top_n: number of neighbours

        Returns:
            list of (neighbour_user_id, similarity)
        """
        self._check_fitted()
        return self._user_similarities.get(user_id, [])[:top_n]

    def get_similar_items(self, item_id: str, top_n: int = DEFAULT_TOP_N) -> List[Tuple[str, float]]:
        """Not supported by user‑based CF."""
        raise NotImplementedError("UserBasedCF does not compute item similarities.")