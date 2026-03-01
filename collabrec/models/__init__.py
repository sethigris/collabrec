# This package contains the recommendation models.
# It includes a base class, two concrete models (item‑based and user‑based), and a factory to create them.


from collabrec.models.base import BaseRecommender, Recommendation
from collabrec.models.item_based import ItemBasedCF
from collabrec.models.user_based import UserBasedCF
from collabrec.models.factory import build_model