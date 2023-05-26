from typing import List
from coworking import Coworking, CoworkingId
from .recommender import Recommender

class HammingRecommender(Recommender):
    def __init__(self, data: List[Coworking]):
        pass

    def fit(self):
        raise NotImplementedError

    def recommend(self, id: CoworkingId, n=4) -> List[CoworkingId]:
        raise NotImplementedError