from typing import List
from coworking import Coworking, CoworkingId
from .recommender import Recommender
import numpy as np

class HammingRecommender(Recommender):
    def __init__(self, data: List[Coworking]):
        # get all tags from all coworkings
        all_tags = [ tag.id for coworking in data for tag in coworking.tags ]
        all_tags = set(all_tags)

        # convert coworkings to vectors
        # if a coworking has a tag, then the corresponding element in the vector is 1, otherwise 0
        def coworking_to_vec(coworking: Coworking):
            vec = []
            for tag in all_tags:
                if tag in coworking.tags:
                    vec.append(1)
                else:
                    vec.append(0)
            return vec
        
        self._data = np.array(list(map(coworking_to_vec, data)))

        # map coworking id to index in self._data
        self._id_to_index = { coworking.id: i for i, coworking in enumerate(data) }
        self._index_to_id = [ coworking.id for coworking in data ]

    def fit(self):
        pass

    def recommend(self, id: CoworkingId, n=4) -> List[CoworkingId]:
        vec = self._data[self._id_to_index[id]]

        # calculate Manhattan distance between vec and all other vectors
        # because vec is a binary vector, Manhattan distance is the same as Hamming distance
        distances = np.sum(np.abs(self._data - vec), axis=1)

        # sort distances in ascending order
        sorted_indices = np.argsort(distances)

        # the first element is the coworking itself, so remove it
        sorted_indices = sorted_indices[1:n+1]

        return [ self._index_to_id[i] for i in sorted_indices ]