from keras.layers import Embedding, Lambda, TimeDistributed
from keras import backend as K

#
# EmbeddingWithDefault
#
# Wrapper over keras Embedding layer that maps all out of range index to an
# default embedding. Internally, default embedding has index `input_dim`
#
# Input: (batch, input_dim)
#
# Output: (batch, input_dim, output_dim)
#
# Usage:
#
# layer = EmbeddingWithDefault(1000, 10)(prev_layer)
#
# [0, 1, 998, 999, 9999, -1] -> [emb[0], emb[1], emb[998], emb[999], emb[1000], emb[1000]]
#
#
# class EmbeddingWithDefault(Embedding):
#     def __init__(self, input_dim, output_dim, **kwargs):
#         super().__init__(input_dim+1, output_dim, **kwargs)
#
#     def call(self, inputs):
#         def index_map(index):
#             return idx if 0 <= idx and idx < input_dim else input_dim
#
#         mapped_indices = K.map_fn(index_map, inputs)
#         return super().__init__(mapped_indices)

#
# MappedEmbedding
#
# Wrapper over keras Embedding layer that creates an embedding for index in
# `index_set`, for index out of `index_set`, they will be mapped to an default
# embedding.
#
# Internally, there will be `len(index_set)+1` embeddings.
#
# layer = MappedEmbedding({2, 3, 5}, 10)(prev_layer)
#
# [0,1,2,3,4,5,6] -> [emb_default, emb_default, emb_2, emb_3, emb_default, emb_default, emb_5, emb_default]
#
class MappedEmbedding(Embedding):
    def __init__(self, index_set, output_dim, **kwargs):
        self.index_map = {idx: i for i, idx in enumerate(index_set)}
        super(MappedEmbedding, self).__init__(len(self.index_map)+1, output_dim, **kwargs)


    def call(self, inputs):
        mapped_inputs = K.map_fn(lambda inp:
            K.map_fn(lambda idx:
                self.index_map[idx] if idx in self.index_map else len(self.index_map),
                inp
            ),
            inputs)
        return super(MappedEmbedding, self).call(mapped_inputs)
