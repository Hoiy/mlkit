from keras.layers import Embedding, Lambda, TimeDistributed
from keras import backend as K
import tensorflow as tf

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
# `embed_indices`, for index out of `embed_indices`, they will be mapped to an default
# embedding.
#
# Internally, there will be `len(embed_indices)+1` embeddings.
#
# layer = MappedEmbedding(embed_indices=[2, 3, 5], output_dim=10)(prev_layer)
#
# [0,1,2,3,4,5,6] -> [emb_default, emb_default, emb_2, emb_3, emb_default, emb_default, emb_5, emb_default]
#
class MappedEmbedding(Embedding):
    def __init__(self, embed_indices, output_dim, **kwargs):
        assert len(set(embed_indices)) == len(embed_indices)
        self.embed_indices = embed_indices
        super(MappedEmbedding, self).__init__(len(embed_indices)+1, output_dim, **kwargs)

    def build(self, input_shape):
        self.table = tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(list(self.embed_indices), list(range(len(self.embed_indices)))),
            len(self.embed_indices)
        )
        sess = K.get_session()
        with sess.as_default():
            self.table.init.run()
        return super(MappedEmbedding, self).build(input_shape)

    def call(self, inputs):
        mapped_inputs = self.table.lookup(inputs)
        return super(MappedEmbedding, self).call(mapped_inputs)

    def get_config(self):
        config = {'embed_indices': list(self.embed_indices)}
        base_config = super(MappedEmbedding, self).get_config()
        del base_config['input_dim']
        return dict(list(base_config.items()) + list(config.items()))
