import tensorflow as tf
from tensorflow.keras import backend as K
from mmnrm.layers.base import TrainableLayer


class WeightedCombination(TrainableLayer):
        
    def build(self, input_shape):
        num_of_tensors = int(input_shape[-1])
        
        self.linear_weights = self.add_weight(name="linear_weights",
                                               shape=(1, num_of_tensors),
                                               initializer=self.initializer,
                                               regularizer=self.regularizer,
                                               trainable=True)
        
        super(WeightedCombination, self).build(input_shape) 
        
    def call(self, x):
        return K.sum(x * K.softmax(self.linear_weights, axis=-1), axis=-1)
    
    def compute_mask(self, inputs, mask=None):
        return None # clear the mask after the combination
    

class KmaxAggregation(tf.keras.layers.Layer):
    
    def __init__(self, k, **kwargs):
        super(KmaxAggregation, self).__init__(**kwargs)
        self.k=k
        
    def build(self, input_shape):
        self.dim = int(input_shape[-1])
        
        super(KmaxAggregation, self).build(input_shape) 
    
    def call(self, x): # B, P, D
        x = tf.linalg.matrix_transpose(x) # B, D, P

        top_k, _ = tf.math.top_k(x, k=self.k) # B, D, K
       
        x = tf.reshape(top_k, shape=(-1, self.k*self.dim))

        return x
    
    def compute_mask(self, inputs, mask=None):
        return None # clear the mask after the combination
    
class SelfAttention(TrainableLayer):
    
    def __init__(self, attention_dimension=None, aggregation=True, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.attention_dimension = attention_dimension
        self.aggregation = aggregation
    
    def build(self, input_shape):
        
        emb_dim = int(input_shape[2])
        if self.attention_dimension is None:
            self.attention_dimension = emb_dim

        self.W_attn_project = self.add_weight(name="self_attention_projection",
                                                shape=[emb_dim, self.attention_dimension],
                                                initializer=self.initializer,
                                                regularizer=self.regularizer,)

        self.W_attn_score = self.add_weight(name="self_attention_score",
                                              shape=[self.attention_dimension, 1],
                                              initializer=self.initializer,
                                              regularizer=self.regularizer,)
        
        super(SelfAttention, self).build(input_shape) 
        
    def call(self, x, mask=None):
        
        x_projection = K.dot(x, self.W_attn_project)
        x_tanh = K.tanh(x_projection)
        x_attention = K.dot(x_tanh, self.W_attn_score) # B, Q, 1
        #print("x_attention", x_attention)
        if mask is not None:
            mask = K.expand_dims(mask)
            #print("mask", mask)
            x_attention = x_attention + ((1.0 - K.cast(mask, dtype=self.dtype)) * -10000.0)
        #print("x_attention_maked", x_attention_maked)
        x_attention_softmax = K.softmax(x_attention, axis = 1)
        #print("x_attention_softmax", x_attention_softmax)
        
        x = x_attention_softmax * x
        
        if self.aggregation:
            return K.sum(x, axis = 1)
        else:
            return x
    
    def compute_mask(self, inputs, mask=None):
        return None #clear the mask

class RareWordFreqCombine(TrainableLayer):
    
    def __init__(self, threshold, **kwargs):
        super(RareWordFreqCombine, self).__init__(**kwargs)
        self.threshold = threshold
        
    def build(self, input_shape):
        self.w = self.add_weight(shape=[1,],
                                 initializer=self.initializer,
                                 regularizer=self.regularizer,)

        
        super(RareWordFreqCombine, self).build(input_shape) 
    
    def call(self, x, mask=None):
        """
        x[0] - score_1
        x[1] - score_2
        x[2] - query tokens ids
        """
        
        score_1 = x[0]   # exact
        score_2 = x[1]   # semantic
        query_ids = x[2] # query_ids
        
        filter_threshold = tf.cast(query_ids<self.threshold, dtype = tf.float32)
        num_rare_words = K.sum(filter_threshold)
        
        weight = K.sigmoid(num_rare_words*self.w)
        
        return weight*x[0] + (1-weight)*x[1]

class TermAggregation(TrainableLayer):
    
    def __init__(self, aggregate=True, **kwargs):
        super(TermAggregation, self).__init__(**kwargs)
        self.aggregate = aggregate
        self.distribution_tensor = None
        
    def build(self, input_shape):
        
        q_embeddings = int(input_shape[1][-1]) # q terms embeddings 
        
        self.w_q_embeddings = self.add_weight(shape=[q_embeddings, 1],
                                 initializer=self.initializer,
                                 regularizer=self.regularizer,)
        
        super(TermAggregation, self).build(input_shape) 
    
    def call(self, x, mask=None):
        """
        x[0] - q term vectors
        x[1] - q terms embeddings 
        """

        q_weights = K.squeeze(K.dot(x[1], self.w_q_embeddings), axis=-1) # [None, q, 1]
        self.distribution_tensor = K.softmax(q_weights)
        q_distribution = K.expand_dims(self.distribution_tensor)
        
        if self.aggregate:
            return K.sum(x[0] * q_distribution, axis=1)
        else:
            return x[0] * q_distribution
        

class QueryTermWeighting(TrainableLayer):
    
    def build(self, input_shape):
        
        q_embeddings = int(input_shape[-1]) # q terms embeddings 
        
        self.w_q_embeddings = self.add_weight(shape=(q_embeddings,1),
                                 initializer=self.initializer,
                                 regularizer=self.regularizer,)
        
        super(QueryTermWeighting, self).build(input_shape) 
    
    def call(self, x, mask=None):
        """
        x - q terms embeddings 
        """
        x = tf.einsum("bqe,eo->bq", x, self.w_q_embeddings)
        # masking
        x = x + ((1.0 - K.cast(mask, dtype=self.dtype)) * -10000.0)
        return tf.nn.softmax(x, axis=-1)
    
    
class AprioriLayer(TrainableLayer):
    
    def __init__(self):
        super(AprioriLayer, self).__init__()
        
        self.qtw_layer = QueryTermWeighting()
        
    def call(self, x):
        
        query_input, query_matches, query_embeddings = x
        
        ## sentence a priori importance
        query_input_masked = query_input != 0      
        
        ## compute the query term importance
        query_weights = self.qtw_layer(query_embeddings, mask=query_input_masked)
        
        query_matches = tf.cast(query_matches, "float32")

        return tf.einsum("bpq,bq->bp", query_matches, query_weights), query_weights
    

class AprioriLayerWmask(TrainableLayer):
    
    def __init__(self):
        super(AprioriLayerWmask, self).__init__()
        
        self.qtw_layer = QueryTermWeighting()
        
    def call(self, x):
        
        query_embeddings, query_mask, query_matches = x
        
        query_emb = query_embeddings * tf.expand_dims(query_mask, axis=-1)

        query_weights = self.qtw_layer(query_emb, mask=query_mask)

        return tf.reduce_sum(query_weights * query_matches, axis=-1, keepdims=True)
    
    
    
class MatchesLayer(TrainableLayer):
    
    def __init__(self, match_threshold=None):
        super(MatchesLayer, self).__init__()
        self.match_threshold = match_threshold
        
    def build(self, input_shape):

        if self.match_threshold is None:
            self.match_threshold = self.add_weight(shape=(1,),
                                     initializer=self.initializer,
                                     regularizer=self.regularizer,)
        
        super(MatchesLayer, self).build(input_shape) 
    
    def call(self, x):
        
        threshold = tf.math.sigmoid(self.match_threshold)
        
        matches = tf.cast(tf.squeeze(x, axis=-1)>=threshold, tf.int8)
        
        query_matches = tf.cast(tf.reduce_sum(matches, axis=-1)>0,tf.int8)


        return query_matches