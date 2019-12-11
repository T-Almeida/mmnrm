import tensorflow as tf
from tensorflow.keras import backend as K

"""
All layers of this module produce an interaction matrix from a query and a sentence 
"""

class MatrixInteraction(tf.keras.layers.Layer):
    
    def __init__(self, dtype="float32", **kwargs):
        super(MatrixInteraction, self).__init__(dtype=dtype, **kwargs)
        
    def build(self, input_shape):
        # capture the query and document sizes because they are fixed
        
        self.query_max_elements = input_shape[0][1]
        self.setence_max_elements = input_shape[1][1]
        
        super(MatrixInteraction, self).build(input_shape)
        
    def _query_sentence_vector_to_matrices(self, query_vector, sentence_vector):
        """
        Auxiliar function that will replicate each vector in order to procude a matrix with size [B,Q,S]
        where B is batch dimension, Q max terms in the query and S max terms in the sentence
        """
        query_matrix = K.expand_dims(query_vector, axis=2)
        query_matrix = K.repeat_elements(query_matrix, self.setence_max_elements, axis=2)
        
        sentence_matrix = K.expand_dims(sentence_vector, axis=1)
        sentence_matrix = K.repeat_elements(sentence_matrix, self.query_max_elements, axis=1)
        
        return query_matrix, sentence_matrix

class MatrixInteractionMasking(MatrixInteraction):
    """
    Class that handle the computation of the interaction matrix mask 
    """
    
    def __init__(self, mask_value=0, **kwargs):
        super(MatrixInteractionMasking, self).__init__(**kwargs)
        self.mask_value = mask_value
    
    def compute_mask(self, x, mask=None):
        
        query_mask = K.cast(K.not_equal(x[0], self.mask_value), dtype=self.dtype)
        document_mask = K.cast(K.not_equal(x[1], self.mask_value), dtype=self.dtype)
        
        return tf.einsum("bq,bd->bqd", query_mask, document_mask)
    
    
class ExactInteractions(MatrixInteractionMasking): 
    
    def __init__(self, **kwargs):
        super(ExactInteractions, self).__init__(**kwargs)
    
    def call(self, x):
        """
        x[0] - padded query tokens id's
        x[1] - padded sentence tokens id's
        if use_term_importace
            x[2] - query vector with term importances (TF-IDF)
            x[3] - sentence vector with term importances (TF-IDF)
        """
        
        assert(len(x) in [2,4]) # sanity check
        
        mask = self.compute_mask(x)
        
        query_matrix, sentence_matrix = self._query_sentence_vector_to_matrices(x[0], x[1])
        
        interaction_matrix = K.cast(K.equal(query_matrix, sentence_matrix), dtype=self.dtype) * mask
        
        if len(x)==4:
            query_importance_matrix, sentence_importance_matrix = self._query_sentence_vector_to_matrices(x[2], x[3])
            
            query_importance_matrix = query_importance_matrix * mask
            sentence_importance_matrix = sentence_importance_matrix * mask
            
            interaction_matrix = K.concatenate([K.expand_dims(interaction_matrix),
                                                K.expand_dims(query_importance_matrix),
                                                K.expand_dims(sentence_importance_matrix)])
        
        return interaction_matrix
    

class SemanticInteractions(MatrixInteractionMasking):
    
    def __init__(self, 
                 embedding_matrix,
                 learn_term_weights=True,
                 initializer='glorot_uniform',
                 regularizer=None,
                 **kwargs):
        
        super(SemanticInteractions, self).__init__(**kwargs)
        # embedding layer as sugested in https://github.com/tensorflow/tensorflow/issues/31086
        # instead of use keras.Embedding
        self.embeddings = tf.constant(embedding_matrix, dtype=self.dtype)
        self.embedding_dim = embedding_matrix.shape[1]
        self.learn_term_weights = learn_term_weights
        self.initializer = initializer
        self.regularizer = regularizer
        print("[EMBEDDING MATRIX SHAPE]", embedding_matrix.shape)
    
    def build(self, input_shape):
        
        # add some weight that will learn term importance projection of the query and sentence embeddings
        if self.learn_term_weights:
            self.query_w = self.add_weight(name="query_w",
                                           shape=(self.embedding_dim, 1),
                                           initializer=self.initializer,
                                           regularizer=self.regularizer,
                                           trainable=True)
            
            self.sentence_w = self.add_weight(name="sentence_w",
                                              shape=(self.embedding_dim, 1),
                                              initializer=self.initializer,
                                              regularizer=self.regularizer,
                                              trainable=True)
        
        super(SemanticInteractions, self).build(input_shape) 
    
    def call(self, x):
        
        """
        x[0] - padded query tokens id's
        x[1] - padded sentence tokens id's
        """
        
        mask = self.compute_mask(x)
        
        # embbed the tokens
        query_embeddings = tf.nn.embedding_lookup(self.embeddings, x[0])
        sentence_embeddings = tf.nn.embedding_lookup(self.embeddings, x[1])
        
        interaction_matrix = tf.einsum("bqe,bde->bqd", query_embeddings, sentence_embeddings) * mask
        
        if self.learn_term_weights:
            mask = K.expand_dims(mask)
            
            query_projection = K.dot(query_embeddings, self.query_w)
            sentence_projection = K.dot(sentence_embeddings, self.sentence_w)
            
            query_projection_matrix, sentence_projection_matrix = self._query_sentence_vector_to_matrices(query_projection, sentence_projection) 
            
            query_projection_matrix = query_projection_matrix * mask
            sentence_projection_matrix = sentence_projection_matrix * mask
            
            interaction_matrix = K.expand_dims(interaction_matrix)
            interaction_matrix = K.concatenate([interaction_matrix, query_projection_matrix, sentence_projection_matrix])
        
        return interaction_matrix

class ContextedSemanticInteractions(tf.keras.layers.Layer):
    pass