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