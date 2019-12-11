import tensorflow as tf
from tensorflow.keras import backend as K

class DocumentFixWindowSplit(tf.keras.layers.Layer):
    
    def __init__(self, window_size = 10, mask_value = 0., **kwargs):
        super(DocumentFixWindowSplit, self).__init__(**kwargs)
        self.window_size = window_size
        self.mask_value = mask_value
        
    def build(self, input_shape):
        self.num_splites = input_shape[1]//self.window_size
        super(DocumentFixWindowSplit, self).build(input_shape) 
        
    def call(self, x):
        return tf.transpose(tf.split(x, self.num_splites, axis=1), perm=[1,0,2])
    
    def compute_mask(self, inputs):
        return tf.transpose(tf.split(tf.not_equal(inputs, self.mask_value), self.num_splites, axis=1), perm=[1,0,2])
    
    
class IdentityMask(tf.keras.layers.Layer):
    
    def __init__(self, mask_value = 0., **kwargs):
        super(IdentityMask, self).__init__(**kwargs)
        self.mask_value = mask_value
        
    def call(self, x, mask=None):
        return x
    
    def compute_mask(self, inputs):
        return tf.not_equal(inputs, self.mask_value)