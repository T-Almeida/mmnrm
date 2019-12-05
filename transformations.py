import tensorflow as tf
from tensorflow.keras import backend as K

class DocumentFixWindowSplit(tf.keras.layers.Layer):
    
    def __init__(self, window_size = 10, **kwargs):
        super(DocumentWindowSplit, self).__init__(**kwargs)
        self.window_size = window_size
        
    def build(self, input_shape):
        self.num_splites = input_shape[1]//self.window_size
        super(DocumentWindowSplit, self).build(input_shape) 
        
    def call(self, x):
        return tf.transpose(tf.split(x, self.num_splites, axis=1), perm=[1,0,2])