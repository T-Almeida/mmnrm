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
    
    def compute_mask(self, inputs, mask=None):
        return tf.transpose(tf.split(tf.not_equal(inputs, self.mask_value), self.num_splites, axis=1), perm=[1,0,2])
    
    
class IdentityMask(tf.keras.layers.Layer):
    
    def __init__(self, mask_value = 0., **kwargs):
        super(IdentityMask, self).__init__(**kwargs)
        self.mask_value = mask_value
        
    def call(self, x, mask=None):
        return x
    
    def compute_mask(self, inputs):
        return tf.not_equal(inputs, self.mask_value)
    
    
class MaskedConcatenate(tf.keras.layers.Layer):
    """
    Concatenation of a list of tensor with a custom beahaviour for the mask
    """
    def __init__(self, index_to_keep, **kwargs):
        """
        Corrent behaviour will return the mask of the input that corresponds to the index_to_keep var
        """
        super(MaskedConcatenate, self).__init__(**kwargs)
        self.index_to_keep = index_to_keep
        
    def call(self, x, mask=None):
        return K.concatenate(x)
    
    def compute_mask(self, x, mask=None):
        assert(isinstance(mask, list))
        return mask[self.index_to_keep]

class ResidualContextLSTM(tf.keras.layers.Layer):
    def __init__(self, size, activation="relu", **kwargs):
        super(ResidualContextLSTM, self).__init__(**kwargs)
        self.lstm = tf.keras.layers.LSTM(size, activation=activation, return_sequences=True)
        
    def call(self, x, mask=None):
        context = self.lstm(x, mask=mask)
        return context + x # residual

class ResidualContextBiLSTM(tf.keras.layers.Layer):
    def __init__(self, size, activation="relu", **kwargs):
        super(ResidualContextBiLSTM, self).__init__(**kwargs)
        self.bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(size, activation=activation, return_sequences=True), merge_mode="ave")
        
    def call(self, x, mask=None):
        context = self.bilstm(x, mask=mask)
        return context + x # residual
    
class ShuffleRows(tf.keras.layers.Layer):
    """
    Shuffle a tensor along the row dimension (Batch, Row, Colums)
    """
        
    def _build_indexes(self, x):
        indices_tensor_shape = K.shape(x)[:-1]
        l = tf.range(indices_tensor_shape[1], dtype="int32")
        l = tf.random.shuffle(l)


        rows_dim =  K.expand_dims(indices_tensor_shape[0])

        l=tf.tile(l, rows_dim)
        return K.expand_dims(tf.reshape(l, indices_tensor_shape))
    
    def call(self, x, mask=None):
        """
        x[0] - tensor matrix
        x[1] - indices that will guide the permutation, this should follow the format:
        
                permutation => (R1,R2,R3,R4)
                indices => [[[R1], [R2], [R3], [R4]]] * batch_size,
                
                where R1 R2 R3 R4 are the permutated index of the rows
                
                example:  [[[1],[2],[3],[0]]]*10

        """
        indices = self._build_indexes(x)

        return tf.gather_nd(x, indices, batch_dims=1)

        
    
    def compute_mask(self, x, mask=None):
        return None
        #return tf.gather_nd(mask, x[1], batch_dims=1)
    
