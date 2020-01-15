import tensorflow as tf
from tensorflow.keras import backend as K

from mmnrm.layers.interaction import SemanticInteractions, ExactInteractions
from mmnrm.layers.local_relevance import MultipleNgramConvs, MaskedSoftmax
from mmnrm.layers.transformations import MaskedConcatenate, ShuffleRows
from mmnrm.layers.aggregation import WeightedCombination

def build_PACRR(max_q_length,
                max_d_length,
                emb_matrix = None,
                trainable_embeddings = False,
                learn_term_weights = False,
                dense_hidden_units = None,
                max_ngram = 3,
                k_max = 2,
                out_put_dim = 1,
                shuffle_query_terms = False, 
                k_polling_avg = None, # do k_polling avg after convolution
                polling_avg = False, # do avg polling after convolution
                use_mask = True,
                filters=32, # can be a list or a function of input features and n-gram
                name_model = None,
                **kwargs):
    
    prefix_name = ""
    
    # init layers
    
    input_query = tf.keras.layers.Input((max_q_length,), dtype="int32")
    input_query_idf = tf.keras.layers.Input((max_q_length,), dtype="float32")
    input_sentence = tf.keras.layers.Input((max_d_length,), dtype="int32")
    
    if emb_matrix is None:
        interaction = ExactInteractions()
    else:
        interaction = SemanticInteractions(emb_matrix, learn_term_weights=learn_term_weights, trainable_embeddings=trainable_embeddings)
        
    ngram_convs = MultipleNgramConvs(max_ngram=max_ngram,
                                     k_max=k_max,
                                     k_polling_avg=k_polling_avg,
                                     polling_avg=polling_avg,
                                     use_mask=use_mask,
                                     filters=filters,
                                     activation="relu")
    softmax_IDF = MaskedSoftmax()
    concatenate = MaskedConcatenate(0)
    shuffle = ShuffleRows()
    
    if dense_hidden_units is None:
        aggregation_layer = tf.keras.layers.LSTM(out_put_dim, 
                                    dropout=0.0, 
                                    recurrent_regularizer=None, 
                                    recurrent_dropout=0.0, 
                                    unit_forget_bias=True, 
                                    recurrent_activation="hard_sigmoid", 
                                    bias_regularizer=None, 
                                    activation="relu", 
                                    recurrent_initializer="orthogonal", 
                                    kernel_regularizer=None, 
                                    kernel_initializer="glorot_uniform",
                                    unroll=True) # speed UP!!!
    elif isinstance(dense_hidden_units, list) :
        def _network(x):
            x = tf.keras.layers.Flatten()(x)
            for i,h in enumerate(dense_hidden_units):
                x = tf.keras.layers.Dense(h, activation="relu", name="aggregation_dense_"+str(i))(x)
            dout = tf.keras.layers.Dense(1, name="aggregation_output")(x)
            return dout
        
        aggregation_layer = _network
    else:
        raise RuntimeError("dense_hidden_units must be a list with the hidden size per layer")
    
    # build layers
    
    norm_idf = K.expand_dims(softmax_IDF(input_query_idf))
        
    x = interaction([input_query, input_sentence])
    x = ngram_convs(x)
    x = concatenate([x, norm_idf])
    if shuffle_query_terms:
        prefix_name += "S"
        x = shuffle(x)
    x = aggregation_layer(x)
    
    
    if name_model is None:
        name_model = (prefix_name+"_" if prefix_name != "" else "") + "PACRR"
    
    model = tf.keras.models.Model(inputs=[input_query, input_sentence, input_query_idf], outputs=x, name=name_model)
    return model

def semantic_exact_PACRR(semantic_pacrr, 
                         exact_pacrr,
                         dense_hidden_units=[4]):
    
    max_q_length = semantic_pacrr.input[0].shape[1]
    max_d_length = semantic_pacrr.input[1].shape[1]
    
    # init layers
    input_query = tf.keras.layers.Input((max_q_length,), dtype="int32")
    input_query_idf = tf.keras.layers.Input((max_q_length,), dtype="float32")
    input_sentence = tf.keras.layers.Input((max_d_length,), dtype="int32")
    
    concatenation = tf.keras.layers.Lambda(lambda x: K.concatenate(list(map(lambda y: K.expand_dims(y), x))) )
    combination = WeightedCombination()
    def _score(x):
        for i,h in enumerate(dense_hidden_units):
            x = tf.keras.layers.Dense(h, activation="relu")(x)
        return tf.keras.layers.Dense(1, activation="relu")(x)
    # build layers
    
    semantic_repr = semantic_pacrr([input_query, input_sentence, input_query_idf])
    exact_repr = exact_pacrr([input_query, input_sentence, input_query_idf])
    
    semantic_exact_repr = concatenation([semantic_repr, exact_repr])
    
    combined = combination(semantic_exact_repr)
    
    score = _score(combined)
    
    return  tf.keras.models.Model(inputs=[input_query, input_sentence, input_query_idf], outputs=score, name="Combined_PACRR")
    