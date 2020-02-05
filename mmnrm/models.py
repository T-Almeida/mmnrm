import tensorflow as tf
from tensorflow.keras import backend as K

from mmnrm.layers.interaction import SemanticInteractions, ExactInteractions
from mmnrm.layers.local_relevance import MultipleNgramConvs, MaskedSoftmax
from mmnrm.layers.transformations import MaskedConcatenate, ShuffleRows
from mmnrm.layers.aggregation import WeightedCombination, KmaxAggregation

def build_PACRR(max_q_length,
                max_d_length,
                emb_matrix = None,
                learn_context = False,
                trainable_embeddings = False,
                learn_term_weights = False,
                dense_hidden_units = None,
                max_ngram = 3,
                k_max = 2,
                activation="relu",
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
        interaction = SemanticInteractions(emb_matrix, 
                                           learn_term_weights=learn_term_weights, 
                                           trainable_embeddings=trainable_embeddings,
                                           learn_context=learn_context,
                                           use_mask=True)
        
    ngram_convs = MultipleNgramConvs(max_ngram=max_ngram,
                                     k_max=k_max,
                                     k_polling_avg=k_polling_avg,
                                     polling_avg=polling_avg,
                                     use_mask=use_mask,
                                     filters=filters,
                                     activation=activation)
    softmax_IDF = MaskedSoftmax()
    
    if use_mask:
        concatenate = MaskedConcatenate(0)
    else:
        concatenate = tf.keras.layers.Concatenate()
    
    if dense_hidden_units is None:
        aggregation_layer = tf.keras.layers.LSTM(out_put_dim, 
                                    dropout=0.0, 
                                    recurrent_regularizer=None, 
                                    recurrent_dropout=0.0, 
                                    unit_forget_bias=True, 
                                    recurrent_activation="hard_sigmoid", 
                                    bias_regularizer=None, 
                                    activation=activation, 
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
        shuffle = ShuffleRows()
        prefix_name += "S"
        x = shuffle(x)
    x = aggregation_layer(x)
    
    
    if name_model is None:
        name_model = (prefix_name+"_" if prefix_name != "" else "") + "PACRR"
    
    model = tf.keras.models.Model(inputs=[input_query, input_sentence, input_query_idf], outputs=x, name=name_model)
    return model

def sentence_PACRR(pacrr, sentence_per_doc, type_combination=0, activation="relu"):
    """
    type_combination - 0: use MLP
                       1: use WeightedCombination + MLP
                       2: GRU
    """
    max_q_length = pacrr.input[0].shape[1]
    max_d_length = pacrr.input[1].shape[1]
    
    input_query = tf.keras.layers.Input((max_q_length,), dtype="int32") # (None, Q)
    input_query_idf = tf.keras.layers.Input((max_q_length,), dtype="float32") # (None, Q)
    input_doc = tf.keras.layers.Input((sentence_per_doc, max_d_length), dtype="int32") # (None, P, S)
    
    #aggregate = tf.keras.layers.GRU(1, activation="relu")
    #aggregate = WeightedCombination()
    
    def aggregate(x):
        #x = tf.keras.layers.Dense(25, activation="relu")(x)
        x = KmaxAggregation(k=5)(x)
        #x = tf.squeeze(x, axis=-1)
        x = tf.keras.layers.Dense(6, activation="selu")(x)
        return tf.keras.layers.Dense(1, activation=None)(x)
    
    #def aggregate(x):
        #x = tf.keras.layers.Dense(25, activation="relu")(x)
    #    return K.max(tf.squeeze(x, axis=-1), axis=-1, keepdims=True)
    
    sentences = tf.unstack(input_doc, axis=1) #[(None,S), (None,S), ..., (None,S)]
    pacrr_sentences = []
    for sentence in sentences:
        pacrr_sentences.append(pacrr([input_query, sentence, input_query_idf]))
        
    pacrr_sentences = tf.stack(pacrr_sentences, axis=1)
    #pacrr_sentences = tf.squeeze(pacrr_sentences, axis=-1)
    score = aggregate(pacrr_sentences)
    
    return tf.keras.models.Model(inputs=[input_query, input_doc, input_query_idf], outputs=score, name="Sentence_"+pacrr.name), pacrr_sentences
        
def semantic_exact_PACRR(semantic_pacrr, 
                         exact_pacrr,
                         type_combination=0,
                         dense_hidden_units=[4]):
    
    """
    type_combination - 0: use MLP
                       1: use WeightedCombination + MLP

                       
    """
    
    max_q_length = semantic_pacrr.input[0].shape[1]
    max_d_length = semantic_pacrr.input[1].shape[1]
    
    # init layers
    input_query = tf.keras.layers.Input((max_q_length,), dtype="int32")
    input_query_idf = tf.keras.layers.Input((max_q_length,), dtype="float32")
    input_sentence = tf.keras.layers.Input((max_d_length,), dtype="int32")
    
    
    def _aggregate(x):
        if type_combination==0:
            return tf.keras.layers.Concatenate(axis=-1)(x)
        elif type_combination==1:
            x = tf.keras.layers.Lambda(lambda x: K.concatenate(list(map(lambda y: K.expand_dims(y), x))) )(x)
            return WeightedCombination()(x)
        else:
            raise RuntimeError("invalid type_combination")
        
    
    def _score(x):
        for i,h in enumerate(dense_hidden_units):
            x = tf.keras.layers.Dense(h, activation="relu")(x)
        return tf.keras.layers.Dense(1, activation="relu")(x)
    # build layers
    
    semantic_repr = semantic_pacrr([input_query, input_sentence, input_query_idf])
    exact_repr = exact_pacrr([input_query, input_sentence, input_query_idf])
    
    combined = _aggregate([semantic_repr, exact_repr])
    
    score = _score(combined)
    
    return  tf.keras.models.Model(inputs=[input_query, input_sentence, input_query_idf], outputs=score, name="Combined_PACRR")
    