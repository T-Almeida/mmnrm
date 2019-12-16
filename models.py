import tensorflow as tf
from tensorflow.keras import backend as K

from interactions import SemanticInteractions
from local_relevance import MultipleNgramConvs, MaskedSoftmax
from transformations import MaskedConcatenate


def build_PACCR(max_q_length,
                max_d_length,
                emb_matrix,
                max_ngram = 3,
                k_max = 2,
                k_polling_avg = None, # do k_polling avg after convolution
                polling_avg = False, # do avg polling after convolution
                use_mask = True,
                filters=32, # can be a list or a function of input features and n-gram
                **kwargs):
    
    # init layers
    
    input_query = tf.keras.layers.Input((max_q_length,), dtype="int32")
    input_query_idf = tf.keras.layers.Input((max_q_length,), dtype="float32")
    input_sentence = tf.keras.layers.Input((max_d_length,), dtype="int32")
    
    semantic_interaction = SemanticInteractions(emb_matrix)
    ngram_convs = MultipleNgramConvs(max_ngram=max_ngram,
                                      k_max=k_max,
                                      k_polling_avg=k_polling_avg,
                                      polling_avg=polling_avg,
                                      use_mask=use_mask,
                                      filters=filters)
    softmax_IDF = MaskedSoftmax()
    concatenate = MaskedConcatenate(0)
    lstm = tf.keras.layers.LSTM(1)
    
    # build layers
    
    norm_idf = K.expand_dims(softmax_IDF(input_query_idf))
        
    x = semantic_interaction([input_query, input_sentence])
    x = ngram_convs(x)
    x = concatenate([x, norm_idf])
    x = lstm(x)
    
    model = tf.keras.models.Model(inputs=[input_query, input_sentence, input_query_idf], outputs=x, name="PACRR")
    return model