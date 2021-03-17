import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K

from mmnrm.layers.interaction import SemanticInteractions, ExactInteractions
from mmnrm.layers.local_relevance import MultipleNgramConvs, MaskedSoftmax, SimpleMultipleNgramConvs
from mmnrm.layers.transformations import *
from mmnrm.layers.aggregation import *

def savable_model(func):
    def function_wrapper(**kwargs):
        
        # create tokenizer
        if 'tokenizer' in kwargs:
            tk = kwargs['tokenizer']['class'].load_from_json(**kwargs['tokenizer']['attr'])
            tk.update_min_word_frequency(kwargs['tokenizer']['min_freq'])
        else:
            raise ValueError('Missing tokenizer parameter in the config')
            
        # create emb matrix
        if 'embedding' in kwargs:
            emb = kwargs['embedding']['class'].maybe_load(**kwargs['embedding']['attr'], tokenizer=tk)
            emb_matrix = emb.embedding_matrix()
            
            # check if it's normalized
            assert(all([ int(np.linalg.norm(v)+0.001)==1 for v in emb_matrix ]))
        else:
            raise ValueError('Missing embedding parameter in the config')
        
        model = func(**kwargs["model"], emb_matrix=emb_matrix)
        kwargs['func_name'] = func.__name__
        model.savable_config = kwargs
        model.tokenizer = tk

        return model

    return function_wrapper


def mish(inputs):
    return inputs * tf.math.tanh(tf.math.softplus(inputs))

@savable_model
def shallow_interaction_model(max_q_length=30,
              max_s_per_q_term=5,
              max_s_length=30,
              emb_matrix=None,
              filters=16,
              kernel_size=[3,3],
              aggregation_size=16,
              q_term_weight_mode=0,
              aggregation_mode=0,
              extraction_mode=0,
              score_mode=0,
              return_snippets_score=False,
              train_context_emgeddings=False,
              activation="selu"):
    """
    q_term_weight_mode: 0 - use term aggregation with embeddings
                        1 - use term aggregation with idf
                        
    aggregation_mode: 0 - use Bidirectional GRU
                      1 - use Bidirectional GRU + sig for sentence score follow another Bidirectional GRU for aggregation
                      2 - use Bidirectional GRU + sig for sentence score
                      3 - compute score independently + sig for sentence score
                      4 - mlp + sig
    
    extraction_mode: 0 - use CNN + GlobalMaxPool
                     1 - use CNN + [GlobalMaxPool, GlobalAvgPool]
                     2 - use CNN + [GlobalMaxPool, GlobalAvgPool, GlobalK-maxAvgPool]
                     3 - use CNN + [GlobalMaxPool, GlobalK-maxAvgPool]
                     4 - use CNN + GlobalKmaxPool
                     5 - use MultipleNgramConvs
    
    score_mode: 0 - Dense layer
                1 - MLP
                     
    

    """
    
    if activation=="mish":
        activation = mish
    
    kernel_size = tuple(kernel_size)
    
    return_q_embeddings = q_term_weight_mode==0
    
    input_query = tf.keras.layers.Input((max_q_length,), dtype="int32") # (None, Q)
    input_doc = tf.keras.layers.Input((max_q_length, max_s_per_q_term, max_s_length), dtype="int32") # (None, P, S)
    input_query_idf = tf.keras.layers.Input((max_q_length,), dtype="float32")
    
    interactions = SemanticInteractions(emb_matrix, return_q_embeddings=return_q_embeddings, learn_context=train_context_emgeddings)
    
    if extraction_mode==0:
        conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3,3), activation=activation)
        pool = tf.keras.layers.GlobalMaxPool2D()

        def extract(x):
            if return_embeddings:
                x, query_embeddings = interactions(x)
            else:
                x = interactions(x)
                query_embeddings = K.expand_dims(input_query_idf, axis=-1)
            x = conv(x)
            x = pool(x)
            return x, query_embeddings
        
    elif extraction_mode in [1, 2, 3]:
        conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,padding="SAME", activation=activation)
        max_pool = tf.keras.layers.GlobalMaxPool2D()
        avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        masked_avg_pool = GlobalMaskedAvgPooling2D()
        kmax_avg_pool = GlobalKmaxAvgPooling2D(kmax=5)
        concatenate = tf.keras.layers.Concatenate(axis=-1)
        
        def extract(x):
            if return_q_embeddings:
                x_interaction, query_embeddings = interactions(x)
            else:
                x_interaction = interactions(x)
                query_embeddings = K.expand_dims(input_query_idf, axis=-1)
            x = conv(x_interaction)
            max_x = max_pool(x)
            _concat = [max_x]
            if extraction_mode in [1, 2]:
                avg_x = avg_pool(x)
                _concat.append(avg_x)
            if extraction_mode in [2, 3]:
                kmax_x = kmax_avg_pool(x)
                _concat.append(kmax_x)
            x = concatenate(_concat)
            
            return x, query_embeddings
    elif extraction_mode==4:
        conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,padding="SAME", activation=activation)
        kmax_pool = GlobalKmax2D()
        
        def extract(x):
            if return_q_embeddings:
                x_interaction, query_embeddings = interactions(x)
            else:
                x_interaction = interactions(x)
                query_embeddings = K.expand_dims(input_query_idf, axis=-1)
            x = conv(x_interaction)
            x = kmax_pool(x)

            return x, query_embeddings
    elif extraction_mode==5:
        
        ngram_convs = MultipleNgramConvs(max_ngram=3, k_max=2,filters=filters, k_polling_avg = None, activation=activation)
        gru_pacrr = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation=activation))
        squeeze_l = tf.keras.layers.Lambda(lambda y: tf.squeeze(y, axis=-1))
        def extract(x):
            if return_q_embeddings:
                x_interaction, query_embeddings = interactions(x)
            else:
                x_interaction = interactions(x)
                query_embeddings = K.expand_dims(input_query_idf, axis=-1)
                
            x = ngram_convs(x_interaction)
            x = gru_pacrr(x)
            x = squeeze_l(x)
            
            return x, query_embeddings
        
    elif extraction_mode==6:   
        ngrams_convs = SimpleMultipleNgramConvs(max_ngram=3, filters=filters, activation=activation)
        
        max_pool = tf.keras.layers.GlobalMaxPool2D()
        avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        kmax_avg_pool = GlobalKmaxAvgPooling2D(kmax=5)
        concatenate = tf.keras.layers.Concatenate(axis=-1)
        
        def extract(x):
            if return_q_embeddings:
                x_interaction, query_embeddings = interactions(x)
            else:
                x_interaction = interactions(x)
                query_embeddings = K.expand_dims(input_query_idf, axis=-1)
            
            ngrams_signals_x = []
            for t_x in ngrams_convs(x_interaction):
                _max = max_pool(t_x)
                _avg = avg_pool(t_x)
                _kavg = kmax_avg_pool(t_x)
                ngrams_signals_x.append(concatenate([_max, _avg, _kavg]))
            
            x = concatenate(ngrams_signals_x)
            
            return x, query_embeddings
        
    elif extraction_mode==7:  
        conv_sem = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,padding="SAME", activation=activation)
        conv_exact = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,padding="SAME", activation=activation)
        max_pool = tf.keras.layers.GlobalMaxPool2D()
        avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        kmax_avg_pool = GlobalKmaxAvgPooling2D(kmax=5)
        concatenate = tf.keras.layers.Concatenate(axis=-1)
        
        exact_signals = ExactInteractions()
        
        def extract(x):
            if return_q_embeddings:
                x_interaction, query_embeddings = interactions(x)
            else:
                x_interaction = interactions(x)
                query_embeddings = K.expand_dims(input_query_idf, axis=-1)
            
            x_exact = exact_signals(x)
            
            x = [conv_sem(x_interaction), conv_exact(x_exact)]
            
            x = [ concatenate([max_pool(_x), avg_pool(_x), kmax_avg_pool(_x)]) for _x in x]
            
            x = concatenate(x)

            return x, query_embeddings
        
    elif extraction_mode==8:  
        conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,padding="SAME", activation=activation)

        max_pool = tf.keras.layers.GlobalMaxPool2D()
        avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        kmax_avg_pool = GlobalKmaxAvgPooling2D(kmax=5)
        concatenate = tf.keras.layers.Concatenate(axis=-1)
        
        exact_signals = ExactInteractions()
        
        def extract(x):
            if return_q_embeddings:
                x_interaction, query_embeddings = interactions(x)
            else:
                x_interaction = interactions(x)
                query_embeddings = K.expand_dims(input_query_idf, axis=-1)
            
            x_exact = exact_signals(x)
            x = concatenate([x_interaction, x_exact])
            x = conv(x)
            
            x = concatenate([max_pool(x), avg_pool(x), kmax_avg_pool(x)])

            return x, query_embeddings
        
        
    else:
        raise RuntimeError("invalid extraction_mode")
        
    if aggregation_mode==0:
        aggregation_senteces = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(aggregation_size))
        
    elif aggregation_mode==1:

        l1 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(1, return_sequences=True), merge_mode="sum")
        l2 = tf.keras.layers.Activation('sigmoid')
        l3 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(aggregation_size), merge_mode="sum")

        def aggregation_senteces(x):
            x = l1(x)
            x = l2(x)
            x = l3(x)

            return x
        
    elif aggregation_mode==2:
        
        l1_a = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(1, return_sequences=True), merge_mode="sum")
        l2_a = tf.keras.layers.Activation('sigmoid')
        l3_a = tf.keras.layers.Lambda(lambda y: tf.squeeze(y, axis=-1))
        
        def aggregation_senteces(x):
            x = l1_a(x)
            x = l2_a(x)
            x = l3_a(x)
            return x
    elif aggregation_mode==3:
        l1_a = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation="sigmoid"))
        l2_a = tf.keras.layers.Lambda(lambda y: tf.squeeze(y, axis=-1))
        
        def aggregation_senteces(x):
            x = l1_a(x)
            x = l2_a(x)
            return x    
    elif aggregation_mode==4:
        
        mlp = tf.keras.models.Sequential()
        mlp.add(tf.keras.layers.Dense(aggregation_size, activation=activation))
        mlp.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        
        l1_a = tf.keras.layers.TimeDistributed(mlp)
        l2_a = tf.keras.layers.Lambda(lambda y: tf.squeeze(y, axis=-1))
        
        def aggregation_senteces(x):
            x = l1_a(x)
            x = l2_a(x)
            return x  
    elif aggregation_mode==5:
        l1_a = SelfAttention(aggregation=False)
        l2_a = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation="sigmoid"))
        l3_a = tf.keras.layers.Lambda(lambda y: tf.squeeze(y, axis=-1))
        
        def aggregation_senteces(x):
            x = l1_a(x)
            x = l2_a(x)
            x = l3_a(x)
            return x    
    else:
        raise RuntimeError("invalid aggregation_mode")
        
    aggregation = TermAggregation(aggregate=False)
    aggregation_sum = tf.keras.layers.Lambda(lambda x:K.sum(x, axis=1))
    
    if score_mode==0:
        output_score = tf.keras.layers.Dense(1)
    elif score_mode==1:
        l1_score = tf.keras.layers.Dense(max_q_length, activation=activation)
        l2_score = tf.keras.layers.Dense(1)
        
        def output_score(x):
            x = l1_score(x)
            x = l2_score(x)
            
            return x
    else:
        raise RuntimeError("invalid score_mode")
    
    input_doc_unstack = tf.unstack(input_doc, axis=1)
    
    output_i = []
    for input_i in input_doc_unstack:
        input_i_unstack = tf.unstack(input_i, axis=1) 
        
        output_j = []
        for input_j in input_i_unstack:
            _out, query_embeddings = extract([input_query, input_j])
            output_j.append(_out) # [None, FM]
        output_j_stack = tf.stack(output_j, axis=1) # [None, P_Q, FM]
        
        output_i.append(aggregation_senteces(output_j_stack)) # [None, FM]
        
    output_i_stack = tf.stack(output_i, axis=1)  # [None, Q, FM]
    
    # aggregation
    q_term_weight_vector = aggregation([output_i_stack, query_embeddings])
    doc_vector = aggregation_sum(q_term_weight_vector)
    
    # score
    score = output_score(doc_vector)
    
    _output = [score]
    if return_snippets_score:
        _output.append(q_term_weight_vector)
        
    model = tf.keras.models.Model(inputs=[input_query, input_doc, input_query_idf], outputs=_output)
    model.sentence_tensor = output_i_stack
    return model




@savable_model
def sibm(emb_matrix,
         max_q_terms=30,
         max_passages=20,
         max_p_terms=30,
         filters=16,
         kernel_size=[3,3],
         match_threshold = 0.99,
         activation="mish",
         use_avg_pool=True,
         use_kmax_avg_pool=True,
         top_k_list = [3,5,10,15],
         score_hidden_units = None,
         semantic_normalized_query_match = False,
         return_snippets_score = False,
         DEBUG = False):
    
    if activation=="mish":
        activation = mish
    
    if score_hidden_units is None:
        score_hidden_units = top_k_list[-1]
    
    kernel_size = tuple(kernel_size)
        
    input_query = tf.keras.layers.Input((max_q_terms,), dtype="int32") # (None, Q)
    input_doc = tf.keras.layers.Input((max_passages, max_p_terms,), dtype="int32") # (None, P, S)
    
    semantic_interaction_layer = SemanticInteractions(emb_matrix, return_q_embeddings=True, einsum="bq,bps->bpqs")
    
    def embedding_matches_layer(x):
        x, query_embeddings = semantic_interaction_layer(x)
        
        if semantic_normalized_query_match:
            _tensor_squeeze = tf.squeeze(x, axis=-1)
            query_matches = tf.reduce_sum(tf.cast(_tensor_squeeze>=match_threshold, tf.float32) * _tensor_squeeze, axis=-1)/max_p_terms
        else:
            query_matches = tf.cast(tf.reduce_sum(tf.cast(tf.squeeze(x, axis=-1)>=match_threshold, tf.int8), axis=-1)>0,tf.int8)
        
        
        x = tf.reshape(x, shape=(-1, max_q_terms, max_p_terms, 1))

        return x, query_matches, query_embeddings
    
    ## convolutions
    conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding="SAME", activation=activation)
    
    ## pooling
    max_pool = tf.keras.layers.GlobalMaxPool2D()
    avg_pool = tf.keras.layers.GlobalAveragePooling2D()
    kmax_avg_pool = GlobalKmaxAvgPooling2D(kmax=5)

    ## concatenate layer
    concatenate = tf.keras.layers.Concatenate(axis=-1)
    
    ## Dense for sentence
    setence_dense = tf.keras.layers.Dense(1, activation="sigmoid")
    
    
    apriori_importance_layer = AprioriLayer()
    
    def sentence_relevance_nn(x, apriori_importance):

        ## sentence analysis
        x = conv(x)
        signals = [max_pool(x)]
        
        if use_avg_pool:
            signals += [avg_pool(x)]
            
        if use_kmax_avg_pool:
            signals += [kmax_avg_pool(x)]
        
        x = concatenate(signals)
        ## score
        x = setence_dense(x)
        x = tf.reshape(x, shape=(-1,max_passages,))
        
        x = apriori_importance * x
        
        top_k, indeces = tf.math.top_k(x, k=top_k_list[-1], sorted=True)
        
        
        sentence_features = [tf.expand_dims(tf.math.reduce_max(x, axis=-1),axis=-1), 
                             tf.expand_dims(tf.math.reduce_sum(x, axis=-1),axis=-1), 
                             tf.expand_dims(tf.math.reduce_mean(x, axis=-1),axis=-1),
                            ]
        
        sentence_features += [ tf.expand_dims(tf.math.reduce_mean(top_k[:,:k], axis=-1),axis=-1) for k in top_k_list]

        return tf.concat(sentence_features, axis=-1), indeces, x
        #scores, indeces = tf.math.top_k(apriori_importance * x, k=top_k, sorted=True)

            
        #return scores, indeces, x
        
    l1_score = tf.keras.layers.Dense(score_hidden_units, activation=activation)
    l2_score = tf.keras.layers.Dense(1)
    
    def document_score(x):
        x = l1_score(x)
        x = l2_score(x)
        return x
    
    interaction_matrix, query_matches, query_embeddings = embedding_matches_layer([input_query, input_doc])
    
    apriori_importance, query_weigts = apriori_importance_layer([input_query, query_matches, query_embeddings])
    
    s_scores, indices, cnn_sentence_scores = sentence_relevance_nn(interaction_matrix, apriori_importance)
    
    output_list = [document_score(s_scores)]
    
    if return_snippets_score:
        output_list += [s_scores, indices]
        
    if DEBUG:
        output_list += [s_scores, apriori_importance, query_weigts, query_matches, cnn_sentence_scores, interaction_matrix]
    
    return tf.keras.models.Model(inputs=[input_query, input_doc], outputs=output_list)

@savable_model
def sibm2(emb_matrix,
         max_q_terms=30,
         max_passages=20,
         max_p_terms=30,
         filters=16,
         kernel_size=[3,3],
         match_threshold = 0.99,
         activation="mish",
         use_avg_pool=True,
         use_kmax_avg_pool=True,
         top_k_list = [3,5,10,15],
         score_hidden_units = None,
         semantic_normalized_query_match = False,
         return_snippets_score = False,
         DEBUG = False):
    
    if activation=="mish":
        activation = mish
    
    if score_hidden_units is None:
        score_hidden_units = top_k_list[-1]
    
    kernel_size = tuple(kernel_size)
        
    input_query = tf.keras.layers.Input((max_q_terms,), dtype="int32") # (None, Q)
    input_doc = tf.keras.layers.Input((max_passages, max_p_terms,), dtype="int32") # (None, P, S)
    input_mask_passages = tf.keras.layers.Input((max_passages,), dtype="bool") # (None, P)
    
    semantic_interaction_layer = SemanticInteractions(emb_matrix, return_q_embeddings=True, einsum="bq,bps->bpqs")
    
    def embedding_matches_layer(x):
        x, query_embeddings = semantic_interaction_layer(x)
        
        if semantic_normalized_query_match:
            _tensor_squeeze = tf.squeeze(x, axis=-1)
            query_matches = tf.reduce_sum(tf.cast(_tensor_squeeze>=match_threshold, tf.float32) * _tensor_squeeze, axis=-1)/max_p_terms
        else:
            query_matches = tf.cast(tf.reduce_sum(tf.cast(tf.squeeze(x, axis=-1)>=match_threshold, tf.int8), axis=-1)>0,tf.int8)
        
        
        x = tf.reshape(x, shape=(-1, max_q_terms, max_p_terms, 1))

        return x, query_matches, query_embeddings
    
    ## convolutions
    conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding="SAME", activation=activation)
    
    ## pooling
    max_pool = tf.keras.layers.GlobalMaxPool2D()
    avg_pool = tf.keras.layers.GlobalAveragePooling2D()
    kmax_avg_pool = GlobalKmaxAvgPooling2D(kmax=5)

    ## concatenate layer
    concatenate = tf.keras.layers.Concatenate(axis=-1)
    
    ## Dense for sentence
    setence_dense = tf.keras.layers.Dense(1, activation="sigmoid")
    
    
    apriori_importance_layer = AprioriLayer()
    
    def sentence_relevance_nn(x, apriori_importance, mask_passages):
        
        mask_passages = tf.reshape(mask_passages, shape=(-1,)) #None, 1
        mask_passages_indices = tf.cast(tf.where(mask_passages), tf.int32)
        x = tf.gather_nd(x, mask_passages_indices) 
        
        ## sentence analysis
        x = conv(x)
        signals = [max_pool(x)]
        
        if use_avg_pool:
            signals += [avg_pool(x)]
            
        if use_kmax_avg_pool:
            signals += [kmax_avg_pool(x)]
        
        x = concatenate(signals)
        ## score
        x = setence_dense(x)
        
        scores_before_DEBUG = x
        
        x = tf.scatter_nd(mask_passages_indices, tf.squeeze(x), tf.shape(mask_passages))
        
        x = tf.reshape(x, shape=(-1,max_passages,))
        
        x = apriori_importance * x
        
        top_k, indeces = tf.math.top_k(x, k=top_k_list[-1], sorted=True)
        
        
        sentence_features = [tf.expand_dims(tf.math.reduce_max(x, axis=-1),axis=-1), 
                             tf.expand_dims(tf.math.reduce_sum(x, axis=-1),axis=-1), 
                             tf.expand_dims(tf.math.reduce_mean(x, axis=-1),axis=-1),
                            ]
        
        sentence_features += [ tf.expand_dims(tf.math.reduce_mean(top_k[:,:k], axis=-1),axis=-1) for k in top_k_list]

        return tf.concat(sentence_features, axis=-1), indeces, x, scores_before_DEBUG
        #scores, indeces = tf.math.top_k(apriori_importance * x, k=top_k, sorted=True)

            
        #return scores, indeces, x
        
    l1_score = tf.keras.layers.Dense(score_hidden_units, activation=activation)
    l2_score = tf.keras.layers.Dense(1)
    
    def document_score(x):
        x = l1_score(x)
        x = l2_score(x)
        return x
    
    interaction_matrix, query_matches, query_embeddings = embedding_matches_layer([input_query, input_doc])
    
    apriori_importance, query_weigts = apriori_importance_layer([input_query, query_matches, query_embeddings])
    
    s_scores, indices, cnn_sentence_scores, scores_before = sentence_relevance_nn(interaction_matrix, apriori_importance, input_mask_passages)
    
    output_list = [document_score(s_scores)]
    
    if return_snippets_score:
        output_list += [s_scores, indices]
        
    if DEBUG:
        output_list += [s_scores, scores_before, apriori_importance, query_weigts, query_matches, cnn_sentence_scores, interaction_matrix]
    
    return tf.keras.models.Model(inputs=[input_query, input_doc, input_mask_passages], outputs=output_list)


import sys
import subprocess
import pkg_resources

required = {'transformers'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

if not missing:
    from transformers import BertTokenizer, TFBertModel

    def transformer_model(func):
        def function_wrapper(**kwargs):
            # this setups the entired model tokenizer

            # create tokenizer
            if 'checkpoint_name' in kwargs:
                tokenizer = BertTokenizer.from_pretrained(kwargs['checkpoint_name'])
            else:
                raise ValueError('Missing checkpoint_name parameter in the config')

            model = func(**kwargs["model"], checkpoint_name=kwargs['checkpoint_name'])
            kwargs['func_name'] = func.__name__
            model.savable_config = kwargs
            model.tokenizer = tokenizer

            return model

        return function_wrapper

    @transformer_model
    def sibmtransfomer(max_passages = 20,
                       max_input_size = 128,
                       match_threshold = 0.9,
                       apriori_exact_match = False,
                       bert_train = False,
                       hidden_size = 768,
                       activation = "mish",
                       top_k_list = [3,5,10,15],
                       checkpoint_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"): 

        if activation=="mish":
            activation = mish

        input_ids = tf.keras.layers.Input((max_passages, max_input_size, ), dtype="int32") #
        input_masks = tf.keras.layers.Input((max_passages, max_input_size, ), dtype="int32") #
        input_segments = tf.keras.layers.Input((max_passages, max_input_size, ), dtype="int32") # 
        input_mask_passages = tf.keras.layers.Input((max_passages,), dtype="bool") # (None, P)

        bert_model = TFBertModel.from_pretrained(checkpoint_name, 
                                                 output_attentions = False,
                                                 output_hidden_states = False,
                                                 return_dict=True,
                                                 from_pt=True)
        bert_model.trainable = bert_train

        apriori_layer = AprioriLayerWmask()

        def skip_padding_data(input_ids, input_masks, input_segments, input_mask_passages):
            mask_passages = tf.reshape(input_mask_passages, shape=(-1,)) #None, 1
            mask_passages_indices = tf.cast(tf.where(mask_passages), tf.int32)

            input_ids = tf.gather_nd(tf.reshape(input_ids, shape=(-1, 128)),mask_passages_indices)
            input_masks = tf.gather_nd(tf.reshape(input_masks, shape=(-1, 128)),mask_passages_indices)
            input_segments = tf.gather_nd(tf.reshape(input_segments, shape=(-1, 128)),mask_passages_indices)

            return input_ids, input_masks, input_segments, mask_passages, mask_passages_indices

        def bert_contextualized_embeddings(input_ids, input_masks, input_segments):

            output = bert_model([input_ids, input_masks, input_segments])

            return output.pooler_output, output.last_hidden_state[:,1:,:]

        def embedding_matches_layer(embeddings, input_ids, input_masks, input_segments):

            #embeddings = x[0][:,1:,:] 
            #input_ids = x[1][:,1:]
            #input_masks = x[2][:,1:]
            #input_segments = x[3][:,1:]

            input_ids = input_ids[:,1:]
            input_masks = input_masks[:,1:]
            input_segments = input_segments[:,1:]

            # query mask, that will ignore the sep tokens
            #print(input_masks)
            #print(input_segments)
            mask_q = ((input_masks+input_segments)==1)
            mask_sep_tokens = input_ids == 3
            mask_q = tf.cast(tf.math.logical_xor(mask_sep_tokens,  (mask_q | mask_sep_tokens)), tf.float32)

            # sentence mask
            mask_s = tf.cast((input_segments==1), tf.float32)

            mask_interaction = tf.einsum("bq,bs->bqs", mask_q, mask_s)

            if apriori_exact_match:

                interaction_matrix = tf.cast(tf.einsum("bq,bs->bqs", input_ids, input_ids)==tf.expand_dims(tf.square(input_ids),axis=-1), tf.float32) * mask_interaction  
            else:
                ## using the embedding to perform the exact matching extraction
                embeddings = embeddings / tf.norm(embeddings, axis=-1, keepdims=True)

                interaction_matrix = tf.einsum("bqe,bse->bqs", embeddings, embeddings) * mask_interaction

            query_matches = tf.cast(tf.reduce_sum(tf.cast(interaction_matrix >= match_threshold, tf.int8), axis=-1)>0,tf.float32)

            return query_matches, mask_q

        l1_sentences_score = tf.keras.layers.Dense(1024, activation=activation)
        l2_sentences_score = tf.keras.layers.Dense(1, activation="sigmoid")

        def sentences_scores_layer(cls_embedding):

            x = l1_sentences_score(cls_embedding)
            x = l2_sentences_score(x)

            return x

        def document_features_layer(sentences_score, apriori_score, mask_passages, mask_passages_indices):

            x = apriori_score * sentences_score

            x = tf.scatter_nd(mask_passages_indices, tf.squeeze(x), tf.shape(mask_passages))

            x = tf.reshape(x, shape=(-1,max_passages,))

            top_k, indices = tf.math.top_k(x, k=top_k_list[-1], sorted=True)

            sentence_features = [tf.expand_dims(tf.math.reduce_max(x, axis=-1),axis=-1), 
                                 tf.expand_dims(tf.math.reduce_sum(x, axis=-1),axis=-1), 
                                 tf.expand_dims(tf.math.reduce_mean(x, axis=-1),axis=-1),
                                ]

            sentence_features += [ tf.expand_dims(tf.math.reduce_mean(top_k[:,:k], axis=-1),axis=-1) for k in top_k_list]

            return tf.concat(sentence_features, axis=-1)


        l1_score = tf.keras.layers.Dense(max_passages, activation=activation)
        l2_score = tf.keras.layers.Dense(1)

        def document_score(x):
            x = l1_score(x)
            x = l2_score(x)
            return x

        ## forward pass
        data_ids, data_masks, data_segments, mask_passages, mask_passages_indices = skip_padding_data(input_ids, input_masks, input_segments, input_mask_passages)

        cls_embedding, embeddings = bert_contextualized_embeddings(data_ids, data_masks, data_segments)

        query_matches, query_mask = embedding_matches_layer(embeddings, data_ids, data_masks, data_segments)

        sentence_scores = sentences_scores_layer(cls_embedding)

        apriori_scores = apriori_layer([embeddings, query_mask, query_matches])

        document_features = document_features_layer(sentence_scores, apriori_scores, mask_passages, mask_passages_indices)

        output_list = [document_score(document_features)]

        return tf.keras.models.Model(inputs=[input_ids, input_masks, input_segments, input_mask_passages], outputs=output_list)