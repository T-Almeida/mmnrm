import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K

from mmnrm.layers.interaction import SemanticInteractions, ExactInteractions
from mmnrm.layers.local_relevance import MultipleNgramConvs, MaskedSoftmax, SimpleMultipleNgramConvs
from mmnrm.layers.transformations import *
from mmnrm.layers.aggregation import *

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

            model = func(**kwargs["model"])
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
                       top_k_list = [3,5,10,15],): 

        if activation=="mish":
            activation = mish

        input_ids = tf.keras.layers.Input((max_passages, max_input_size, ), dtype="int32") #
        input_masks = tf.keras.layers.Input((max_passages, max_input_size, ), dtype="int32") #
        input_segments = tf.keras.layers.Input((max_passages, max_input_size, ), dtype="int32") # 
        input_mask_passages = tf.keras.layers.Input((max_passages,), dtype="bool") # (None, P)

        bert_model = TFBertModel.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract', 
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