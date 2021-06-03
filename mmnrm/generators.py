import tensorflow as tf
import numpy as np
from mmnrm.utils import flat_list
from nltk.tokenize.punkt import PunktSentenceTokenizer

def train_test_generator_for_model(model):
    if model.name == "arcII":
        return _train_test_generator_for_model_full_doc(model)
    elif model.name in ["UPWM_acrII"]:
        return _train_test_generator_for_model_senteces(model)
    else:
        raise ValueError(f"There aren't any generator that supports the model: {model}")

##
# AUXILIAR FUNCTION FOR FEED
##


def _build_data_generators_full_doc(tokenizer, queries_sw=None, docs_sw=None):
        
    def maybe_tokenize(document):
        if "tokens" not in document:
            document["tokens"] = tokenizer.texts_to_sequences([document["text"]])[0]
            if docs_sw is not None:
                document["tokens"] = [token for token in document["tokens"] if token not in docs_sw]
    
    def train_generator(data_generator):
        while True:

            # get the batch triplet
            query, pos_docs, neg_docs = next(data_generator)

            # tokenization, this can be cached for efficientcy porpuses NOTE!!
            tokenized_query = tokenizer.texts_to_sequences(query)

            if queries_sw is not None:
                for tokens in tokenized_query:
                    tokenized_query = [token for token in tokens if token not in queries_sw] 
            
            saveReturn = True
            
            for batch_index in range(len(pos_docs)):
                
                # tokenizer with cache in [batch_index][tokens]
                maybe_tokenize(pos_docs[batch_index])
                
                # assertion
                if len(pos_docs[batch_index]["tokens"])<=3:
                    saveReturn = False
                    break # try a new resampling, NOTE THIS IS A EASY FIX PLS REDO THIS!!!!!!!
                          # for obvious reasons
                
                maybe_tokenize(neg_docs[batch_index])
                
                # assertion
                if len(neg_docs[batch_index]["tokens"])<=3:
                    saveReturn = False
                    break
                
            if saveReturn: # this is not true, if the batch is rejected
                yield tokenized_query, pos_docs, neg_docs

    def test_generator(data_generator):
        for _id, query, docs in data_generator:
            tokenized_queries = []
            for i in range(len(_id)):
                # tokenization
                tokenized_query = tokenizer.texts_to_sequences([query[i]])[0]

                if queries_sw is not None:
                    tokenized_query = [token for token in tokenized_query if token not in queries_sw] 
                
                tokenized_queries.append(tokenized_query)
                    
        
                for doc in docs[i]:
                    maybe_tokenize(doc)
                                                 
            yield _id, tokenized_queries, docs
            
    return train_generator, test_generator

def _train_test_generator_for_model_full_doc(model):

    if "model" in model.savable_config:
        cfg = model.savable_config["model"]
    
    train_gen, test_gen = _build_data_generators_full_doc(model.tokenizer)
    
    pad_tokens = lambda x, max_len, dtype='int32': tf.keras.preprocessing.sequence.pad_sequences(x, 
                                                                                           maxlen=max_len,
                                                                                           dtype=dtype, 
                                                                                           padding='post', 
                                                                                           truncating='post', 
                                                                                           value=0)

    
    def maybe_padding(document):
        if isinstance(document["tokens"], list):
            document["tokens"] = pad_tokens([document["tokens"]], cfg["max_p_terms"])[0]
            
    def train_generator(data_generator):
 
        for query, pos_docs, neg_docs in train_gen(data_generator):
            
            query = pad_tokens(query, cfg["max_q_terms"])
            
            pos_docs_array = []

            neg_docs_array = []

            
            # pad docs, use cache here
            for batch_index in range(len(pos_docs)):
                maybe_padding(pos_docs[batch_index])
                pos_docs_array.append(pos_docs[batch_index]["tokens"])
                maybe_padding(neg_docs[batch_index])
                neg_docs_array.append(neg_docs[batch_index]["tokens"])
            
            yield [query, np.array(pos_docs_array)], [query, np.array(neg_docs_array)]
            
    def test_generator(data_generator):
        
        for ids, query, docs in test_gen(data_generator):
            
            docs_ids = []
            docs_array = []
            docs_mask_array = []
            query_array = []
            query_ids = []
            
            for i in range(len(ids)):
                
                for doc in docs[i]:
                    # pad docs, use cache here
                    maybe_padding(doc)
                    docs_array.append(doc["tokens"])
                    docs_ids.append(doc["id"])
                
                query_tokens = pad_tokens([query[i]], cfg["max_q_terms"])[0]
                query_tokens = [query_tokens] * len(docs[i])
                query_array.append(query_tokens)
                    
                query_ids.append([ids[i]]*len(docs[i]))
            
            #print(np.array(docs_mask_array))
            
            yield flat_list(query_ids), [np.array(flat_list(query_array)), np.array(docs_array)], docs_ids, None
            
    return train_generator, test_generator

def _build_data_generators_for_sentences(tokenizer, queries_sw=None, docs_sw=None):
    
    punkt_sent_tokenizer = PunktSentenceTokenizer().span_tokenize
    
    def maybe_tokenize(documents):
        if "tokens" not in documents:
            split = [documents["text"][s:e] for s,e in punkt_sent_tokenizer(documents["text"])]
            documents["tokens"] = tokenizer.texts_to_sequences(split)
            if docs_sw is not None:
                for tokenized_sentence in documents["tokens"]:
                    tokenized_sentence = [token for token in tokenized_sentence if token not in docs_sw]
    
    def train_generator(data_generator):
        while True:

            # get the batch triplet
            query, pos_docs, neg_docs = next(data_generator)

            # tokenization, this can be cached for efficientcy porpuses NOTE!!
            tokenized_query = tokenizer.texts_to_sequences(query)

            if queries_sw is not None:
                for tokens in tokenized_query:
                    tokenized_query = [token for token in tokens if token not in queries_sw] 
            
            saveReturn = True
            
            for batch_index in range(len(pos_docs)):
                
                # tokenizer with cache in [batch_index][tokens]
                maybe_tokenize(pos_docs[batch_index])
                
                # assertion
                if all([ len(sentence)==0  for sentence in pos_docs[batch_index]["tokens"]]):
                    saveReturn = False
                    break # try a new resampling, NOTE THIS IS A EASY FIX PLS REDO THIS!!!!!!!
                          # for obvious reasons
                
                maybe_tokenize(neg_docs[batch_index])
                
            if saveReturn: # this is not true, if the batch is rejected
                yield tokenized_query, pos_docs, neg_docs

    def test_generator(data_generator):
        for _id, query, docs in data_generator:
            tokenized_queries = []
            for i in range(len(_id)):
                # tokenization
                tokenized_query = tokenizer.texts_to_sequences([query[i]])[0]

                if queries_sw is not None:
                    tokenized_query = [token for token in tokenized_query if token not in queries_sw] 
                
                tokenized_queries.append(tokenized_query)
                    
        
                for doc in docs[i]:
                    maybe_tokenize(doc)
                                                 
            yield _id, tokenized_queries, docs
            
    return train_generator, test_generator

def _train_test_generator_for_model_senteces(model):

    if "model" in model.savable_config:
        cfg = model.savable_config["model"]
    
    train_gen, test_gen = _build_data_generators_for_sentences(model.tokenizer)
    
    pad_tokens = lambda x, max_len, dtype='int32': tf.keras.preprocessing.sequence.pad_sequences(x, 
                                                                                           maxlen=max_len,
                                                                                           dtype=dtype, 
                                                                                           padding='post', 
                                                                                           truncating='post', 
                                                                                           value=0)

    pad_sentences = lambda x, max_lim, dtype='int32': x[:max_lim] + [[]]*(max_lim-len(x))
    
    def maybe_padding(document):
        if isinstance(document["tokens"], list):
            #overflow prevention
            bounded_doc_passage = min(cfg["max_passages"],len(document["tokens"]))
            document["sentences_mask"] = [True] * bounded_doc_passage + [False] * (cfg["max_passages"]-bounded_doc_passage)
            document["tokens"] = pad_tokens(pad_sentences(document["tokens"], cfg["max_passages"]), cfg["max_p_terms"])
            
    def train_generator(data_generator):
 
        for query, pos_docs, neg_docs in train_gen(data_generator):
            
            query = pad_tokens(query, cfg["max_q_terms"])
            
            pos_docs_array = []
            pos_docs_mask_array = []
            neg_docs_array = []
            neg_docs_mask_array = []
            
            # pad docs, use cache here
            for batch_index in range(len(pos_docs)):
                maybe_padding(pos_docs[batch_index])
                pos_docs_array.append(pos_docs[batch_index]["tokens"])
                pos_docs_mask_array.append(pos_docs[batch_index]["sentences_mask"])
                maybe_padding(neg_docs[batch_index])
                neg_docs_array.append(neg_docs[batch_index]["tokens"])
                neg_docs_mask_array.append(neg_docs[batch_index]["sentences_mask"])
            
            yield [query, np.array(pos_docs_array), np.array(pos_docs_mask_array)], [query, np.array(neg_docs_array), np.array(neg_docs_mask_array)]
            
    def test_generator(data_generator):
        
        for ids, query, docs in test_gen(data_generator):
            
            docs_ids = []
            docs_array = []
            docs_mask_array = []
            query_array = []
            query_ids = []
            
            for i in range(len(ids)):
                
                for doc in docs[i]:
                    # pad docs, use cache here
                    maybe_padding(doc)
                    docs_array.append(doc["tokens"])
                    docs_mask_array.append(doc["sentences_mask"])
                    docs_ids.append(doc["id"])
                
                query_tokens = pad_tokens([query[i]], cfg["max_q_terms"])[0]
                query_tokens = [query_tokens] * len(docs[i])
                query_array.append(query_tokens)
                    
                query_ids.append([ids[i]]*len(docs[i]))
            
            #print(np.array(docs_mask_array))
            
            yield flat_list(query_ids), [np.array(flat_list(query_array)), np.array(docs_array), np.array(docs_mask_array)], docs_ids, None
            
    return train_generator, test_generator