import time
import tempfile
import shutil
import subprocess
import os
from collections import defaultdict

from mmnrm.utils import save_model_weights, load_model_weights, set_random_seed, merge_dicts, flat_list, index_from_list

import random
import numpy as np
import pickle
import nltk
import math

from mmnrm.training import BaseCollection

class TrainCollectionV2(BaseCollection):
    def __init__(self, 
                 query_list, 
                 goldstandard, 
                 query_docs_subset=None, 
                 use_relevance_groups=False,
                 verbose=True, 
                 **kwargs):
        """
        query_list - must be a list with the following format :
                     [
                         {
                             id: <str>
                             query: <str>
                         },
                         ...
                     ]
        
        goldstandard - must be a dictionary with the following format:
                       {
                           id: {
                               0: [<str:id>, <str:id>, ...],
                               1: [<str:id>, <str:id>, ...],
                               2: ...,
                               ...
                           },
                           ...
                       }
                       
        query_docs_subset (optional) - if a previous retrieved method were used to retrieved the TOP_K documents, this parameter
                                       can be used to ignore the collection and instead use only the TOP_K docs
                                       {
                                           id: [{
                                               id: <str>
                                               text: <str>
                                               score: <float>
                                           }, ...],
                                           ...
                                       }
        """
        super(TrainCollectionV2, self).__init__(**kwargs)
        self.query_list = query_list # [{query data}]
        self.goldstandard = goldstandard # {query_id:[relevance docs]}
        self.use_relevance_groups = use_relevance_groups
        self.verbose = verbose
        
        if "sub_set_goldstandard" in kwargs:
            self.sub_set_goldstandard = kwargs.pop("sub_set_goldstandard")
        else:
            self.sub_set_goldstandard = None
        
        if "collection" in kwargs:
            self.collection = kwargs.pop("collection")
        else:
            self.collection = None
        
        self.skipped_queries = []

        self.__build(query_docs_subset)
    
    def __find_relevance_group(self, doc_id, search_gs):
        for k in search_gs.keys():
            if doc_id in search_gs[k]:
                return k
        return -1
    
    def __build(self, query_docs_subset):
        
        if query_docs_subset is None:
            # number of samples
            return #
        
        self.sub_set_goldstandard = {}
        self.collection = {}
  
        # filter the goldstandard
        for _id, relevance in query_docs_subset.items():
            
            if _id not in self.goldstandard:
                self.skipped_queries.append(_id)
                continue
            
            # do not use queries without true positives
            # this add an overhead that can be avoided by refactor the follwing for loop!
            unique_relevants = set(sum([self.goldstandard[_id][k] for k in self.goldstandard[_id].keys() if k>0], []))
            if all([ doc["id"] not in unique_relevants for doc in relevance ]):
                self.skipped_queries.append(_id)
                continue
            
            self.sub_set_goldstandard[_id] = defaultdict(list)
            
            for doc in relevance:
                k = self.__find_relevance_group(doc["id"], self.goldstandard[_id])
                if k>0:
                    if self.use_relevance_groups:
                        self.sub_set_goldstandard[_id][k].append({"id":doc["id"],"score":doc["score"]})
                    else:
                        self.sub_set_goldstandard[_id][1].append({"id":doc["id"],"score":doc["score"]})
                else:
                    # default add to the less relevance group
                    self.sub_set_goldstandard[_id][0].append({"id":doc["id"],"score":doc["score"]})
                
                #add to the collection
                self.collection[doc["id"]] = doc["text"]
        
        # remove the skipped queries from the data
        index_to_remove = []
        
        for skipped in self.skipped_queries:
            _index = index_from_list(self.query_list, lambda x: x["id"]==skipped)
            if _index>-1:
                index_to_remove.append(_index)
        index_to_remove.sort(key=lambda x:-x)
        
        # start removing from the tail
        for _index in index_to_remove:
            del self.query_list[_index]
        
        # stats
        if self.verbose:
            max_keys = max(map(lambda x:max(x.keys()), self.sub_set_goldstandard.values()))
            
            for k in range(max_keys+1):
                print("Minimum number of relevance type({}) in the queries of the goldstandard sub set: {}".format(k, min(map(lambda x: len(x[k]), self.sub_set_goldstandard.values()))))
            
                print("Mean number of relevance type({}) in the queries of the goldstandard sub set: {}".format(k, sum(map(lambda x: len(x[k]), self.sub_set_goldstandard.values()))/len(self.sub_set_goldstandard)))
            
            print("Sub Collection size", len(self.collection))
            print("Number of skipped question, due to lack of true positives", len(self.skipped_queries))
    
    def __get_goldstandard(self):
        
        if self.collection is not None:
            return self.sub_set_goldstandard
        else:
            return self.goldstandard
    
    def get_steps(self):
        
        training_data = self.__get_goldstandard()
        
        # an epoch will be defined with respect to the total number of positive pairs
        total_positives = sum(map(lambda x: sum([ len(x[k]) for k in x.keys() if k>0]), training_data.values()))
          
        return total_positives//self.b_size

    def _generate(self, collection=None, **kwargs):
        
        # sanity check
        assert(not(self.sub_set_goldstandard==None and collection==None))
        
        training_data = self.__get_goldstandard()
        
        # TODO this condition is dependent on the previous
        if collection is None:
            collection = self.collection
            
        while True:
            # TODO check if it is worthit to use numpy to vectorize these operations
            
            y_query = []
            y_pos_doc = []
            y_neg_doc = []
            
            # build $batch_size triples and yield
            query_indexes = random.sample(population=list(range(len(self.query_list))), k=self.b_size)
            for q_i in query_indexes:
                selected_query = self.query_list[q_i]
                #print(selected_query["id"])
                # select the relevance group, (only pos)
                positive_keys=list(training_data[selected_query["id"]].keys())
                #print("positive_keys", positive_keys)
                positive_keys.remove(0)
                #print("positive_keys", positive_keys)
                if len(positive_keys)>1:
                    group_len = list(map(lambda x: len(training_data[selected_query["id"]][x]), positive_keys))
                    total = sum(group_len)
                    prob = list(map(lambda x: x/total, group_len))
                    #print("probs", prob)
                    relevance_group = np.random.choice(positive_keys, p=prob)
                else:
                    relevance_group = positive_keys[0]
                
                _pos_len = len(training_data[selected_query["id"]][relevance_group])
                pos_doc_index = random.randint(0, _pos_len-1) if _pos_len>1 else 0
                pos_doc_id = training_data[selected_query["id"]][relevance_group][pos_doc_index]
                pos_doc = {"text":collection[pos_doc_id["id"]], "score":pos_doc_id["score"]}
                
                _neg_len = len(training_data[selected_query["id"]][relevance_group-1])
                neg_doc_index = random.randint(0, _neg_len-1) if _neg_len>1 else 0
                neg_doc_id = training_data[selected_query["id"]][relevance_group-1][neg_doc_index]
                neg_doc = {"text":collection[neg_doc_id["id"]], "score":neg_doc_id["score"]}
                
                y_query.append(selected_query["query"])
                y_pos_doc.append(pos_doc)
                y_neg_doc.append(neg_doc)
            
            yield (np.array(y_query), np.array(y_pos_doc), np.array(y_neg_doc))
    
    def get_config(self):
        super_config = super().get_config()
        
        data_json = {
            "query_list": self.query_list,
            "goldstandard": self.goldstandard,
            "use_relevance_groups": self.use_relevance_groups,
            "verbose": self.verbose,
            "sub_set_goldstandard": self.sub_set_goldstandard,
            "collection": self.collection,
        } 
        
        return dict(data_json, **super_config) #fast dict merge
    
    
class TestCollectionV2(BaseCollection):
    def __init__(self, 
                 query_list,
                 query_docs, 
                 evaluator=None,
                 skipped_queries = [],
                 **kwargs):
        """
        query_list - must be a list with the following format :
                     [
                         {
                             id: <str>
                             query: <str>
                         },
                         ...
                     ]
                       
        query_docs  - dictionary with documents to be ranked by the model
                       {
                           id: [{
                               id: <str>
                               text: <str>
                               score: <float>
                           }],
                           ...
                       }
                       
        """
        super(TestCollectionV2, self).__init__(**kwargs)
        self.query_list = query_list 
        self.query_docs = query_docs
        self.evaluator = evaluator
        
        self.skipped_queries = skipped_queries
        
        if isinstance(self.evaluator, dict):
            self.evaluator = self.evaluator["class"].load(**self.evaluator)

    def get_config(self):
        super_config = super().get_config()
        
        data_json = {
            "query_list": self.query_list,
            "query_docs": self.query_docs,
            "skipped_queries": self.skipped_queries,
            "evaluator": self.evaluator.get_config()
        } 
        
        return dict(data_json, **super_config) #fast dict merge
    
    def _generate(self, **kwargs):
        
        for query_data in self.query_list:
            if query_data["id"] in self.skipped_queries:
                continue
                
            for i in range(0, len(self.query_docs[query_data["id"]]), self.b_size):
                docs = self.query_docs[query_data["id"]][i:i+self.b_size]
                
                yield query_data["id"], query_data["query"], docs
    
    def evaluate_pre_rerank(self, output_metris=["recall_100", "map_cut_20", "ndcg_cut_20", "P_20"]):
        """
        Compute evaluation metrics over the documents order before been ranked
        """ 
        ranked_format = {k:list(map(lambda x:(x[1]["id"], len(v)-x[0]), enumerate(v))) for k,v in self.query_docs.items()}
        
        metrics = self.evaluate(ranked_format)
        
        if isinstance(output_metris, list):
            return [ (m, metrics[m]) for m in output_metris]
        else:
            return metrics
    
    def evaluate_oracle(self, output_metris=["recall_100", "map_cut_20", "ndcg_cut_20", "P_20"]):
        metrics = self.evaluator.evaluate_oracle()
    
        if isinstance(output_metris, list):
            return [ (m, metrics[m]) for m in output_metris]
        else:
            return metrics

    def evaluate(self, ranked_query_docs):
        return self.evaluator.evaluate(ranked_query_docs)

def compute_extra_features(query_tokens, tokenized_sentences_doc, idf_fn):
    
    doc_tokens = sum(tokenized_sentences_doc, [])
    
    bi_gram_doc_tokens = set([(doc_tokens[i],doc_tokens[i+1]) for i in range(len(doc_tokens)-1)])
    bi_gram_query_tokens = set([(query_tokens[i],query_tokens[i+1]) for i in range(len(query_tokens)-1)])
    
    doc_tokens = set(doc_tokens)
    query_tokens = set(query_tokens)
    
    # compute percentage of q-terms in D
    num_dt_in_Q = len([ x for x in doc_tokens if x in query_tokens])
    num_Q = len(query_tokens)
    
    qt_in_D = num_dt_in_Q/num_Q
    
    # compute the weighted percentage of q-terms in D
    w_dt_in_Q = sum([ idf_fn(x) for x in doc_tokens if x in query_tokens])
    w_Q = sum([idf_fn(x) for x in query_tokens])
    
    W_qt_in_D = w_dt_in_Q/w_Q
    
    # compute the percentage of bigrams matchs in D
    num_bi_dt_in_bi_Q = len([ x for x in bi_gram_doc_tokens if x in bi_gram_query_tokens])
    num_bi_Q = len(bi_gram_query_tokens)
    
    bi_qt_in_bi_D = num_bi_dt_in_bi_Q/num_bi_Q
    
    return [qt_in_D, W_qt_in_D, bi_qt_in_bi_D]
    
    
    
def sentence_splitter_builderV2(tokenizer, mode=4, max_sentence_size=21):
    """
    Return a transform_inputs_fn for training and test as a tuple
    
    For now only the mode 4 is supported since it was the best from the previous version!
    
    mode 4: similar to 2, but uses sentence splitting instead of fix size
    """
    idf_from_id_token = lambda x: math.log(tokenizer.document_count/tokenizer.word_docs[tokenizer.index_word[x]])
    
    def train_splitter(data_generator):

        while True:
        
            # get the batch triplet
            query, pos_docs, neg_docs = next(data_generator)
            
            # tokenization
            query = tokenizer.texts_to_sequences(query)

            new_pos_docs = []
            new_neg_docs = []
            
            new_pos_extra_features = []
            new_neg_extra_features = []
            
            # sentence splitting
            if mode==4:
                
                for b in range(len(pos_docs)):
                    new_pos_docs.append([])
                    new_neg_docs.append([])
                    
                    _temp_pos_docs = nltk.sent_tokenize(pos_docs[b]["text"])
                    _temp_pos_docs = tokenizer.texts_to_sequences(_temp_pos_docs)
                    
                    _temp_neg_docs = nltk.sent_tokenize(neg_docs[b]["text"])
                    _temp_neg_docs = tokenizer.texts_to_sequences(_temp_neg_docs)
                    
                    # compute extra features
                    extra_features_pos_doc = compute_extra_features(query[b], _temp_pos_docs, idf_from_id_token)
                    extra_features_neg_doc = compute_extra_features(query[b], _temp_neg_docs, idf_from_id_token)
                    
                    # add the bm25 score
                    extra_features_pos_doc.append(pos_docs[b]["score"])
                    extra_features_neg_doc.append(neg_docs[b]["score"])
                    
                    # add all the extra features
                    new_pos_extra_features.append(extra_features_pos_doc)
                    new_neg_extra_features.append(extra_features_neg_doc)
                    
                    # split by exact matching
                    for t_q in query[b]:
                        # entry for the query-term
                        new_pos_docs[-1].append([])
                        new_neg_docs[-1].append([])
                        
                        for pos_sent in _temp_pos_docs:
                            # exact math for the pos_document
                            for i,t_pd in enumerate(pos_sent):
                                if t_pd==t_q:
                                    new_pos_docs[-1][-1].append(pos_sent)
                                    break

                        for neg_sent in _temp_neg_docs:
                            for i,t_nd in enumerate(neg_sent):
                                if t_nd==t_q:
                                    new_neg_docs[-1][-1].append(neg_sent)
                                    break
            else:
                raise NotImplementedError("Missing implmentation for mode "+str(mode))
                
            yield query, new_pos_docs, new_pos_extra_features, new_neg_docs, new_neg_extra_features
            
            
    def test_splitter(data_generator):

        for _id, query, docs in data_generator:

            # tokenization
            tokenized_query = tokenizer.texts_to_sequences([query])[0]
            for doc in docs:
                if isinstance(doc["text"], list):
                    continue # cached tokenization

                # sentence splitting
                new_docs = []
                if mode==4:
                    _temp_new_docs = tokenizer.texts_to_sequences(nltk.sent_tokenize(doc["text"]))
                    
                    doc["extra_features"] = compute_extra_features(tokenized_query, _temp_new_docs, idf_from_id_token)+[doc["score"]]
                    
                    for t_q in tokenized_query:
                        new_docs.append([])
                        for _new_doc in _temp_new_docs:
                            for i,t_d in enumerate(_new_doc):
                                if t_d==t_q:
                                    new_docs[-1].append(_new_doc)
                                    break
                else:
                    raise NotImplementedError("Missing implmentation for mode "+str(mode))
                                                                    
                doc["text"] = new_docs
                                                                    
            yield _id, tokenized_query, docs

    return train_splitter, test_splitter
