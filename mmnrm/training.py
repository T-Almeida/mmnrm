"""
This file contains an abstraction for implment pairwise loss training
"""

import tensorflow as tf
from tensorflow.keras import backend as K
import time
import tempfile
import os

import random
import numpy as np
import pickle

def hinge_loss(positive_score, negative_score):
    return K.mean(K.maximum(0., 1. - positive_score + negative_score))
    

class PairwiseTraining():
    
    def __init__(self, 
                 model,
                 train_collection,
                 test_collection = None,
                 loss=hinge_loss, 
                 optimizer="adam", # keras optimizer
                 **kwargs):
        
        super(PairwiseTraining, self).__init__(**kwargs)
        self.model = model
        self.train_collection = train_collection
        self.test_collection = test_collection
        self.loss = loss
        
        self.optimizer = tf.keras.optimizers.get(optimizer)
    
    def draw_graph(self, *data):

        logdir = 'logs/func/pairwise_training' 
        writer = tf.summary.create_file_writer(logdir)

        tf.summary.trace_on(graph=True, profiler=True)

        self.training_step(*data)

        with writer.as_default():
            tf.summary.trace_export(
              name="training_trace",
              step=0,
              profiler_outdir=logdir)
    
    @tf.function # check if this can reutilize the computational graph for the prediction phase
    def model_score(self, inputs):
        print("[DEBUG] CALL MODEL_SCORE FUNCTION")
        return self.model(inputs)
    
    @tf.function # build a static computational graph
    def training_step(self, pos_in, neg_in):

        # manual optimization
        with tf.GradientTape() as tape:
            pos_score = self.model_score(pos_in)
            neg_score = self.model_score(neg_in)

            loss = self.loss(pos_score, neg_score)

        # using auto-diff to get the gradients
        grads = tape.gradient(loss, self.model.trainable_weights)

        # applying the gradients using an optimizer
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        return loss

    
    def train(self, epoch, draw_graph=True):
        
        # create train generator
        steps = self.train_collection.get_steps()
        generator_X = self.train_collection.generator()
        
        positive_inputs, negative_inputs = next(generator_x)
        
        if draw_graph:
            self.draw_graph(positive_inputs, negative_inputs)

        for e in range(epoch):
            loss_step = []
            for s in range(steps):
                
                s_time = time.time()
                loss = self.training_step(positive_inputs, negative_inputs)
                loss_step.append(loss)
                print("Step {}/{} | Loss {} | time {} \t\t".format(s, steps, loss, time.time()-s_time), end="\r")
                
                positive_inputs, negative_inputs = next(generator_x)
            
            # perform evaluation if data is available
            if self.test_collection is not None:
                generator_Y = self.test_collection.generator()
                
                results = defaultdict(list)
                for query_id, Y in generator_Y:
                    scores = self.model_score(Y)
                    results[query_id].append(scores)
                
            print("\rEpoch {} | avg loss {}".format(e, np.mean(loss_step)))
            
            ## TEST
            return results
            
            

class TestCollection:
    def __init__(self, query_list, goldstandard_trec_file, query_docs, trec_script_eval_path, transform_inputs_fn=None, b_size=64, **kwargs):
        """
        query_list - must be a list with the following format :
                     [
                         {
                             id: <str>
                             query: <str>
                         },
                         ...
                     ]
        
        goldstandard_trec_file - name of the file with trec goldstandard:
                       
        query_docs  - dictionary with documents to be ranked by the model
                       {
                           id: [{
                               id: <str>
                               text: <str>
                           }],
                           ...
                       }
                       
        trec_script_eval_path - path to the TREC evaluation script
        """
        self.query_list = query_list 
        self.goldstandard_trec_file = goldstandard_trec_file 
        self.query_docs = query_docs
        self.trec_script_eval_path = trec_script_eval_path
        self.transform_inputs_fn = transform_inputs_fn
        
    def batch_size(self, b_size=32):
        # build style method
        self.b_size = b_size
        return self
    
    def set_transform_inputs_fn(self, transform_inputs_fn):
        # build style method
        self.transform_inputs_fn = transform_inputs_fn
        return self
    
    def generator(self):
        # generator for the query, pos and negative docs
        gen_Y = self.__generator()
        
        # apply transformation ??
        if self.transform_inputs_fn is not None:
            gen_Y = self.transform_inputs_fn(gen_Y)
        
        # finally yield the input to the model
        with True:
            yield next(gen_Y)
    
    def __generate(self):
        
        for query_data in self.query_list:
            for i in range(0, len(self.query_docs[query_data["id"]]), self.b_size):
                
                docs = self.query_docs[query_data["id"]][i:i+self.b_size]
                queries = [query_data["text"]] * len(docs)
                
                yield query_data["id"], [queries, docs]
    
    def __metrics_to_dict(self, metrics):
        return dict(map(lambda x: tuple(map(lambda y: y.strip(),x.split("\tall"))), metrics.split("\n")[:-1]))
    
    def evaluate(self, ranked_query_docs):
        metrics = None
        temp_dir = tempfile.mkdtemp()
        
        try:
            with open(join(temp_dir, "qret.txt"), "w") as f:
                for i, rank_data in enumerate(ranked_query_docs.items):
                    for j,doc in enumerate(rank_data[1]):
                        _str = "{} Q0 {} {} {} run\n".format(rank_data[0],
                                                           doc["id"],
                                                           j+1,
                                                           doc["score"])
                        f.write(_str)
                        
            # evaluate
            trec_eval_res = subprocess.Popen(
                [self.trec_script_eval_path, '-m', 'all_trec', self.goldstandard_trec_file, os.path.join(temp_dir, "qret.txt")],
                stdout=subprocess.PIPE, shell=False)

            (out, err) = trec_eval_res.communicate()
            trec_eval_res = out.decode("utf-8")
            
            metrics = self.__metrics_to_dict(trec_eval_res)

        except Exception as e:
            raise e # maybe handle the exception in the future
        finally:
            # always remove the temp directory
            print("Remove {}".format(temp_dir))
            shutil.rmtree(temp_dir)
            
        return metrics
            
            
class TrainCollection:
    def __init__(self, query_list, goldstandard, transform_inputs_fn=None, query_docs_subset=None, verbose=True, b_size=64, **kwargs):
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
                               pos_docs: [<str:id>, <str:id>, ...],
                               neg_docs: [<str:id>, <str:id>, ...] (optional, if ommited negative sampling will be performed)
                           },
                           ...
                       }
                       
        query_docs_subset (optional) - if a previous retrieved method were used to retrieved the TOP_K documents, this parameter
                                       can be used to ignore the collection and instead use only the TOP_K docs
                                       {
                                           id: [{
                                               id: <str>
                                               text: <str>
                                           }],
                                           ...
                                       }
        """
        self.query_list = query_list # [{query data}]
        self.goldstandard = goldstandard # {query_id:[relevance docs]}
        self.transform_inputs_fn = transform_inputs_fn
        self.verbose = verbose
        self.b_size = b_size
        
        if "sub_set_goldstandard" in kwargs:
            self.sub_set_goldstandard = kwargs.pop("sub_set_goldstandard")
        else:
            self.sub_set_goldstandard = None
        
        if "collection" in kwargs:
            self.collection = kwargs.pop("collection")
        else:
            self.collection = None
            
        if len(kwargs)>0:
            raise ValueError("Following arguments were not recognized:", kwargs.keys())
        
        self.__build(query_docs_subset)
    
    def __build(self, query_docs_subset):
        
        if query_docs_subset is None:
            # number of samples
            return # 
        
        self.sub_set_goldstandard = {}
        self.collection = {}
        
        # coverage test
        unique_pos_docs_goldstandard_ids = set()
        unique_neg_docs_goldstandard_ids = set()
        
        for g_result in self.goldstandard.values():
            for _id in g_result["pos_docs"]:
                unique_pos_docs_goldstandard_ids.add(_id)
            for _id in g_result["neg_docs"]:
                unique_neg_docs_goldstandard_ids.add(_id)
        
        unjudged_docs = 0
        
        # filter the goldstandard
        for _id, relevance in query_docs_subset.items():
            self.sub_set_goldstandard[_id] = {
                "pos_docs":[],
                "neg_docs":[]
            }
            
            for doc in relevance:
                if doc["id"] in self.goldstandard[_id]["pos_docs"]:
                    self.sub_set_goldstandard[_id]["pos_docs"].append(doc["id"])
                    self.collection[doc["id"]] = doc["text"]
                elif doc["id"] in self.goldstandard[_id]["neg_docs"]:
                    self.sub_set_goldstandard[_id]["neg_docs"].append(doc["id"])
                    self.collection[doc["id"]] = doc["text"]
                else:
                    unjudged_docs+=1

        # stats
        if self.verbose:
            print("Minimum number of pos_docs in a query of the goldstandard sub set:", min(map(lambda x: len(x["pos_docs"]), self.sub_set_goldstandard.values())))
            print("Minimum number of neg_docs in a query of the goldstandard sub set:", min(map(lambda x: len(x["neg_docs"]), self.sub_set_goldstandard.values())))
            
            print("Mean number of pos_docs in a query of the goldstandard sub set:", sum(map(lambda x: len(x["pos_docs"]), self.sub_set_goldstandard.values()))/len(self.sub_set_goldstandard))
            print("Mean number of neg_docs in a query of the goldstandard sub set:", sum(map(lambda x: len(x["neg_docs"]), self.sub_set_goldstandard.values()))/len(self.sub_set_goldstandard))
            
            print("Sub Collection size", len(self.collection))
            print("Number of documents without judgment", unjudged_docs)
    
    def batch_size(self, b_size=32):
        # build style method
        self.b_size = b_size
        return self
    
    def set_transform_inputs_fn(self, transform_inputs_fn):
        # build style method
        self.transform_inputs_fn = transform_inputs_fn
        return self
    
    def __get_goldstandard(self):
        
        if self.collection is not None:
            return self.sub_set_goldstandard
        else:
            return self.goldstandard
    
    def get_steps(self):
        
        training_data = self.__get_goldstandard()
        
        # an epoch will be defined with respect to the total number of positive pairs
        total_positives = sum(map(lambda x: len(x["pos_docs"]), training_data.values()))
          
        return total_positives//self.b_size
    
    def generator(self, collection=None):
        # generator for the query, pos and negative docs
        gen_X = self.__generator(collection)
        
        # apply transformation ??
        if self.transform_inputs_fn is not None:
            gen_X = self.transform_inputs_fn(gen_X)
        
        # finally yield the input to the model
        with True:
            yield next(gen_X)
                
        
    
    def __generator(self, collection=None):
        
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
            for _ in range(self.b_size):
                selected_query = self.query_list[random.randint(0, len(self.query_list)-1)]
                
                pos_doc_index = random.randint(0, len(training_data[selected_query["id"]]["pos_docs"])-1)
                pos_doc_id = training_data[selected_query["id"]]["pos_docs"][pos_doc_index]
                pos_doc = collection[pos_doc_id]
                
                neg_doc_index = random.randint(0, len(training_data[selected_query["id"]]["neg_docs"])-1)
                neg_doc_id = training_data[selected_query["id"]]["neg_docs"][neg_doc_index]
                neg_doc = collection[neg_doc_id]
                
                y_query.append(selected_query["query"])
                y_pos_doc.append(pos_doc)
                y_neg_doc.append(neg_doc)
            
            yield np.array(y_query), np.array(y_pos_doc), np.array(y_neg_doc)
    
    def get_config(self):
        data_json = {
            "query_list": self.query_list,
            "goldstandard": self.goldstandard,
            "verbose": self.verbose,
            "sub_set_goldstandard": self.sub_set_goldstandard,
            "collection": self.collection,
            "b_size": self.b_size,
        } 
        
        return data_json
    
    def save(self, path):
        with open(path+".p", "wb") as f:
            pickle.dump(self.get_config(), f)
            
    @staticmethod
    def load(path):
        with open(path+".p", "rb") as f:
            config = pickle.load(f)
        
        return TrainCollection(**config)