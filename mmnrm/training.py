"""
This file contains an abstraction for implment pairwise loss training
"""

import tensorflow as tf
from tensorflow.keras import backend as K
import time
import tempfile
import shutil
import subprocess
import os
from collections import defaultdict

from mmnrm.utils import save_model_weights, load_model_weights

import random
import numpy as np
import pickle

import wandb


def hinge_loss(positive_score, negative_score):
    return K.mean(K.maximum(0., 1. - positive_score + negative_score))

def pairwise_cross_entropy(positive_score, negative_score):
    positive_exp = K.exp(positive_score)
    return K.mean(-K.log(positive_exp/(positive_exp+K.exp(negative_score))))
    

class BaseTraining():
    def __init__(self, 
                 model,
                 loss,
                 train_collection,
                 validation_collection=None,
                 test_collection = None,
                 comparation_metric = "ndcg_cut_20",
                 path_store = "/backup/NIR/best_model_weights",
                 optimizer="adam", # keras optimizer
                 callbacks=[],
                 wandb_config={},
                 **kwargs): 
        super(BaseTraining, self).__init__(**kwargs)
        self.model = model
        self.loss = loss
        
        self.train_collection = train_collection
        self.validation_collection = validation_collection
        self.test_collection = test_collection
        
        self.comparation_metric = comparation_metric
        
        self.optimizer = tf.keras.optimizers.get(optimizer)
        
        self.path_store = path_store
        
        self.callbacks = callbacks
        
        ## config wandb
        wandb.init(project="nir-on-robust04",
                   name=model.name,
                   config=wandb_config)
        
    def draw_graph(self, name, *data):

        logdir = 'logs/func/'+name 
        writer = tf.summary.create_file_writer(logdir)

        tf.summary.trace_on(graph=True, profiler=True)

        self.training_step(*data)

        with writer.as_default():
            tf.summary.trace_export(
              name="training_trace",
              step=0,
              profiler_outdir=logdir)
            
    def train(self, epoch, draw_graph=True):
        raise NotImplementedError("This is an abstract class, should not be initialized")
    
class PairwiseTraining(BaseTraining):
    
    def __init__(self, loss=hinge_loss, **kwargs):
        super(PairwiseTraining, self).__init__(loss=loss, **kwargs)
    
    
    @tf.function # check if this can reutilize the computational graph for the prediction phase
    def model_score(self, inputs):
        print("\r[DEBUG] CALL MODEL_SCORE FUNCTION")
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

    def evaluate_test_set(self, test_set):
        generator_Y = test_set.generator()
                
        q_scores = defaultdict(list)

        for i, _out in enumerate(generator_Y):
            query_id, Y, docs_ids = _out
            s_time = time.time()
            scores = self.model_score(Y).numpy()[:,0].tolist()
            print("\rEvaluation {} | time {}".format(i, time.time()-s_time), end="\r")
            q_scores[query_id].extend(list(zip(docs_ids,scores)))

        # sort the rankings
        for query_id in q_scores.keys():
            q_scores[query_id].sort(key=lambda x:-x[1])

        # evaluate
        return test_set.evaluate(q_scores)
    
    def train(self, epoch, draw_graph=True):
        
        # create train generator
        steps = self.train_collection.get_steps()
        generator_X = self.train_collection.generator()
        
        positive_inputs, negative_inputs = next(generator_X)
        
        current_best = 0
        model_path = os.path.join(self.path_store, self.model.name+".h5")
        
        if draw_graph:
            self.draw_graph(positive_inputs, negative_inputs)
        
        for c in self.callbacks:
            c.on_train_start(self)
        
        for e in range(epoch):
            loss_step = []
            
            #execute callbacks
            for c in self.callbacks:
                c.on_epoch_start(self, e)
                
            for s in range(steps):
                
                #execute callbacks
                for c in self.callbacks:
                    c.on_step_start(self, e, s)
                    
                s_time = time.time()
                    
                loss = self.training_step(positive_inputs, negative_inputs)
                loss_step.append(loss)
                print("\rStep {}/{} | Loss {} | time {}".format(s, steps, loss, time.time()-s_time), end="\r")
                # send loss
                wandb.log({'loss': float(loss)})
                positive_inputs, negative_inputs = next(generator_X)
                
                #execute callbacks
                f_time = time.time()-s_time
                for c in self.callbacks:
                    c.on_step_end(self, e, s, loss, f_time)
            
            # perform evaluation if data is available
            if self.validation_collection is not None:
                metrics = self.evaluate_test_set(self.validation_collection)
                print("\nEpoch {} | avg loss {} | recall@100 {} | map@20 {} | NDCG@20 {} | P@20 {}"\
                                                  .format(e, 
                                                   np.mean(loss_step),
                                                   metrics["recall_100"],
                                                   metrics["map_cut_20"],
                                                   metrics["ndcg_cut_20"],
                                                   metrics["P_20"]))
                
                wandb.log({'recall@100': metrics["recall_100"],
                           'map@20': metrics["map_cut_20"],
                           'ndcg@20': metrics["ndcg_cut_20"],
                           'P@20': metrics["P_20"],
                           'loss': np.mean(loss_step),
                           'epoch': e})
                
                if metrics[self.comparation_metric] > current_best:
                    current_best = metrics[self.comparation_metric]
                    save_model_weights(model_path, self.model)
                    wandb.run.summary["best_"+self.comparation_metric] = current_best
                    print("Saved current best with score:", current_best)
            else:
                print("\nEpoch {} | avg loss {} | ".format(e, np.mean(loss_step)))
            
            #execute callbacks
            for c in self.callbacks:
                c.on_epoch_end(self, e, np.mean(loss_step))
        
        for c in self.callbacks:
            c.on_train_end(self)
        
        # load best model and test    
        if current_best>0:
            load_model_weights(model_path, self.model)
            print("Loaded current best with score:", current_best)
        
        if self.test_collection is not None:
            metrics = self.evaluate_test_set(self.test_collection)
            print(metrics)
            wandb.run.summary["test_"+self.comparation_metric] = metrics["ndcg_cut_20"]

            print("\n recall@100 {} | map@20 {} | NDCG@20 {} | P@20 {}"\
                                              .format(
                                               metrics["recall_100"],
                                               metrics["map_cut_20"],
                                               metrics["ndcg_cut_20"],
                                               metrics["P_20"]))
                                                                                          
                 
            
            
# Create a more abstract class that uses common elemetns like, b_size, transform_input etc...
            
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
    
    def get_config(self):
        data_json = {
            "query_list": self.query_list,
            "goldstandard_trec_file": self.goldstandard_trec_file,
            "query_docs": self.query_docs,
            "trec_script_eval_path": self.trec_script_eval_path
        } 
        
        return data_json
    
    def save(self, path):
        with open(path+".p", "wb") as f:
            pickle.dump(self.get_config(), f)
            
    @staticmethod
    def load(path):
        with open(path+".p", "rb") as f:
            config = pickle.load(f)
        
        return TestCollection(**config)
    
    def set_transform_inputs_fn(self, transform_inputs_fn):
        # build style method
        self.transform_inputs_fn = transform_inputs_fn
        return self
    
    def generator(self):
        # generator for the query, pos and negative docs
        gen_Y = self.__generate()
        
        if self.transform_inputs_fn is not None:
            gen_Y = self.transform_inputs_fn(gen_Y)
        
        # finally yield the input to the model
        for Y in gen_Y:
            yield Y
    
    def __generate(self):
        
        for query_data in self.query_list:
            for i in range(0, len(self.query_docs[query_data["id"]]), self.b_size):
                
                docs = self.query_docs[query_data["id"]][i:i+self.b_size]
                
                yield query_data["id"], query_data["query"], docs
    
    def __metrics_to_dict(self, metrics):
        return dict(map(lambda k:(k[0], float(k[1])), map(lambda x: tuple(map(lambda y: y.strip(), x.split("\tall"))), metrics.split("\n")[1:-1])))
    
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
    
    def evaluate(self, ranked_query_docs):
        metrics = None
        temp_dir = tempfile.mkdtemp()
        
        try:
            with open(os.path.join(temp_dir, "qret.txt"), "w") as f:
                for i, rank_data in enumerate(ranked_query_docs.items()):
                    for j,doc in enumerate(rank_data[1]):
                        _str = "{} Q0 {} {} {} run\n".format(rank_data[0],
                                                           doc[0],
                                                           j+1,
                                                           doc[1])
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
    def __init__(self, 
                 query_list, 
                 goldstandard, 
                 query_docs_subset=None, 
                 use_relevance_groups=False,
                 transform_inputs_fn=None, 
                 verbose=True, 
                 b_size=64, 
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
                                           }, ...],
                                           ...
                                       }
        """
        self.query_list = query_list # [{query data}]
        self.goldstandard = goldstandard # {query_id:[relevance docs]}
        self.use_relevance_groups = use_relevance_groups
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
            self.sub_set_goldstandard[_id] = defaultdict(list)
            
            for doc in relevance:
                k = self.__find_relevance_group(doc["id"], self.goldstandard[_id])
                if k>0:
                    if self.use_relevance_groups:
                        self.sub_set_goldstandard[_id][k].append(doc["id"])
                    else:
                        self.sub_set_goldstandard[_id][1].append(doc["id"])
                else:
                    # default add to the less relevance group
                    self.sub_set_goldstandard[_id][0].append(doc["id"])
                
                #add to the collection
                self.collection[doc["id"]] = doc["text"]

        # stats
        if self.verbose:
            max_keys = max(map(lambda x:max(x.keys()), self.sub_set_goldstandard.values()))
            
            for k in range(max_keys+1):
                print("Minimum number of relevance type({}) in the queries of the goldstandard sub set: {}".format(k, min(map(lambda x: len(x[k]), self.sub_set_goldstandard.values()))))
            
                print("Mean number of relevance type({}) in the queries of the goldstandard sub set: {}".format(k, sum(map(lambda x: len(x[k]), self.sub_set_goldstandard.values()))/len(self.sub_set_goldstandard)))
            
            print("Sub Collection size", len(self.collection))
    
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
        total_positives = sum(map(lambda x: sum([ len(x[k]) for k in x.keys() if k>0]), training_data.values()))
          
        return total_positives//self.b_size
    
    def generator(self, collection=None):
        # generator for the query, pos and negative docs
        gen_X = self.__generator(collection)
        
        # apply transformation ??
        if self.transform_inputs_fn is not None:
            gen_X = self.transform_inputs_fn(gen_X)
        
        # finally yield the input to the model
        while True:
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
                pos_doc = collection[pos_doc_id]
                
                _neg_len = len(training_data[selected_query["id"]][relevance_group-1])
                neg_doc_index = random.randint(0, _neg_len-1) if _neg_len>1 else 0
                neg_doc_id = training_data[selected_query["id"]][relevance_group-1][neg_doc_index]
                neg_doc = collection[neg_doc_id]
                
                y_query.append(selected_query["query"])
                y_pos_doc.append(pos_doc)
                y_neg_doc.append(neg_doc)
            
            yield (np.array(y_query), np.array(y_pos_doc), np.array(y_neg_doc))
    
    def get_config(self):
        data_json = {
            "query_list": self.query_list,
            "goldstandard": self.goldstandard,
            "use_relevance_groups": self.use_relevance_groups,
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