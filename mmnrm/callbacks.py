import tensorflow as tf
from tensorflow.keras import backend as K

from mmnrm.utils import save_model_weights, load_model_weights

import numpy as np

import matplotlib.pyplot as plt

import os
import time
import wandb
import tempfile
import shutil
import math

from collections import defaultdict

class Callback:
    def __init__(self, **kwargs):
        pass
    
    def on_epoch_start(self, training_obj, epoch):
        pass
    
    def on_epoch_end(self, training_obj, epoch):
        pass
        
    def on_step_start(self,training_obj, epoch, step):
        pass
        
    def on_step_end(self, training_obj, epoch, step, loss, time):
        pass
    
    def on_train_start(self, training_obj):
        pass
    
    def on_train_end(self, training_obj):
        pass
        
class TriangularLR(Callback):
    """
    From: https://arxiv.org/pdf/1506.01186.pdf
    
    adaptation from: https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/callbacks/cyclical_learning_rate.py
    """
    
    def __init__(
            self,
            base_lr=0.001,
            max_lr=0.006,
            step_size=2020.,
            mode='triangular2',
            gamma=1.,
            scale_fn=None,
            scale_mode='cycle'):
        super(TriangularLR, self).__init__()

        if mode not in ['triangular', 'triangular2',
                        'exp_range']:
            raise KeyError("mode must be one of 'triangular', "
                           "'triangular2', or 'exp_range'")
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2.**(x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** x
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.

        self._reset()
        
    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr is not None:
            self.base_lr = new_base_lr
        if new_max_lr is not None:
            self.max_lr = new_max_lr
        if new_step_size is not None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * \
                np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * \
                np.maximum(0, (1 - x)) * self.scale_fn(self.clr_iterations)
        
    def on_step_end(self, training_obj, epoch, step, loss, time):

        self.trn_iterations += 1
        self.clr_iterations += 1
        
        update_lr = self.clr()
        K.set_value(training_obj.optimizer.lr, update_lr)

        
class LRfinder(Callback):
    """
    adapted from: https://gist.github.com/WittmannF/c55ed82d27248d18799e2be324a79473
    """
    def __init__(self, min_lr, max_lr, steps=188, epoch=32, mom=0.9, stop_multiplier=None, 
                 reload_weights=True, batches_lr_update=5):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.mom = mom
        self.reload_weights = reload_weights
        self.batches_lr_update = batches_lr_update
        if stop_multiplier is None:
            self.stop_multiplier = -20*self.mom/3 + 10 # 4 if mom=0.9
                                                       # 10 if mom=0
        else:
            self.stop_multiplier = stop_multiplier
            
        self.iteration=steps*epoch
        self.learning_rates = np.geomspace(self.min_lr, self.max_lr, \
                                           num=self.iteration//self.batches_lr_update+1)
        
        self.losses = []
        self.lrs = []
        self.step_count = 0
        
        self.temp_dir = tempfile.mkdtemp()
    
    def on_train_start(self, training_obj):
        save_model_weights(os.path.join(self.temp_dir,"temp.h5"), training_obj.model)
    
    def on_step_end(self, training_obj, epoch, step, loss, time):

        if self.step_count%self.batches_lr_update==0:
            
            load_model_weights(os.path.join(self.temp_dir,"temp.h5"), training_obj.model)
            
            lr = self.learning_rates[self.step_count//self.batches_lr_update]            
            K.set_value(training_obj.optimizer.lr, lr)

            self.losses.append(loss)
            self.lrs.append(lr)
        self.step_count+=1

    def on_train_end(self, training_obj):
        
        load_model_weights(os.path.join(self.temp_dir,"temp.h5"), training_obj.model)
        
        shutil.rmtree(self.temp_dir)
                
        plt.figure(figsize=(12, 6))
        plt.plot(self.lrs, self.losses)
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.xscale('log')
        plt.show()

        
class Validation(Callback):
    def __init__(self, 
                 validation_collection=None, 
                 test_collection=None, 
                 comparison_metric="ndcg_cut_20", 
                 path_store = "/backup/NIR/best_model_weights",
                 **kwargs):
        super(Validation, self).__init__(**kwargs)
        self.validation_collection = validation_collection
        self.test_collection = test_collection
        self.current_best = 0
        self.comparison_metric = comparison_metric
        self.path_store = path_store
        
    def evaluate(self, model_score_fn, collection):
        generator_Y = collection.generator()
                
        q_scores = defaultdict(list)

        for i, _out in enumerate(generator_Y):
            query_id, Y, docs_ids = _out
            s_time = time.time()
            scores = model_score_fn(Y).numpy()[:,0].tolist()
            print("\rEvaluation {} | time {}".format(i, time.time()-s_time), end="\r")
            q_scores[query_id].extend(list(zip(docs_ids,scores)))

        # sort the rankings
        for query_id in q_scores.keys():
            q_scores[query_id].sort(key=lambda x:-x[1])

        # evaluate
        return collection.evaluate(q_scores)
    
    def on_epoch_start(self, training_obj, epoch):
        self.model_path = os.path.join(self.path_store, training_obj.model.name+".h5")
    
    def on_epoch_end(self, training_obj, epoch):
        if self.validation_collection is None:
            return None
        
        metrics = self.evaluate(training_obj.model_score, self.validation_collection)
        
        if metrics[self.comparison_metric]>self.current_best:
            self.current_best = metrics[self.comparison_metric]
            save_model_weights(self.model_path, training_obj.model)
            print("Saved current best:", self.current_best)
        
        print("Epoch {} | recall@100 {} | map@20 {} | NDCG@20 {} | P@20 {}"\
                                                  .format(epoch, 
                                                   metrics["recall_100"],
                                                   metrics["map_cut_20"],
                                                   metrics["ndcg_cut_20"],
                                                   metrics["P_20"]))
        return metrics
    
    def on_train_end(self, training_obj):
        if self.test_collection is None:
            return None
        
        if self.current_best>0:
            load_model_weights(self.model_path, training_obj.model)
        
        metrics = self.evaluate(training_obj.model_score, self.test_collection)
        print("\nTestSet final evaluation | recall@100 {} | map@20 {} | NDCG@20 {} | P@20 {}"\
                                                  .format(
                                                   metrics["recall_100"],
                                                   metrics["map_cut_20"],
                                                   metrics["ndcg_cut_20"],
                                                   metrics["P_20"]))
        return metrics

class PrinterEpoch(Callback):
    def __init__(self, steps_per_epoch=None, **kwargs):
        super(PrinterEpoch, self).__init__(**kwargs)
        self.losses_per_epoch = []
        self.steps_per_epoch = steps_per_epoch
    
    def on_epoch_start(self, training_obj, epoch):
        self.losses_per_epoch.append([])
    
    def on_epoch_end(self, training_obj, epoch):
        avg_loss = sum(self.losses_per_epoch[-1])/len(self.losses_per_epoch[-1])
        print("Epoch {} | avg Loss {}".format(epoch, avg_loss))
        return avg_loss
        
    def on_step_end(self, training_obj, epoch, step, loss, time):
        self.losses_per_epoch[-1].append(loss)
        print("\rStep {}/{} | Loss {} | time {}".format(step, self.steps_per_epoch, loss, time), end="\r")
    
class WandBValidationLogger(Validation, PrinterEpoch):
    def __init__(self, wandb_args, **kwargs):
        Validation.__init__(self, **kwargs)
        PrinterEpoch.__init__(self, **kwargs)
        self.wandb = wandb
        self.wandb.init(**wandb_args)
    
    def on_epoch_start(self, training_obj, epoch):
        PrinterEpoch.on_epoch_start(self, training_obj, epoch)
        Validation.on_epoch_start(self, training_obj, epoch)
    
    def on_epoch_end(self, training_obj, epoch):
        avg_loss = PrinterEpoch.on_epoch_end(self, training_obj, epoch)
        metrics = Validation.on_epoch_end(self, training_obj, epoch)
        self.wandb.log({'recall@100': metrics["recall_100"],
                       'map@20': metrics["map_cut_20"],
                       'ndcg@20': metrics["ndcg_cut_20"],
                       'P@20': metrics["P_20"],
                       'loss': avg_loss,
                       'epoch': epoch})
        
        self.wandb.run.summary["best_"+self.comparison_metric] = self.current_best
        
    def on_step_end(self, training_obj, epoch, step, loss, time):
        PrinterEpoch.on_step_end(self, training_obj, epoch, step, loss, time)
        self.wandb.log({'loss': float(loss)})
        
    def on_train_end(self, training_obj):
        metrics = Validation.on_train_end(self, training_obj)
        
        self.wandb.run.summary["test_"+self.comparison_metric] = metrics["ndcg_cut_20"]
        
def step_decay(epoch, initial_lrate):

    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    
    return lrate


class LearningRateScheduler(Callback):
    def __init__(self, initial_learning_rate, lr_fn=step_decay, **kwargs):
        super(LearningRateScheduler, self).__init__(**kwargs)
        self.initial_learning_rate = initial_learning_rate
        self.lr_fn = lr_fn
        
    def on_epoch_end(self, training_obj, epoch):
        
        new_lr = self.lr_fn(epoch, self.initial_learning_rate)

        # update the lr
        K.set_value(training_obj.optimizer.lr, new_lr)

    

