import numpy as np
import random
import tensorflow as tf
import h5py

def set_random_seed(seed_value=42):
    tf.random.set_seed(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    
    
def save_model_weights(file_name, model):
    with h5py.File(file_name, 'w') as f:
        weight = model.get_weights()
        for i in range(len(weight)):
            f.create_dataset('weight'+str(i), data=weight[i])


def load_model_weights(file_name, model):
    with h5py.File(file_name, 'r') as f:
        weight = []
        for i in range(len(f.keys())):
            weight.append(f['weight'+str(i)][:])
        model.set_weights(weight)
        
def merge_dicts(list_of_dicts):
    # fast merge according to https://stackoverflow.com/questions/1781571/how-to-concatenate-two-dictionaries-to-create-a-new-one-in-python
    
    temp = dict(list_of_dicts[0], **list_of_dicts[1])
    
    for i in range(2, len(list_of_dicts)):
        temp.update(list_of_dicts[i])
        
    return temp

def flat_list(x):
    return sum(x, [])

def index_from_list(searchable_list, comparison):
    for i,item in enumerate(searchable_list):
        if comparison(item):
            return i
    return -1