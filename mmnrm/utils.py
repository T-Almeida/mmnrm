import numpy as np
import random
import tensorflow as tf
import h5py
import pickle
import mmnrm.modelsv2
from datetime import datetime as dt


def set_random_seed(seed_value=42):
    tf.random.set_seed(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    
    
def save_model_weights(file_name, model):
    with h5py.File(file_name+".h5", 'w') as f:
        weight = model.get_weights()
        for i in range(len(weight)):
            f.create_dataset('weight'+str(i), data=weight[i])

def load_model_weights(file_name, model):
    with h5py.File(file_name+".h5", 'r') as f:
        weight = []
        for i in range(len(f.keys())):
            weight.append(f['weight'+str(i)][:])
        model.set_weights(weight)
        
def load_neural_model(path_to_weights):
    
    rank_model = load_model(path_to_weights, change_config={"return_snippets_score":True})
    tk = rank_model.tokenizer
    
    model_cfg = rank_model.savable_config["model"]
    
    max_input_query = model_cfg["max_q_length"]
    max_input_sentence = model_cfg["max_s_length"]
    max_s_per_q_term = model_cfg["max_s_per_q_term"]
    
    # redundant code... replace
    max_sentences_per_query = model_cfg["max_s_per_q_term"]

    pad_query = lambda x, dtype='int32': tf.keras.preprocessing.sequence.pad_sequences(x, 
                                                                                       maxlen=max_input_query,
                                                                                       dtype=dtype, 
                                                                                       padding='post', 
                                                                                       truncating='post', 
                                                                                       value=0)

    pad_sentences = lambda x, dtype='int32': tf.keras.preprocessing.sequence.pad_sequences(x, 
                                                                                           maxlen=max_input_sentence,
                                                                                           dtype=dtype, 
                                                                                           padding='post', 
                                                                                           truncating='post', 
                                                                                           value=0)

    pad_docs = lambda x, max_lim, dtype='int32': x[:max_lim] + [[]]*(max_lim-len(x))

    idf_from_id_token = lambda x: math.log(tk.document_count/tk.word_docs[tk.index_word[x]])

    train_sentence_generator, test_sentence_generator = sentence_splitter_builderV2(tk, 
                                                                                      max_sentence_size=max_input_sentence,
                                                                                      mode=4)
    
    def test_input_generator(data_generator):

        data_generator = test_sentence_generator(data_generator)

        for _id, query, docs in data_generator:

            #tokenization
            query_idf = list(map(lambda x: idf_from_id_token(x), query))

            tokenized_docs = []
            ids_docs = []
            offsets_docs = []

            for doc in docs:

                padded_doc = pad_docs(doc["text"], max_lim=max_input_query)
                for q in range(len(padded_doc)):
                    padded_doc[q] = pad_docs(padded_doc[q], max_lim=max_sentences_per_query)
                    padded_doc[q] = pad_sentences(padded_doc[q])
                tokenized_docs.append(padded_doc)
                ids_docs.append(doc["id"])
                offsets_docs.append(doc["offset"])

            # padding
            query = pad_query([query])[0]
            query = [query] * len(tokenized_docs)
            query_idf = pad_query([query_idf], dtype="float32")[0]
            query_idf = [query_idf] * len(tokenized_docs)

            yield _id, [np.array(query), np.array(tokenized_docs), np.array(query_idf)], ids_docs, offsets_docs

    return rank_model, test_input_generator
        
def save_model(file_name, model):
    cfg = model.savable_config
    with open(file_name+".cfg","wb") as f:
        pickle.dump(model.savable_config ,f)
        
    # keep using h5py for weights
    save_model_weights(file_name, model)
    
def load_model(file_name, change_config={}):
    
    with open(file_name+".cfg","rb") as f:
        cfg = pickle.load(f)
    
    cfg["model"] = merge_dicts(cfg["model"], change_config)
    
    # create the model with the correct configuration
    model = getattr(mmnrm.modelsv2, cfg['func_name'])(**cfg)
    
    # load weights
    load_model_weights(file_name, model)
    
    return model
        
def merge_dicts(*list_of_dicts):
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

def overlap(snippetA, snippetB):
    """
    snippetA: goldSTD
    """
    if snippetA[0]>snippetB[1] or snippetA[1] < snippetB[0]:
        return 0
    else:
        if snippetA[0]>=snippetB[0] and snippetA[1] <= snippetB[1]:
            return snippetA[1] - snippetA[0] + 1
        if snippetA[0]>=snippetB[0] and snippetA[1] > snippetB[1]:
            return snippetB[1] - snippetA[0] + 1
        if snippetA[0]<snippetB[0] and snippetA[1] <= snippetB[1]:
            return snippetA[1] - snippetB[0] + 1
        if snippetA[0]<snippetB[0] and snippetA[1] > snippetB[1]:
            return snippetB[1] - snippetA[0] + 1
        
    return 0

def to_date(_str):
    for fmt in ("%Y-%m", "%Y-%m-%d", "%Y"):
        try:
            return dt.strptime(_str, fmt)
        except ValueError:
            pass
    raise ValueError("No format found")
