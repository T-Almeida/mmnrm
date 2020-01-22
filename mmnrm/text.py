"""
Auxiliar code needed to better handle text inputs
"""
from collections import defaultdict
from os.path import join
import random

def TREC04_queries_transform(queires, type_query="title"):
    return list(map(lambda x:{"id":x["number"],
                         "query":x[type_query]
                        }, queires))

def TREC04_goldstandard_transform(goldstandard):
    _g = {}
    for _id, relevance in goldstandard.items():
        _g[_id]=defaultdict(list)
        
        for doc in relevance:
            _rel = int(doc[1])
            
            if _rel<0: #assert
                raise RuntimeError("value of relevance is negative??", doc[1])
            
            _g[_id][_rel].append(doc[0])
            
    return _g

def TREC04_results_transform(results):
    _g = {}
    for _id, relevance in results.items():
        _g[_id] = list(map(lambda x:{"id":x["DOCNO"], "text":x["HEADER"]+" "+x["HEADLINE"]+" "+x["TEXT"]}, relevance))
    return _g

def TREC04_merge_goldstandard_files(list_of_files, path_to_store):
    
    name = join(path_to_store, "temp_gs_{}.txt".format(int(random.random()*10000)))
    with open(name,"w") as wf:
        for f in list_of_files:
            with open(f,"r") as rf:
                for line in rf:
                    wf.write(rf)
                    
    return name
    