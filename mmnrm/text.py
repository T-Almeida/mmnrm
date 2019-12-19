"""
Auxiliar code needed to better handle text inputs
"""

def TREC04_queries_transform(queires, type_query="title"):
    return list(map(lambda x:{"id":x["number"],
                         "query":x[type_query]
                        }, queires))

def TREC04_goldstandard_transform(goldstandard):
    _g = {}
    for _id, relevance in goldstandard.items():
        _g[_id]={
            "pos_docs":[],
            "neg_docs":[]
        }
        for doc in relevance:
            _rel = int(doc[1])
            if _rel==0:
                _g[_id]["neg_docs"].append(doc[0])
            elif _rel>0:
                _g[_id]["pos_docs"].append(doc[0])
            else:
                raise RuntimeError("value of relevance is negative??", doc[1])
    return _g

def TREC04_results_transform(results):
    _g = {}
    for _id, relevance in results.items():
        _g[_id] = list(map(lambda x:{"id":x["DOCNO"], "text":x["HEADER"]+" "+x["HEADLINE"]+" "+x["TEXT"]}, relevance))
    return _g
