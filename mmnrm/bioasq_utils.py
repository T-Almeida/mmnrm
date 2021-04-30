import os
from os.path import join
import tempfile
import shutil
import pickle
import gc
import json
import tarfile
import codecs
import sys 
from mmnrm.evaluation import f_map, f_recall
from datetime import datetime as dt
from nir.utils import change_bm25_parameters
import copy
from trectools import fusion, TrecRun
from collections import defaultdict

def fusion_rrf(run_paths, output_path):
    runs = [ TrecRun(path) for path in run_paths]
    fused_run = fusion.reciprocal_rank_fusion(runs)
    fused_run.print_subset(output_path, topics=fused_run.topics())

def load_trec_run(file, queries):
    final_run = copy.deepcopy(queries)
    run = defaultdict(list)
    with open(file, "r") as f:
        for line in f:
            line = line.split(" ")
            run[line[0]].append({
                "id":line[2],
                "score":line[4],
            })
     
    for q in final_run:
        q["documents"] = run[q["id"]]
    
    return final_run


def load_trec_run_snippets(file, queries):
    final_run = copy.deepcopy(queries)
    run = defaultdict(list)
    with open(file, "r") as f:
        for line in f:
            line = line.split(" ")
            run[line[0]].append({
                "id":line[2],
                "score":line[4],
            })
     
    for q in final_run:
        q["snippets"] = run[q["id"]]
    
    return final_run

   
def convert_trec_run_to_bioasq(trec_path, queries, bioasq_path):
    run = load_trec_run(trec_path, queries)
    write_as_bioasq(run, bioasq_path)

    
def convert_trec_run_to_bioasq_wSnippets(trec_path, trec_path_snippets, cached_snipepts, queries, bioasq_path):
    doc_run = load_trec_run(trec_path, queries)
    snippet_run = load_trec_run_snippets(trec_path_snippets, queries)
    
    snippet_run = {q["id"]:q for q in snippet_run}
    
    for q in doc_run:
        q["snippets"] = [ cached_snipepts[s["id"]] for s in snippet_run[q["id"]]["snippets"] ]
       
    write_as_bioasq(doc_run, bioasq_path)
    
def write_as_bioasq(run, file, max_docs=10, max_snippets=10):
    final_run = copy.deepcopy(run)
    
    for query in final_run:
        
        if "query" in query:
            query["body"] = query.pop("query")
        
        if "documents" in query:
            query["documents"] = list(map(lambda x:"http://www.ncbi.nlm.nih.gov/pubmed/"+x["id"], query["documents"]))[:max_docs]
        else: 
            query["documents"] = []
        
        if "snippets" not in query:
            query["snippets"] = []
        else:
            snippets = []
            for snippet in query["snippets"][:max_snippets]:
                section = "title" if snippet["is_title"] else "abstract"
                snippets.append({
                    "beginSection": section,
                    "endSection": section,
                    "text": snippet["text"],
                    "document": "http://www.ncbi.nlm.nih.gov/pubmed/"+snippet["doc_id"],
                    "offsetInBeginSection": snippet["start"],
                    "offsetInEndSection": snippet["end"]
                })
            query["snippets"] = snippets
               
        
    with open(file, "w", encoding="utf-8") as f:
        json.dump({"questions":final_run}, f)

def evaluation(run, gs, top_n):
    predictions = []
    expectations = []

    for query in run:
        if query["id"] in gs:
            predictions.append(list(map(lambda x:x["id"], query["documents"])))
            expectations.append(gs[query["id"]]) #gs
            
    return f_map(predictions, expectations, bioASQ_version=8, at=top_n),f_recall(predictions, expectations, at=top_n)

def separate_queries_goldstandard(queires, additional_keys=[]):
    clean_queires = []
    gs = {}
    additional_keys = ["id", "query"] + additional_keys
    for x in queires:
        clean_queires.append({k:x[k] for k in additional_keys})
        gs[x["id"]] = list(map(lambda y : y.split("/")[-1], x["documents"]))
    return clean_queires, gs

def create_document_run(queries, run, snippets = None):
    final_run = copy.deepcopy(queries)
    
    for query in final_run:
        query["documents"] = run[query["id"]]
        if snippets is not None:
            query["snippets"] = snippets[query["id"]]
    
    return final_run

def save_document_run(run, file):
    with open(file, "w") as f:
        json.dump(run,f)
        
def load_document_run(file, dict_format=False):
    with open(file, "r") as f:
        run = json.load(f)
    
    if isinstance(run,list) and dict_format:
        _temp = {}
        for q in run:
            _temp[q["id"]] = q["documents"]
        run = _temp
    
    return run


def write_as_trec_snippets(run, file):
    
    with open(file, "w") as f:
        for query in run:
            for rank,s in enumerate(query["snippets"]):
                f.write("{} Q0 {} {} {} {}\n".format(query["id"],
                                                     s["snippet_id"],
                                                     rank,
                                                     s["score"],
                                                     "bioasq_as_trec"))

def write_as_trec(run, file):
    
    with open(file, "w") as f:
        for query in run:
            for rank,doc in enumerate(query["documents"]):
                f.write("{} Q0 {} {} {} {}\n".format(query["id"],
                                                     doc["id"],
                                                     rank,
                                                     doc["score"],
                                                     "bioasq_as_trec"))
                   

def load_document_trec_run(file, *runs):
    full_run = {}
    runs = runs[0]

    for q in load_document_run(runs[0]):
        _temp = {}
        for doc in q["documents"]:
            _temp[doc["id"]] = doc
        
        full_run[q["id"]] = _temp
    
    for r in runs[1:]:
        for q in load_document_run(r):
            for doc in q["documents"]:
                if doc["id"] not in full_run[q["id"]]:
                    full_run[q["id"]][doc["id"]] = doc
        
    run = defaultdict(list)
    with open(file, "r") as f:
        for line in f:
            line = line.split(" ")
            run[line[0]].append({
                "id":line[2],
                "score":line[4],
                "text":full_run[line[0]][line[2]]["text"],
                "title":full_run[line[0]][line[2]]["title"]
            })
            
    return run

def load_bioasq_format(file, maps=None):
    """
    Load the BioASQ format file and apply any a mapping list if needed
    """
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)["questions"]
        if maps:
            for query in data:
                for old_key,new_key in maps:
                    query[new_key] = query.pop(old_key)
                    
    return data


"""
Load the BioASQ query and apply any a mapping list if needed

For backward compatibility
"""   
load_queries = load_bioasq_format

def subset_byId(data, set_ids):
    return [ x for x in data if x["id"] in set_ids]
    

def dir_tree_run(action, base_dir):
    """
    Apply funcition "action" to the individual files from tree directory
    """
    _temp_f_name = ""
    for f_name in os.listdir(base_dir):
        _temp_f_name = os.path.join(base_dir,f_name)
        if os.path.isdir(_temp_f_name):
            dir_tree_run(action,_temp_f_name)
        else:
            action(_temp_f_name)

            
def process_open_xml(proc_id, xml_files, output_dir):
    import pubmed_parser as pp
    
    def filter_mesh(string):
        return " ".join(map(lambda y:y[0], map(lambda x: x.split(";"), string.split(":")[1:])))
    
    print("[Process-{}] Started".format(proc_id))
    articles = []
    for file_name in xml_files:
        print(proc_id, file_name)
        try:
            articles.extend(pp.parse_medline_xml(file_name, year_info_only=False, nlm_category=False))
        except etree.XMLSyntaxError:
            print("Error on File " + file_name)
        
        gc.collect()
            
    articles_filter = filter(lambda x: (x["abstract"] is not None and len(x["abstract"])>0 and x["pubdate"] != ""), articles)

    articles_mapped = list(map(lambda x:{"id":x["pmid"],
                                         "title":x["title"],
                                         "abstract":x["abstract"],
                                         "keywords":x["keywords"],
                                         "pubdate":x["pubdate"],
                                         "mesh_terms":filter_mesh(x["mesh_terms"]),
                                         "delete":x["delete"]}
                               ,articles_filter))

    file_name = output_dir+"/pubmed_2019_{0:03}.p".format(proc_id)
    print("[Process-{}]: Store {}".format(proc_id, file_name))

    with open(file_name, "wb") as f:
        pickle.dump(articles_mapped, f)

    del articles
    print("[Process-{}] Ended".format(proc_id))

def multiprocess_xml_to_json(xml_files, n_process, max_store_size=int(3e6), store_path="/backup/pubmed_archive_json/"):
    from multiprocessing import Process
    
    total_files = len(xml_files)
    itter = total_files//n_process
    
    tmp_path = tempfile.mkdtemp()
    process = []
    
    try:
        
        for _i,i in enumerate(range(0, total_files, itter)):
            process.append(Process(target=process_open_xml, args=(_i, xml_files[i:i+itter], tmp_path)))

        print("[MULTIPROCESS LOOP] Starting", n_process, "process")
        for p in process:
            p.start()

        print("[MULTIPROCESS LOOP] Wait", n_process, "process")
        for p in process:
            p.join()

        del process
        gc.collect()
           
        ## merge 
        resulting_files = sorted(os.listdir(tmp_path))
        articles = []
        for file in resulting_files:
            with open(os.path.join(tmp_path, file), "rb") as f:
                articles.extend(pickle.load(f))

        # batch save
        
        size = len(articles)
        print(size)
        itter = max_store_size

        for i in range(0, size, itter):
            file_name = store_path+"/pubmedMedline_2019_{0:08}_to_{1:08}".format(i, min(size, i+itter))
            print("Save file",file_name,":",end="")
            json.dump(articles[i:i+itter], open(file_name,"w"))
            print("Done")



    except Exception as e:
        raise e

    finally:
        shutil.rmtree(tmp_path)
    

            
def multiprocess_xml_read(xml_files, n_process, max_store_size=int(3e6), store_path="/backup/pubmed_archive_json/", open_fn=process_open_xml):
    
    from multiprocessing import Process
    
    total_files = len(xml_files)
    itter = total_files//n_process
    
    tmp_path = tempfile.mkdtemp()
    process = []
    
    try:
        
        for _i,i in enumerate(range(0, total_files, itter)):
            process.append(Process(target=open_fn, args=(_i, xml_files[i:i+itter], tmp_path)))

        print("[MULTIPROCESS LOOP] Starting", n_process, "process")
        for p in process:
            p.start()

        print("[MULTIPROCESS LOOP] Wait", n_process, "process")
        for p in process:
            p.join()

        del process
        gc.collect()
           
        ## merge 
        resulting_files = sorted(os.listdir(tmp_path))
        articles = []
        for file in resulting_files:
            with open(os.path.join(tmp_path, file), "rb") as f:
                articles.extend(pickle.load(f))

    except Exception as e:
        raise e

    finally:
        shutil.rmtree(tmp_path)
    
    return articles
            
def collection_iterator(file_name, f_map=None):
    return collection_iterator_fn(file_name=file_name, f_map=f_map)()

def collection_iterator_fn(file_name, f_map=None):
    
    reader = codecs.getreader("ascii")
    tar = tarfile.open(file_name)

    print("[CORPORA] Openning tar file", file_name)

    members = tar.getmembers()
    
    def generator():
        for m in members:
            print("[CORPORA] Openning tar file {}".format(m.name))
            f = tar.extractfile(m)
            articles = json.load(reader(f))
            if f_map is not None:
                articles = list(map(f_map, articles))
            yield articles
            f.close()
            del f
            gc.collect()
    return generator




def create_filter_query_function():
    if sys.version_info < (3,):
        maketrans = string.maketrans
    else:
        maketrans = str.maketrans
    filters = '+-=&|><!(){}[]^"~*?:\/'
    tab = maketrans(filters, " "*(len(filters)))
    
    def f(query_string):
        return query_string.translate(tab)
    return f



def to_date(_str):
    for fmt in ("%Y-%m", "%Y-%m-%d", "%Y"):
        try:
            return dt.strptime(_str, fmt)
        except ValueError:
            pass
    raise ValueError("No format found")


def execute_search(es, queries, top_n, index_name, k1=0.4, b=0.4, limit_date = None):
    
    print("Setting the k1 and b for BM25")
    change_bm25_parameters(k1, b, index_name, es)
    
    query_filter = create_filter_query_function()
    
    predictions = []
    
    if limit_date is None:
        print("The inquery limit_date will be used")

    for i, query_data in enumerate(queries):
        
        #query = query_data["query"]
        
        #query = query_filter(query_data["body"])
        query = query_data["query"]
        if "limit_date" in query_data:
            limit_date = query_data["limit_date"]
        elif limit_date is None:
            raise ValueError("Need to set or provide a limit_date")
        
            
        
        query_es = {
                      "query": {
                        "bool": {
                          "must": [
                            {
                              "query_string": {
                                "query": query_filter(query), 
                                "analyzer": "english",
                                "fields": [ "text" ]
                              }
                            },
                            {
                              "range": {
                                "pubdate": {
                                  "lte": limit_date
                                }
                              }
                            }
                          ], 
                          "filter": [], 
                          "should": [], 
                          "must_not": []
                        }
                      }
                    }
            
        retrieved = es.search(index=index_name, body=query_es, size=top_n, request_timeout=200)
        
        clean_results = list(map(lambda x: {"id":x['_source']["id"], "text":x['_source']["text"],"title":x['_source']["title"], "score":x["_score"]}, retrieved['hits']['hits']))
        
        predictions.append((query_data["id"], clean_results))
        
        if not i%20:
            print("Running query:", i, end="\r")
    
    return dict(predictions)

