"""

_graph.py


Represent colocations as a graph

"""
import pandas as pd


def _doc_ent_nodes(d):
    """
    
    """
    # find all the columns containing entities
    ent_columns = [e for e in d.columns if "ENTITY" in e]
    entities = {"name":[], "type":[]}
    # for every entity type
    for e in ent_columns:
        t = e.split("_")[-1]
        # for every list of entities
        for l in d[e].values:
            # for every element in the list:
            for ent in l:
                if ent not in entities["name"]:
                    entities["name"].append(ent)
                    entities["type"].append(t)
                    
    return pd.DataFrame(entities)


def _doc_ent_edges(d):
    """
    build a weighted edge list
    """
    # find all the columns containing entities
    ent_columns = [e for e in d.columns if "ENTITY" in e]
    
    # aggregate entities across all types
    all_ents = d[ent_columns[0]].copy()
    for e in ent_columns[1:]:
        all_ents += d[e]
        
    # find all edges
    colocations = []
    for t in all_ents:
        if len(t) > 1:
            for i in range(len(t)):
                for j in range(i):
                    colocations.append({"from":t[i], "to":t[j]})
    colocations = pd.DataFrame(colocations)

    # group and count edges across documents
    edges = colocations.groupby(["from", "to"]).size().reset_index()
    edges.columns = ["from", "to", "count"]
    
    return edges


def entity_colocation_graph(d):
    """
    Input a DataFrame containing one or more entity 
    columns; return dataframes with node and edge data
    """
    return _doc_ent_nodes(d), _doc_ent_edges(d)