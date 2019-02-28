"""

_topic.py


Code for visualizing topic models

"""
import numpy as np
import pandas as pd
import holoviews as hv


from scipy.spatial.distance import cdist


def topic_bar_chart(components, num_tokens=20):
    """
    Build a HoloMap bar chart showing the top-N tokens
    for each topic.
    """
    num_topics = len(components.columns)
    topic_dict = {}

    for t in range(num_topics):
        indices = components[t].values.argsort()[-num_tokens:]
        #indices = nmf.components_[t,:].argsort()[-20:]
        topic_words = pd.DataFrame({"weight":components[t].values[indices],
                           "token":[components.index[i] for i in indices]})
        topic_dict[t] = hv.Bars(topic_words, vdims=["weight"], kdims=["token"]).opts(hv.opts.Bars(invert_axes=True))
    
    return hv.HoloMap(topic_dict, kdims="topic")






def doc_embedding_scatter_plot(d, vdims=["topic", "year", "title"]):
    """
    Visualize a dataset with T-SNE x and y columns added.
    """
    #vdims = [x for x in d.columns if x not in ["x", "y"]]
    return hv.Points(d, kdims=["x", "y"], vdims=vdims).opts(
    hv.opts.Points(cmap="Category20", color="topic", alpha=0.5, tools=["hover","box_select"],
                  xlabel="", show_frame=False, ylabel="", yaxis=None, xaxis=None))
    
    
def minimum_semantic_distance_plot(d, vectors, year="year", x="date", 
                                   vdims=["topic", "year", "title"]):
    """
    Visulize anomalies by plotting how far each document is from the
    closest point in the previous years' collection (in topic space)
    """
    years = np.sort(d[year].unique())
    mindists = np.zeros(len(d))
    for y in years[:-1]:
        prev = d["year"] == y
        nxt = d["year"] == y+1
        dists = cdist(vectors[prev], vectors[nxt]).min(axis=0)
        mindists[nxt] = dists
        
    d2 = d.copy()
    d2["mindists"] = mindists

    return hv.Points(d2, kdims=[x, "mindists"], vdims=vdims).opts(
        hv.opts.Points(cmap="Category20", color="topic", alpha=0.5, tools=["hover"],
                  ylabel="minimum semantic distance"))