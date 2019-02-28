"""

_graph.py


"""
import holoviews as hv


def edgelist_to_chord(edges, nodes=None, fromcol="from", tocol="to", mincount=5):
    """
    Input edge and node dataframes and draw a Chord diagram.
    """
    edges = edges.merge(nodes,left_on="from", right_on="name")
    if nodes is not None:
        nodes = hv.Dataset(nodes, "name")
        chord = hv.Chord((edges,nodes)).select(count=(mincount,None))
    
        return chord.opts(hv.opts.Chord(cmap="Category20", edge_color=hv.dim("type").str(),
                        labels="name", node_color=hv.dim("type").str()))