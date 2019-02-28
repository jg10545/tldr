"""

_util.py

General-purpose


"""
import holoviews as hv


def holomap(func, d, col, *args, **kwargs):
    """
    Wrapper to build a hv.HoloMap out of another Holoviews plotting
    function
    
    :func: function that inputs a dataframe and generates a plot
    :d: a dataframe
    :col: column of the dataframe to use as the adjustable variable. make sure
            it's something categorical with a cardinality that isn't too high!
    
    Other args and kwargs are passed to the function
    """
    val_dict = {v:func(d[d[col]==v], *args, **kwargs) for v in d[col].unique()}
    return hv.HoloMap(val_dict, kdims=col)