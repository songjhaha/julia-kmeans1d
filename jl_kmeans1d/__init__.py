from jnumpy import init_jl, init_project

init_jl()
init_project(__file__)

from _kmeans1d import _jl_cluster # type: ignore

def jl_cluster(data, k):
    clusters, centroids = _jl_cluster(data, k) # type: ignore
    # julia is 1-based indices, convert to 0-based
    clusters -= 1
    return clusters, centroids
