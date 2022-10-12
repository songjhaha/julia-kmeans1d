# julia-kmeans1d

A Python library with an implementation of k-means clustering on 1D data.

The core algorithm is wrote on julia by pabloferz and Raf in [discourse](https://discourse.julialang.org/t/c-code-much-faster-than-julia-how-can-i-optimize-it/87868). Which is a translation version of [C++ code](https://github.com/dstein64/kmeans1d/blob/master/kmeans1d/_core.cpp).

Here we wrap julia function as python function with [jnumpy](https://github.com/Suzhou-Tongyuan/jnumpy)

## Benchmark

```
from jl_kmeans1d import jl_cluster
from kmeans1d import cluster
import numpy as np

X1 = np.random.rand(1000)
%timeit jl_cluster(X1, 32) # 2.18 ms
%timeit cluster(X1, 32) # 6.15 ms

X2 = np.random.rand(1000000)
%timeit jl_cluster(X2, 32) # 4.68s
%timeit cluster(X2, 32) # 9.38s
```