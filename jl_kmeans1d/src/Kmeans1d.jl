module Kmeans1d

using TyPython.CPython

# wrote by pabloferz and Raf in https://discourse.julialang.org/t/c-code-much-faster-than-julia-how-can-i-optimize-it/87868
# which is a translation of C++ code https://github.com/dstein64/kmeans1d/blob/master/kmeans1d/_core.cpp
# modify(songjhaha): use Int insteat of Int32; support AbstractFloat; return clusters first

function smawk!(result, rows, cols, allocs, lookup, k)
    length(rows) == 0 && return result

    (; cols_alloc, odd_rows, col_idx_lookup) = allocs[1]
    ## REDUCE
    ncols = 0
    @inbounds for col in cols
        while true
            ncols == 0 && break
            row = rows[ncols]
            if lookup(k, row, col) >= lookup(k, row, cols_alloc[ncols])
                break
            end
            ncols -= 1
        end
        if ncols < length(rows)
            ncols += 1
            cols_alloc[ncols] = col
        end
    end
    _cols = view(cols_alloc, Base.OneTo(ncols))

    # Call recursively on odd-indexed rows
    @inbounds for i in 2:2:length(rows)
        odd_rows[i >> 1] = rows[i]
    end

    smawk!(result, odd_rows, _cols, view(allocs, 2:lastindex(allocs)), lookup, k)

    @inbounds for idx in 1:ncols
        col_idx_lookup[_cols[idx]] = idx
    end

    ## INTERPOLATE

    # Fill-in even-indexed rows
    start = 1
    @inbounds for r in 1:2:length(rows)
        row = rows[r]
        stop = length(_cols) - 1
        if r < (length(rows) - 1)
            stop = col_idx_lookup[result[rows[r + 1]]]
        end
        argmin = _cols[start]
        min = lookup(k, row, argmin)
        for c in start+1:stop+1
            value = lookup(k, row, _cols[c])
            if (c == start) || (value < min)
                argmin = _cols[c]
                min = value
            end
        end
        result[row] = argmin
        start = stop
    end

    return result
end

struct CostCalculator{T<:AbstractFloat}
    cumsum::Vector{T}
    cumsum2::Vector{T}

    function CostCalculator(array::AbstractVector{T}, n::Integer) where {T<:AbstractFloat}
        cumsum = zeros(T, n+1)
        cumsum2 = zeros(T, n+1)
        @inbounds for i in 1:n
            x = array[i]
            cumsum[i+1] = x + cumsum[i]
            cumsum2[i+1] = x * x + cumsum2[i]
        end
        return new{T}(cumsum, cumsum2)
    end
end

function calc(cc::CostCalculator, i, j)
    if j < i
        return zero(eltype(cc.cumsum))
    end

    @inbounds begin
        mu = (cc.cumsum[j + 1] - cc.cumsum[i]) / (j - i + 1)
        result = cc.cumsum2[j + 1] - cc.cumsum2[i]
        result += (j - i + 1) * (mu * mu)
        result -= (2 * mu) * (cc.cumsum[j + 1] - cc.cumsum[i])
    end

    return result
end


struct LookUp{T<:AbstractFloat}
    calculator::CostCalculator{T}
    D::Matrix{T}
end

function (lu::LookUp)(k, i, j)
    col = min(i, j - 1)
    if col == 0
        col = size(lu.D, 2) + col
    end
    return @inbounds lu.D[k - 1, col] + calc(lu.calculator, j, i)
end

function _cluster(array::AbstractVector{S}, n, k)  where {S<:AbstractFloat}
    # Sort input array and save info for de-sorting
    sort_idx = sortperm(array)
    undo_sort_lookup = Vector{Int}(undef, n)
    sorted_array = Vector{S}(undef, n)

    @inbounds for i in 1:n
        sorted_array[i] = array[sort_idx[i]]
        undo_sort_lookup[sort_idx[i]] = i
    end

    #Set D and T using dynamic programming algorithm
    cost_calculator = CostCalculator(sorted_array, n)
    D = Matrix{S}(undef, k, n)
    T = Matrix{Int}(undef, k, n)
    lookup = LookUp(cost_calculator, D)

    @inbounds for i in 1:n
        D[1, i] = calc(cost_calculator, 1, i)
        T[1, i] = 1
    end

    row_argmins = Vector{Int}(undef, n)
    rows = 1:n
    cols = 1:n
    allocs = NamedTuple{(:odd_rows,:cols_alloc,:col_idx_lookup),Tuple{Vector{Int},Vector{Int},Vector{Int}}}[]
    l = length(rows)
    while true
        l == 0 && break
        odd_rows = Vector{Int}(2:2:l)
        cols_alloc = zeros(Int, l)
        col_idx_lookup = Vector{Int}(undef, length(rows))
        push!(allocs, (; odd_rows, cols_alloc, col_idx_lookup))
        l ÷= 2
    end

    for k_ in 2:k
        smawk!(row_argmins, rows, cols, allocs, lookup, k_)
        @inbounds for i in 1:n
            argmin = row_argmins[i]
            min = lookup(k_, i, argmin)
            D[k_, i] = min
            T[k_, i] = argmin
        end
    end

    #Extract cluster assignments by backtracking
    centroids = zeros(S, k)
    sorted_clusters = Vector{Int}(undef, n)
    t = n + 1
    k_ = k
    n_ = n

    @inbounds while t > 1
        t_ = t
        t = T[k_, n_]
        centroid = 0.0
        for i in t:t_-1
            sorted_clusters[i] = k_
            centroid += (sorted_array[i] - centroid) / (i - t + 1)
        end
        centroids[k_] = centroid
        k_ -= 1
        n_ = t - 1
    end

    clusters = Vector{Int}(undef, n)
    @inbounds for i in 1:n
        clusters[i] = sorted_clusters[undo_sort_lookup[i]]
    end

    return clusters, centroids
end

@export_py function cluster(array::Vector, k::Int)::Tuple{Vector{Int}, Vector}
    n = length(array)
    return _cluster(array, n, min(k, n))
end

function init()
    @export_pymodule _kmeans1d begin
        _jl_cluster = Pyfunc(cluster)
    end
end

precompile(init, ())

end  # module
