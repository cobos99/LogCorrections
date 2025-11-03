using LinearAlgebra
using Combinatorics
using Base.Threads
using Base.Iterators

function amplitude(L::Int64, alpha::Float64, conf::Vector{Int64})::ComplexF64
    z = exp(2im * pi / L)
    N = length(conf)
    res = 1
    for i = 1:N-1
        for j = i+1:N
            res *= (z^conf[j] - z^conf[i])
        end
    end
    return res^(4 * alpha)
end

function probability(L::Int64, alpha::Float64, conf::Vector{Int64})::Float64
    N = length(conf)
    res = 1
    for i = 1:N-1
        for j = i+1:N
            res *= 2 * sin(pi * (conf[j] - conf[i]) / L)
        end
    end
    return res^(8 * alpha)
end

function imps_norm(L::Int, alpha::Float64)::Float64
    sites = Vector(range(1, L))
    N = div(L, 2)
    norm = 0
    for conf in combinations(sites, N)
        norm += abs(amplitude(L, alpha, conf))^2
    end
    return norm
end

function imps_norm_multi(L::Int, alpha::Float64)::Float64
    sites = collect(1:L)
    N = div(L, 2)
    iter = combinations(sites, N)
    next = iterate(iter)
    # chunking
    n_confs = binomial(L, N)
    chunk_size = 2^24
    n_chunks = cld(n_confs, chunk_size)
    @show n_chunks
    # results on different threads
    partial_norms = zeros(nthreads())
    for chunk in 1:n_chunks
        @show chunk
        configurations = Vector{Vector{Int64}}()
        sizehint!(configurations, chunk_size)
        for i in 1:chunk_size
            next == nothing && break
            (item, state) = next
            push!(configurations, item)
            next = iterate(iter, state)
        end
        Threads.@threads for i in 1:length(configurations)
            id = Threads.threadid()
            conf = configurations[i]
            partial_norms[id] += probability(L, alpha, conf)
        end
        # trigger garbage collection
        configurations = nothing
        GC.gc()
    end
    norm = sum(partial_norms)
    return norm
end
