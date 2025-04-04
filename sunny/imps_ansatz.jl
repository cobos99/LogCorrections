using LinearAlgebra
using Combinatorics

function amplitude(L::Int64, alpha::Float64, conf::Vector{Int64})::ComplexF64
    z = exp(2im * pi /L)
    N = length(conf)
    res = 1
    for i = 1:N-1
        for j = i+1:N
            res *= (z^conf[j] - z^conf[i])
        end
    end
    return res^(4 * alpha)
end

function imps_normalization(L::Int, N::Int, alpha::Float64)::Float64
    sites = Vector(range(1, L))
    confs = [ c for c in combinations(sites, N) ]
    n = length(confs)
    # println("Total number of combinations: ", n)
    probs = zeros(n)
    Threads.@threads for i in 1:n
        probs[i] = abs(amplitude(L, alpha, confs[i]))^2
    end
    norm = sum(probs)
    return norm
end
