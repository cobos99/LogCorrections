module HeisenbergMPS

    using ITensors, ITensorMPS
    using Printf
    using HDF5

    include("NeelStates.jl")
    using .NeelStates: neelMPS

    Sites = Vector{Index{Vector{Pair{QN, Int64}}}}

    function XXZMPO(sites::Sites, delta::Real, global_neg::Bool)
        sign = (-1)^global_neg
        os = OpSum()
        nsites = length(sites)
        # Hamiltonian bulk terms
        for n in 1:nsites-1
            os += sign,"X",n,"X",n+1
            os += sign,"Y",n,"Y",n+1
            os += sign*delta,"Z",n,"Z",n+1
        end
        # Periodic boundary conditions
        os += sign,"X",1,"X",nsites
        os += sign,"Y",1,"Y",nsites
        os += sign*delta,"Z",1,"Z",nsites
        H = MPO(os, sites)
        return H
    end

    function XXZMPO(nsites::Int, delta::Real, global_neg::Bool)
        sites = siteinds("Qubit", nsites; conserve_qns=true)
        return XXZMPO(sites, delta, global_neg)
    end

    function get_sweeps(;maxdim::Union{Int, Vector{Int}, Nothing}=nothing, mindim::Union{Int, Vector{Int}, Nothing}=nothing,
                        cutoff::Union{Real, Vector{Real}, Nothing}=nothing, noise::Union{Real, Vector{Real}, Nothing}=nothing)
        arguments = [maxdim, mindim, cutoff, noise]
        argnames = ["maxdim", "mindim", "cutoff", "noise"]
        maxlen = 0
        vector_lengths::Vector{Int} = []
        indices::Vector{Int} = []
        firstrow::Vector{String} = []
        for (i, arg) in pairs(arguments)
            if arg === nothing
                continue
            else
                if length(arg) > maxlen
                    maxlen = length(arg)
                end
                if length(arg) > 1
                    push!(vector_lengths, length(arg))
                end
                push!(firstrow, argnames[i])
                push!(indices, i)
            end
        end
        if length(unique(vector_lengths)) > 1
            throw(ArgumentError("All provided vector arguments must have the same length"))
        end
        sweep_matrix::Matrix{Any} = zeros(maxlen+1, length(firstrow))
        sweep_matrix[1, :] = firstrow
        for (j, i) in pairs(indices)
            sweep_matrix[2:end, j] .= arguments[i]
        end
        return Sweeps(sweep_matrix)
    end

    function dmrg(nsites::Int, delta::Real, global_neg::Bool, sweeps::Sweeps; random_init::Bool=false, folderpath::Union{String, Nothing}=nothing)
        if folderpath !== nothing
            if isdir(folderpath)
                filepath = joinpath(folderpath, get_mps_filename(nsites, delta, global_neg, sweeps, random_init))
                if isfile(filepath)
                    psi = loadMPS(filepath)
                    sites = siteinds(psi)
                    H = XXZMPO(sites, delta, global_neg)
                    energy = real(inner(psi', H, psi))
                    return energy, psi
                end
            else
                throw(ArgumentError("folderpath does not exist"))
            end
        end
        sites = siteinds("Qubit", nsites; conserve_qns=true)
        H = XXZMPO(sites, delta, global_neg)
        min_bond_dimension = minimum(sweeps.mindim)
        if random_init
            psi0 = randomMPS(sites, min_bond_dimension)
        else
            psi0 = neelMPS(sites)
        end
        energy, psi = ITensorMPS.dmrg(H, psi0, sweeps)
        if folderpath !== nothing
            saveMPS(filepath, psi)
        end
        return energy, psi
    end

    function get_mps_filename(nsites::Int, delta::Real, global_neg::Bool,
                              sweeps::Sweeps, random_init::Bool)::String
        max_bdim = maximum(sweeps.maxdim)
        min_cutoff = minimum(sweeps.cutoff)
        max_noise = maximum(sweeps.noise)
        min_bdim = maximum(sweeps.mindim)
        return @sprintf("heisemberg_mps_N_%i_delta_%.04f_gsign_%s_random_%s_maxbdim_%i_minbdim_%i_cutoff_%.02e_noise_%.02e.h5", 
                        nsites, delta, global_neg, random_init, max_bdim, min_bdim, min_cutoff, max_noise)
    end

    function saveMPS(filepath::String, psi::MPS)::Nothing
        h5open(filepath, "w") do f
            write(f, splitdir(filepath)[2][begin:end-3], psi)
        end
    end
    
    function loadMPS(filepath::String)::MPS
        h5open(filepath, "r") do f
            psi = read(f, splitdir(filepath)[2][begin:end-3], MPS)
            return psi
        end
    end
end