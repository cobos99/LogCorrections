include("HeisenbergMPS.jl")
include("NeelStates.jl")

using .HeisenbergMPS
using LinearAlgebra
using .NeelStates
using ArgParse
using ITensors
using Printf
using NPZ

BLAS.set_num_threads(1)
NDTensors.Strided.disable_threads()
ITensors.enable_threaded_blocksparse()

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "lengthrange"
            help = "Range of chain lengths to cover"
            nargs = 2
            arg_type = Int
            required = true
        "ndeltas"
            help = "Number of values of Δ to consider"
            arg_type = Int
            required = true
        "maxbdim"
            help = "Maximum bond dimension to consider"
            arg_type = Int
            required = true
        "nsweeps"
            help = "Number of DMRG sweeps"
            arg_type = Int
            required = true
        "--cutoff", "-c"
            help = "Cutoff for the DMRG"
            arg_type = Float64
            default = 1e-16
        "--deltarange", "-d"
            help = "Ranges of delta to cover"
            nargs = 2
            arg_type = Int
            default = [-1, 1]
        "--neg", "-n"
            help = "To use global neg in the Heisenberg hamiltonian"
            action = :store_true
        "--respath", "-r"
            help = "Folder path where to save the results file"
            arg_type = String
            default = pwd()
        "--mpsfolder", "-m"
            help = "Folder path where to save the resulting MPSs"
            arg_type = Union{String, Nothing}
            default = nothing
        "--roll"
            help = "Roll to apply to the Neel state in which the ground state is projected"
            arg_type = Int
            default = 0
    end
    return parse_args(s)
end

function first_nan_index(matrix::Array)
    for i in 1:length(matrix)
        if isnan(matrix[i])
            return i
        end
    end
    return nothing
end

function main()
    parsed_args = parse_commandline()
    results_filename = @sprintf("heisemberg_gs_energy_maxp_N_%i_%i_delta_%i_%i_ndeltas_%i_globalneg_%s_maxbdim_%i_sweeps_%i_cutoff_%.02e.npy",
                                parsed_args["lengthrange"][1], parsed_args["lengthrange"][2], parsed_args["deltarange"][1], parsed_args["deltarange"][2],
                                parsed_args["ndeltas"], parsed_args["neg"], parsed_args["maxbdim"], parsed_args["nsweeps"], parsed_args["cutoff"])
    results_filepath = joinpath(parsed_args["respath"], results_filename)
    results_dimensions = (parsed_args["lengthrange"][2] - parsed_args["lengthrange"][1] + 1, parsed_args["ndeltas"])
    if isfile(results_filepath)
        data = npzread(results_filepath)
        energies_matrix = data[1, :, :]
        probabilities_matrix = data[2, :, :]
        first_nan_el_ind = first_nan_index(energies_matrix)
        if first_nan_el_ind === nothing
            println("Found a complete results file with these settings")
            return
        else
            iteration_start = first_nan_el_ind - 1
        end
    else
        energies_matrix = fill(NaN, results_dimensions)
        probabilities_matrix = fill(NaN, results_dimensions)
        iteration_start = 0
    end

    sweeps = HeisenbergMPS.get_sweeps(maxdim=parsed_args["maxbdim"] .* ones(Int, parsed_args["nsweeps"]), cutoff=parsed_args["cutoff"])
    for L in (parsed_args["lengthrange"][1]+iteration_start-1):parsed_args["lengthrange"][2]
        for (col, Δ) in pairs(range(parsed_args["deltarange"]..., parsed_args["ndeltas"]))
            println(@sprintf("Running L = %i, Δ = %.02f", L, Δ))
            
            energy, psi = HeisenbergMPS.dmrg(L, Δ, parsed_args["neg"], sweeps;  random_init=false, folderpath=parsed_args["mpsfolder"])
            max_probability = NeelStates.neelprobability(psi, neel_roll=parsed_args["roll"])
            energies_matrix[L-parsed_args["lengthrange"][1]+1, col] = energy
            probabilities_matrix[L-parsed_args["lengthrange"][1]+1, col] = max_probability

            data = fill(NaN, (2, results_dimensions...))
            data[1, :, :] = energies_matrix
            data[2, :, :] = probabilities_matrix
            npzwrite(results_filepath, data)
        end
    end
end

if ~isinteractive()
    main()
end