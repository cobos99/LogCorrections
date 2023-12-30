using ArgParse
using Printf
include("DMRGHeisemberg.jl")
using .DMRGHeisemberg

function parse_commandline()
    #= Parse arguments for DMRG from command line =#
    s = ArgParseSettings()

    @add_arg_table! s begin
        "delta"
            help = "Delta parameter of the XXZ model"
            arg_type = Int
            required = true
        "sites"
            help = "Number of chain sites"
            arg_type = Int
            required = true
        "sweeps"
            help = "Number of DMRG sweeps"
            arg_type = Int
            required = true
        "cutoff"
            help = "Precision goal for each sweep"
            arg_type = Float64
            required = true
        "bdims"
            help = "Bond dimension for each sweep. If len(bdims) than the number of sweeps, the last value is used for the later ones"
            arg_type = Int
            action = :store_arg
            nargs = '+'
            required = true
        "--ferro", "-f"
            help = "Use the ferromagnetic hamiltonian"
            action = :store_true
        "--folder", "-s"
            help = "Folder where to save the results"
            default = nothing
    end

    return parse_args(s)
end

function main()
    if ~isinteractive()
        let
            parsed_args = parse_commandline()
            filename = @sprintf("heisemberg_mps_delta_%.04f_ferro_%s_N_%i_mbdim_%i_cutoff_%.02e.h5", parsed_args["delta"], parsed_args["ferro"], parsed_args["sites"], maximum(parsed_args["bdims"]), parsed_args["cutoff"])
            folder = parsed_args["folder"]
            if folder !== nothing && dirname(folder) == folder
                filepath = "$folder/$filename"
            else
                filepath = filename
            end
            energy, MPS = DMRGHeisemberg.hdmrg(
                parsed_args["delta"],
                ~parsed_args["ferro"],
                parsed_args["sites"],
                parsed_args["sweeps"],
                parsed_args["bdims"],
                parsed_args["cutoff"];
                filepath=filepath
            )
            print("Energy $energy")
            end
    else
        throw(SystemError("This program can only be ran in compiled mode"))
    end
end

main()