module DMRGHeisemberg

    using ITensors
    using ITensors.HDF5

    function hdmrg(delta::Real, antiferro::Bool, nsites::Int, sweeps::Int, maxdim::Vector{Int}, cutoff::Real; filepath::Union{Nothing, String}=nothing)
        aferro_factor = (-1)^(antiferro <= 0)
        sites = siteinds("S=1/2", nsites)
        os = OpSum()
        # Hamiltonian bulk terms
        for n in 1:nsites-1
            os += aferro_factor,"X",n,"X",n+1
            os += aferro_factor,"Y",n,"Y",n+1
            os += delta,"Z",n,"Z",n+1
        end
        # Periodic boundary conditions
        os += aferro_factor,"X",1,"X",nsites
        os += aferro_factor,"Y",1,"Y",nsites
        os += delta,"Z",1,"Z",nsites
        H = MPO(os, sites)
        MPS0 = randomMPS(sites, maxdim[1])
        energy, psi = dmrg(H, MPS0; nsweeps=sweeps, maxdim=maxdim, cutoff=cutoff)
        if filepath !== nothing
            f = h5open(filepath, "w")
            write(f, split(basename(filepath), ".")[1], psi)
            close(f)
        end
        return energy, psi
    end
end