module NeelStates

    using ITensors, ITensorMPS

    Sites = Vector{Index{Vector{Pair{QN, Int64}}}}

    function neelMPS(sites::Sites; roll::Int=0)
        nsites = length(sites)
        state_vec = vcat(["Up"], [isodd(n) ? "Up" : "Dn" for n in 2:nsites])
        circshift!(state_vec, roll)
        return MPS(sites, vcat(["Up"], [isodd(n) ? "Up" : "Dn" for n in 2:nsites]))
    end

    function neelMPS(nsites::Int; roll::Int=0)
        sites = siteinds("Qubit", nsites; conserve_qns=true)
        neelMPS(sites; roll=roll)
    end

    function neelprojector(sites::Sites; roll::Int=0)
        neel_state = neelMPS(sites; roll=roll)
        return projector(neel_state, cutoff=1e-15)
    end

    function neelprojector(nsites::Int, roll::Int=0)
        sites = sitesinds("Qubit", nsites; conserve_qns=true)
        return neelprojector(sites; roll=roll)
    end

    function neelprobability(mps::MPS; neel_roll::Int=0)
        projector = neelprojector(siteinds(mps); roll=neel_roll)
        return real(inner(mps', projector, mps))
    end
end