isadjacent(j, k) = (sum(abs.(Tuple(j .- k))) == 1)

"""
    apply(t::ITensor, ρ::LPDO; kwargs...)
    product([...])

Get the product ``tρ`` of the operator `t` multiplying the state `ρ` on the left.  The sites
on which `t` must act are determined by the common indices between `t` and the site indices
of `ρ`.  The site indices of `t` must be contiguous in the LPDO geometry.

# Keywords

- `cutoff::Real`: singular value truncation cutoff.
- `maxdim::Int`: maximum LPDO bond dimension.
- `apply_dag::Bool = false`: apply the gate and the adjoint of the gate on the right.
"""
function ITensorMPS.product(t::ITensor, ρ::LPDO; kwargs...)
    ns = filter(j -> j[2] == 1, findsites(ρ, t))
    nsites = length(ns)
    if nsites == 1
        return product1(t, copy(ρ), ns...; kwargs...)
    elseif nsites == 2
        if isadjacent(ns...)
            return product2(t, copy(ρ), ns...; kwargs...)
        else
            throw(error("apply not supported for tensors acting on non-adjacent sites"))
        end
    else
        throw(error("apply not supported for tensors with more than 2 site indices"))
    end
end

function product1(t::ITensor, ρ::LPDO, ns; apply_dag::Bool=false, kwargs...)
    # 1. Apply the operator to the state.
    # TODO Select the indices not by the "Site" tag but whether they are in common
    #      between `t` and `ρ`. Assuming there is "Site" tag is a bit risky.
    ρ[ns] = replaceprime(t * ρ[ns], 1 => 0; tags="Site")
    if apply_dag
        ns′ = oppositesite(ns)
        adj_t = swapprime(dag(t), 0 => 1; tags="Site")
        ρ[ns′] = replaceprime(ρ[ns′] * adj_t, 0 => 1; tags="Site")
    end
    # 2. Return the new state
    return ρ
end

function product2(t::ITensor, ρ::LPDO, ns1, ns2; apply_dag::Bool=false, kwargs...)
    # 1. Apply the operator to the state
    # TODO Select the indices not by the "Site" tag but whether they are in common
    #      between `t` and `ρ`. Assuming there is "Site" tag is a bit risky.
    newblock = replaceprime(t * ρ[ns1] * ρ[ns2], 1 => 0; tags="Site")
    # 2. Decompose the result
    #    ↳ there doesn't seem to be a reason to choose into which of the two final tensors
    #      we should incorporate the singular values, so we'll just split them evenly (by
    #      giving `factorize` the `ortho="none"` argument).
    ρ[ns1], ρ[ns2] = factorize(newblock, inds(ρ[ns1]); ortho="none", kwargs...)
    if apply_dag
        ns1′ = oppositesite(ns1)
        ns2′ = oppositesite(ns2)
        adj_t = swapprime(dag(t), 0 => 1; tags="Site")
        newblock′ = replaceprime(adj_t * ρ[ns1′] * ρ[ns2′], 0 => 1; tags="Site")
        ρ[ns1′], ρ[ns2′] = factorize(newblock′, inds(ρ[ns1′]); ortho="none", kwargs...)
    end
    # 3. Return the new state
    return ρ
end
