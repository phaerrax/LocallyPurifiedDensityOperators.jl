using ITensors.SiteTypes

import ..ITensors: data

struct LPDO
    data::Matrix{<:ITensor}
end

"""
    ITensors.data(::LPDO)

Returns a view of the Matrix storage of an LPDO.

This is not exported and mostly for internal usage.
"""
ITensors.data(v::LPDO) = v.data

# Utility functions, adapted from the AbstractMPS equivalent methods
Base.getindex(v::LPDO, n...) = getindex(data(v), n...)
Base.isassigned(v::LPDO, n...) = isassigned(data(v), n...)
# These methods also automagically work with a CartesianIndex instead of two Integers, i.e.
#   v[CartesianIndex(1,2)]

Base.eachindex(v::LPDO) = CartesianIndices(data(v))
Base.length(v::LPDO) = size(data(v), 1)

# No empty constructor can be defined if we want to use matrices to store the data: arrays
# with dimension > 1 cannot be resized, so we'd need to specify the size at construction.

"""
    LPDO(N::Int)

Construct an LPDO with N sites with default constructed
ITensors.
"""
function LPDO(N::Int)
    return LPDO(Matrix{ITensor}(undef, N, 2))
end

function _fill_hlinkdims(linkdims::Matrix{<:Integer}, sites::Vector{<:Index})
    @assert size(linkdims) == (length(sites) - 1, 2)
    return linkdims
end

function _fill_hlinkdims(linkdims::Vector{<:Integer}, sites::Vector{<:Index})
    @assert length(linkdims) == length(sites) - 1
    ls = Matrix{Int}(undef, length(sites) - 1, 2)
    ls[:, 1] .= linkdims
    ls[:, 2] .= linkdims
    return ls
end

function _fill_hlinkdims(linkdims::Integer, sites::Vector{<:Index})
    return fill(linkdims, length(sites) - 1, 2)
end

function _fill_vlinkdims(linkdims::Vector{<:Integer}, sites::Vector{<:Index})
    @assert length(linkdims) == length(sites)
    return linkdims
end

function _fill_vlinkdims(linkdims::Integer, sites::Vector{<:Index})
    return fill(linkdims, length(sites))
end

linkstring(j1::Integer, j2::Integer) = string(j1, "/", j2)
linkstring(j::CartesianIndex{2}) = linkstring(Tuple(j)...)

"""
    LPDO([::Type{ElT} = Float64, ]sites; hlinkdims=1, vlinkdims=1)

Construct an LPDO filled with Empty ITensors of type `ElT` from a collection of indices.

Optionally specify the link dimensions with the keyword arguments `h/vlinkdims`, which by default are 1.
"""
function LPDO(
    ::Type{T},
    sites::Vector{<:Index};
    hlinkdims::Union{Integer,Vector{<:Integer},Matrix{<:Integer}}=1,
    vlinkdims::Union{Integer,Matrix{<:Integer}}=1,
) where {T<:Number}
    N = length(sites)
    lpdo_data = Matrix{ITensor}(undef, N, 2)

    _vlinkdims = _fill_vlinkdims(vlinkdims, sites)
    vspaces = if hasqns(sites)
        [[QN() => vld] for vld in _vlinkdims]
    else
        _vlinkdims
    end
    vlinks = [Index(vspaces[i], "vLink,l=$i") for i in 1:N]

    if N == 1
        lpdo_data[1, 1] = ITensor(T, sites[1], dag(vlinks[1]))
        lpdo_data[2, 1] = ITensor(T, vlinks[1], dag(sites[1]'))
        return LPDO(lpdo_data)
    end

    _hlinkdims = _fill_hlinkdims(hlinkdims, sites)
    hspaces = if hasqns(sites)
        [[QN() => hld] for hld in _hlinkdims]
    else
        _hlinkdims
    end
    hlinks = [
        Index(hspaces[j], string("hLink,l=", linkstring(j))) for
        j in CartesianIndices(hspaces)
    ]

    for i in eachindex(sites)
        s = sites[i]
        if i == 1
            lpdo_data[i, 1] = ITensor(T, hlinks[i, 1], dag(vlinks[i]), s)
            lpdo_data[i, 2] = ITensor(T, vlinks[i], hlinks[i, 2], dag(s'))
        elseif i == N
            lpdo_data[i, 1] = ITensor(T, dag(hlinks[i - 1, 1]), dag(vlinks[i]), s)
            lpdo_data[i, 2] = ITensor(T, vlinks[i], dag(hlinks[i - 1, 2]), dag(s'))
        else
            lpdo_data[i, 1] = ITensor(
                T, dag(hlinks[i - 1, 1]), s, hlinks[i, 1], dag(vlinks[i])
            )
            lpdo_data[i, 2] = ITensor(
                T, vlinks[i], dag(hlinks[i - 1, 2]), dag(s'), hlinks[i, 2]
            )
        end
    end
    return LPDO(lpdo_data)
end

LPDO(sites::Vector{<:Index}, args...; kwargs...) = LPDO(Float64, sites, args...; kwargs...)

function Base.setindex!(v::LPDO, T::ITensor, n...)
    # This also automagically works with a CartesianIndex instead of two Integers, i.e.
    #   v[CartesianIndex(1,2)]
    data(v)[n...] = T
    return v
end

function Base.setindex!(v::LPDO, w::LPDO, ::Colon)
    # Ma così non si rischia di sostituire in `v` un blocco di tensori disconnessi dal resto
    # di quelli già presenti?
    data(v)[:] = data(w)
    return v
end

"""
    copy(::LPDO)

Make a shallow copy of an LPDO. By shallow copy, it means that a new LPDO is returned, but
the data of the tensors are still shared between the returned LPDO and the original LPDO.

Therefore, replacing an entire tensor of the returned LPDO will not modify the input LPDO,
but modifying the data of the returned LPDO will modify the input LPDO.

Use [`deepcopy`](@ref) for an alternative that copies the ITensors as well.
"""
Base.copy(v::LPDO) = LPDO(copy(data(v)))

Base.similar(v::LPDO) = LPDO(similar(data(v)))

"""
    deepcopy(::LPDO)

Make a deep copy of an LPDO. By deep copy, it means that a new LPDO is returned that doesn't
share any data with the input LPDO.

Therefore, modifying the resulting LPDO will note modify the original LPDO.

Use [`copy`](@ref) for an alternative that performs a shallow copy that avoids copying the
ITensor data.
"""
Base.deepcopy(v::LPDO) = LPDO(copy.(data(v)))

"""
    LPDO(x::MPS)

Construct an LPDO representing the pure state ``|x⟩⟨x|``.
"""
function LPDO(x::MPS)
    xnorm = norm(x)
    N=length(x)
    lpdo_data = Matrix{ITensor}(undef, N, 2)
    lpdo_data[:, 1] = x.data
    lpdo_data[:, 2] = dag(x').data

    vspaces = if hasqns(x)
        [[QN() => 1] for _ in 1:N]
        # Note: here we cannot use `fill` because every location of the returned array is
        # set to (and is thus === to) the value that is passed to the function. We need to
        # use a comprehension instead, in order to create an array of many independent inner
        # arrays.
    else
        ones(Int, N)
    end
    vlinks = [Index(vspaces[i], "vLink,l=$i") for i in 1:N]

    for i in 1:N
        lpdo_data[i, 1] *= onehot(dag(vlinks[i]) => 1)
        lpdo_data[i, 2] *= onehot(vlinks[i] => 1)
    end

    for i in 1:(N - 1)
        replacetags!(lpdo_data[i, 1], "Link,l=$i", string("hLink,l=", linkstring(i, 1)))
        replacetags!(lpdo_data[i, 2], "Link,l=$i", string("hLink,l=", linkstring(i, 2)))
    end
    for i in 2:N
        replacetags!(
            lpdo_data[i, 1], "Link,l=$(i-1)", string("hLink,l=", linkstring(i-1, 1))
        )
        replacetags!(
            lpdo_data[i, 2], "Link,l=$(i-1)", string("hLink,l=", linkstring(i-1, 2))
        )
    end

    return LPDO(lpdo_data)
end

#
# Printing functions
#

function Base.show(io::IO, v::LPDO)
    print(io, "LPDO")
    (length(v) > 0) && print(io, "\n")
    for i in eachindex(v)
        if !isassigned(v, i)
            println(io, "#undef")
        else
            A = v[i]
            if order(A) != 0
                println(io, "[", linkstring(i), "] $(inds(A))")
            else
                println(io, "[", linkstring(i), "] ITensor()")
            end
        end
    end
end

"""                                                               
    siteind(v::LPDO, j::Integer; kwargs...)          
                                                                  
Return the site Indices found of the LPDO at the site `j` as an IndexSet.                                   
Optionally filter prime tags and prime levels with keyword arguments like `plev` and `tags`.                         
"""
function ITensors.SiteTypes.siteind(v::LPDO, j::Integer; kwargs...)
    N = length(v)
    si = if N == 1
        uniqueinds(v[1, 1], v[1, 2]; kwargs...)
    else
        if j == 1
            uniqueinds(v[j, 1], v[j + 1, 1], v[j, 2]; kwargs...)
        elseif j == N
            uniqueinds(v[j, 1], v[j - 1, 1], v[j, 2]; kwargs...)
        else
            uniqueinds(v[j, 1], v[j - 1, 1], v[j + 1, 1], v[j, 2]; kwargs...)
        end
    end
    return only(si)
end

function ITensors.SiteTypes.siteinds(v::LPDO; kwargs...)
    return [siteind(v, j; kwargs...) for j in 1:length(v)]
end

function vlinkind(v::LPDO, j::Integer; kwargs...)
    return commonind(v[j, 1], v[j, 2]; kwargs...)
end

function vlinkinds(v::LPDO; kwargs...)
    return [vlinkind(v, j; kwargs...) for j in 1:length(v)]
end

function hlinkind(v::LPDO, j::CartesianIndex{2}; kwargs...)
    return commonind(v[j], v[j + CartesianIndex(1, 0)]; kwargs...)
end

function hlinkind(v::LPDO, j1::Integer, j2::Integer; kwargs...)
    return hlinkind(v, CartesianIndex(j1, j2); kwargs...)
end

function hlinkinds(v::LPDO; kwargs...)
    hshape = _fill_hlinkdims(1, siteinds(v))
    # We call `_fill_hlinkdims` so that it gives us the correct matrix indices to refer
    # to the hlinks within the LDPO.
    return [hlinkind(v, j; kwargs...) for j in CartesianIndices(hshape)]
end

function LinearAlgebra.tr(v::LPDO)
    r = ITensors.OneITensor()
    s = siteinds(v)
    for i in 1:length(v)
        r *= v[i, 1] * v[i, 2] * delta(dag(s[i]),s[i]')
    end
    return scalar(r)
end
