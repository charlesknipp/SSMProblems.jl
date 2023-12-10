"""
    Common concrete implementations of Particle types for Particle Filter kernels.
"""
module Utils

abstract type Node{T} end

struct Root{T} <: Node{T} end
Root(T) = Root{T}()
Root() = Root(Any)

"""
    Particle{T}

Particle as immutable LinkedList. 
"""
struct Particle{T} <: Node{T}
    parent::Node{T}
    state::T
end

Particle(state::T) where {T} = Particle(Root(T), state)

Base.show(io::IO, p::Particle{T}) where {T} = print(io, "Particle{$T}($(p.state))")

"""
    ParticleContainer{T}

ParticleContainer is a weighted collection of Particles
"""
mutable struct ParticleContainer{T<:Particle}
    vals::Vector{T}
    log_weights::Vector{Float64}
end

function ParticleContainer(particles::Vector{<:Particle})
    return ParticleContainer(particles, zeros(length(particles)))
end

Base.collect(pc::ParticleContainer) = pc.vals
Base.length(pc::ParticleContainer) = length(pc.vals)
Base.keys(pc::ParticleContainer) = LinearIndices(pc.vals)

Base.@propagate_inbounds Base.getindex(pc::ParticleContainer, i::Int) = pc.vals[i]
Base.@propagate_inbounds Base.getindex(pc::ParticleContainer, i::Vector{Int}) = pc.vals[i]
Base.setindex!(pc::ParticleContainer{T}, p::T, i::Int) where T = Base.setindex!(pc.vals, p, i)

"""
    linearize(particle)

Return the trace of a particle, i.e. the sequence of states from the root to the particle.
"""
function linearize(particle::Particle{T}) where {T}
    trace = T[]
    current = particle
    while !isa(current, Root)
        push!(trace, current.state)
        current = current.parent
    end
    return trace
end

end
