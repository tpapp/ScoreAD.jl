module ScoreAD

import ForwardDiff: Dual, partials

export score_AD, score_AD_log

"""
    score_AD(p)

When `p` is real, return `1` (with the same type).

When `p` is a primitive for automatic differentiation, return ``log ∂p``, ie the
score function.

This function can be used for gradient estimation using simulation. See the
references in the README.
"""
score_AD(p::Real) = one(p)

score_AD(p::Dual{T,V}) where {T,V} = Dual{T}(one(V), partials(log(p)))

"""
    score_AD_log(ℓ)

Similar to [`score_AD`](@ref), but uses the log likelihood, ie
```julia
score_AD(p) == score_AD_log(log(p))
```
"""
score_AD_log(ℓ::Real) = one(ℓ)

score_AD_log(ℓ::Dual{T,V}) where {T,V} = Dual{T}(one(V), partials(ℓ))

end # module
