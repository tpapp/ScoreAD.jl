__precompile__()
module ScoreAD

using ForwardDiff: Dual, value, partials

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

"""
    reject_nonfinite(ℓ)

When the likelihood `ℓ` (or its derivative, when applicable) is not finite,
replace the value with `-Inf`.

Always return the same type as `ℓ`.

# Usage

This useful when calculations using automatic differentiation of a score
function in a (finite) sample, which may deliver non-finite derivatives for
extreme values. In this case, reject the parameter.
"""
@inline reject_nonfinite(ℓ::V) where {V <: Real} =
    ifelse(isfinite(ℓ), ℓ, V(-Inf))

@inline function reject_nonfinite(ℓ::Dual{T,V}) where {T,V}
    p = partials(ℓ)
    if isfinite(value(ℓ)) && all(isfinite, p)
        ℓ
    else
        Dual{T}(V(-Inf), p)
    end
end

end # module
