using Test

using ScoreAD: score_AD, score_AD_log, reject_nonfinite

using Distributions: Normal, logpdf
using Statistics: mean
import ForwardDiff: Dual, value, derivative

"""
    BernoulliData(N, s)

`N` draws from a Bernoulli(θ) distribution, `s` of which are `1`, `N-s` are `0`.
"""
struct BernoulliData
    "Total number of observations."
    N::Int
    "Observations =1."
    s::Int
end

(data::BernoulliData)(θ) = data.s/data.N*score_AD(θ)

@testset "bernoulli" begin
    N = 105
    s = 61
    data = BernoulliData(N, s)
    θ = 0.5
    for θ in range(0.1; stop = 0.9, length = 100)
        @test data(θ) ≈ s/N
        @test derivative(data, θ) ≈ s/N * 1/θ
    end
end

"""
    NormalData(x)

A vector of draws from a Normal(0, σ) distribution.
"""
struct NormalData{T}
    x::Vector{T}
end

function (data::NormalData)(σ)
    d = Normal(0.0, σ)
    mean(x * score_AD_log(logpdf(d, x)) for x in data.x)
end

@testset "normal" begin
    σ = 1.0
    x = rand(Normal(0, σ), 100)
    data = NormalData(x)
    for σ in linspace(0.1, 5, 100)
        @test data(σ) ≈ mean(x)
        @test derivative(data, σ) ≈ mean(x * (x^2/(σ^3) - 1/σ) for x in x)
    end
end

@testset "reject nonfinite" begin
    @test (@inferred reject_nonfinite(-1.0)) ≡ -1.0
    @test (@inferred reject_nonfinite(-Inf)) ≡ -Inf
    @test (@inferred reject_nonfinite(Inf)) ≡ -Inf
    @test (@inferred reject_nonfinite(NaN)) ≡ -Inf

    @test (@inferred reject_nonfinite(Dual(-1.0, (2.0, 3.0)))) ==
        Dual(-1.0, (2.0, 3.0))

    function test_rejected_Dual(ℓ)
        r = @inferred reject_nonfinite(ℓ)
        @test value(r) ≡ oftype(value(ℓ), -Inf)
        @test typeof(r) ≡ typeof(ℓ)
    end

    test_rejected_Dual(Dual(-Inf, (2.0, 3.0)))
    test_rejected_Dual(Dual(Inf, (2.0, 3.0)))
    test_rejected_Dual(Dual(NaN, (2.0, 3.0)))
    test_rejected_Dual(Dual(1.0, (NaN, 3.0)))
    test_rejected_Dual(Dual(1.0, (2.0, -Inf)))
end
