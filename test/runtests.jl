using Base.Test

using ScoreAD

using Distributions
import ForwardDiff: derivative

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
    for θ in linspace(0.1, 0.9, 100)
        @test data(θ) ≈ s/N
        @test derivative(data, θ) ≈ s/N*1/θ
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

σ = 1.0
x = rand(Normal(0, σ), 100)
data = NormalData(x)

@testset "normal" begin
    for σ in linspace(0.1, 5, 100)
        @test data(σ) ≈ mean(x)
        @test derivative(data, σ) ≈ mean(x * (x^2/(σ^3) - 1/σ) for x in x)
    end
end
