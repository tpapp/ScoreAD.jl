# ScoreAD

[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](http://www.repostatus.org/badges/latest/wip.svg)](http://www.repostatus.org/#wip)
[![Build Status](https://travis-ci.org/tpapp/ScoreAD.jl.svg?branch=master)](https://travis-ci.org/tpapp/ScoreAD.jl)
[![Coverage Status](https://coveralls.io/repos/tpapp/ScoreAD.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/tpapp/ScoreAD.jl?branch=master)
[![codecov.io](http://codecov.io/github/tpapp/ScoreAD.jl/coverage.svg?branch=master)](http://codecov.io/github/tpapp/ScoreAD.jl?branch=master)

Score function method for gradient of simulated values using automatic differentiation.

## Introduction

When `x ∼ F(⋅, β)`, integrals of the form

```
s(β) = Eᵦ[h(x)] = ∫ h(x) dF(x,β)
```

can be estimated by simulating values `xᵢ ∼ F(⋅, β)` from the distribution and taking the mean, ie

```
ŝ(β) = 1/N ∑ᵢ h(xᵢ)
```

We are frequently interested in a Monte Carlo estimator for `∂s/∂θ`. The “score function” trick is to multiply and divide by the density before differentiating, and use

```
∂ŝ(β)/∂β = 1/N ∑ᵢ h(xᵢ) ∂log(f(x, β))/∂β

```
in Monte Carlo simulations. Note that various technical conditions are required for this (you need to be able to exchange the integral and the differentiation operators), see the references below.

This package implements `score_AD` and `score_AD_log` to program these seamlessly using automatic differentiation (currently [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) is supported). You can write, for example,

```
dist = SomeDistribution(β)
mean(h(x) * score_AD(pdf(dist, β)) for x in xs)
```

or (note the log, which is preferred for numerical reasons)

```
dist = SomeDistribution(β)
mean(h(x) * score_AD_log(logpdf(dist, β)) for x in xs)
```

and the resulting value calculation will “do the right thing”, and be seamlessly autodifferentiable.

See the unit tests for examples.

## References

The literature for *score function* or *likelihood ratio* methods is vast. The following are a good starting points.

- Fu, M. C. (2006). Gradient estimation. Handbooks in operations research and management science, 13, Chapter 19, 575–616.

- Rubinstein, R. Y., & Kroese, D. P. (2016). Simulation and the Monte Carlo method, Chapter 7. John Wiley \& Sons.
