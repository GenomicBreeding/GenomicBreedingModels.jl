"""
Turing specification of Bayes A linear regression
"""
Turing.@model function turing_bayesA(G, y)
    # Set variance prior.
    σ² ~ truncated(Distributions.Normal(0, 100); lower=0)
    # Set intercept prior.
    intercept ~ Distributions.Normal(0, 100)
    # Set the priors on our coefficients.
    nfeatures = size(G, 2)
    coefficients ~ Distributions.MvNormal(Distributions.Zeros(nfeatures), 10.0 * I)
    # Calculate all the mu terms.
    mu = intercept .+ G * coefficients
    return y ~ Distributions.MvNormal(mu, σ² * I)
end








Turing.@model function turing_bayesC(G, y)
    # Set variance prior.
    σ² ~ truncated(Distributions.Normal(0, 100); lower=0)
    # Set intercept prior.
    intercept ~ Distributions.Normal(0, 100)
    # Set the priors on our coefficients.
    nfeatures = size(G, 2)
    DistBayesC = Distributions.MixtureModel(
        Normal[
            Normal(0.0, 10.0),
            Normal(0.0, 0.0)
        ], [0.1, 0.9 ]
    )
    coefficients = Vector{Float64}(undef, nfeatures)
    for j in 1:nfeatures
        coefficients[j] ~ DistBayesC
    end
    # Calculate all the mu terms.
    mu = intercept .+ G * coefficients
    return y ~ Distributions.MvNormal(mu, σ² * I)
end
