"""
Turing specification of Bayesian ridge regression

# Example usage
```julia
# Benchmarking
genomes = GBCore.simulategenomes(n=10, l=100)
trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=3, f_add_dom_epi=[0.9 0.01 0.00;])
tebv = GBCore.analyse(trials, max_levels = 15, max_time_per_model = 2)
phenomes = tebv.phenomes[1]
G::Matrix{Float64} = genomes.allele_frequencies
y::Vector{Float64} = phenomes.phenotypes[:, 1]
rng::TaskLocalRNG = Random.seed!(123)
model = turing_bayesRR(G, y)
benchmarks = TuringBenchmarking.benchmark_model(
    model;
    # Check correctness of computations
    check=true,
    # Automatic differentiation backends to check and benchmark
    adbackends=[:forwarddiff, :reversediff, :reversediff_compiled, :zygote]
)


# Test more loci
genomes = GBCore.simulategenomes(n=10, l=10_000)
trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=3, f_add_dom_epi=[0.9 0.01 0.00;])
tebv = GBCore.analyse(trials, max_levels = 15, max_time_per_model = 2)
phenomes = tebv.phenomes[1]
G::Matrix{Float64} = genomes.allele_frequencies
y::Vector{Float64} = phenomes.phenotypes[:, 1]
rng::TaskLocalRNG = Random.seed!(123)
# Check for uninferred types in the model
@code_warntype model = turing_bayesRR(G, y)
# Fit
model = turing_bayesRR(G, y)
niter::Int64 = 1_500
nburnin::Int64 = 500
# We use compile=true in AutoReverseDiff() because we do not have any if-statements in our Turing model below
@time chain = Turing.sample(rng, model, NUTS(nburnin, 0.5, max_depth=5, Δ_max=1000.0, init_ϵ=0.2; adtype=AutoReverseDiff(compile=true)), niter-nburnin, progress=true);
# Use the mean paramter values after 150 burn-in iterations
params = Turing.get_params(chain[150:end, :, :]);
b_hat = vcat(mean(params.intercept), mean(stack(params.coefficients, dims=1)[:, :, 1], dims=2)[:,1]);
# Assess prediction accuracy
y_pred::Vector{Float64} = hcat(ones(size(G,1)), G) * b_hat;
UnicodePlots.scatterplot(y, y_pred)
performance::Dict{String, Float64} = metrics(y, y_pred)
```
"""
Turing.@model function turing_bayesRR(G, y)
    # Set variance prior.
    σ² ~ Distributions.Exponential(1.0 / std(y))
    # Set intercept prior.
    intercept ~ Distributions.Normal(mean(y), std(y))
    # Set the priors on our coefficients.
    nfeatures = size(G, 2)
    s ~ truncated(Distributions.Normal(0, 1); lower = 0)
    coefficients ~ Distributions.MvNormal(Distributions.Zeros(nfeatures), I .* s)
    # Calculate all the mu terms.
    mu = intercept .+ G * coefficients
    return y ~ Distributions.MvNormal(mu, σ² * I)
end

dmv = Distributions.MvNormal(Distributions.Zeros(10), I)
mean(rand(dmv, 1_000), dims = 2)
var(rand(dmv, 1_000), dims = 2)

dfd = filldist(Distributions.Normal(0.0, 1.0), 10)
mean(rand(dfd, 1_000), dims = 2)
var(rand(dfd, 1_000), dims = 2)


"""
Turing specification of Bayesian LASSO regression

# Example usage
```julia
# Simulate data
genomes = GBCore.simulategenomes(n=10, l=100)
trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=3, f_add_dom_epi=[0.9 0.01 0.00;])
tebv = GBCore.analyse(trials, max_levels = 15, max_time_per_model = 2)
phenomes = tebv.phenomes[1]
# Extract genotype and phenotype data
G::Matrix{Float64} = genomes.allele_frequencies
y::Vector{Float64} = phenomes.phenotypes[:, trait_idx]
# Regress for just 200 iterations for demonstration purposes only. Use way way more iterations, e.g. 10,000.
rng::TaskLocalRNG = Random.seed!(123)
model = turing_bayesLASSO(G, y)
niter::Int64 = 1_500
nburnin::Int64 = 500
# We use compile=true in AutoReverseDiff() because we do not have any if-statements in our Turing model below
@time chain = Turing.sample(rng, model, NUTS(nburnin, 0.5, max_depth=5, Δ_max=1000.0, init_ϵ=0.2; adtype=AutoReverseDiff(compile=true)), niter-nburnin, progress=true);
# Use the mean paramter values after 150 burn-in iterations
params = Turing.get_params(chain[150:end, :, :]);
b_hat = vcat(mean(params.intercept), mean(stack(params.coefficients, dims=1)[:, :, 1], dims=2)[:,1]);
# Assess prediction accuracy
y_pred::Vector{Float64} = hcat(ones(size(G,1)), G) * b_hat;
UnicodePlots.scatterplot(y, y_pred)
performance::Dict{String, Float64} = metrics(y, y_pred)
```
"""
Turing.@model function turing_bayesLASSO(G, y)
    # Set variance prior.
    σ² ~ Distributions.Exponential(1.0 / std(y))
    # Set intercept prior.
    intercept ~ Distributions.Normal(mean(y), std(y))
    # Set the priors on our coefficients.
    nfeatures = size(G, 2)
    s ~ truncated(Distributions.Normal(0, 1); lower = 0)
    coefficients ~ filldist(Distributions.Laplace(0.0, s), nfeatures)
    # Calculate all the mu terms.
    mu = intercept .+ G * coefficients
    return y ~ Distributions.MvNormal(mu, σ² * I)
end


"""
Turing specification of Bayes A linear regression

# Example usage
```julia
# Simulate data
genomes = GBCore.simulategenomes(n=10, l=100)
trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=3, f_add_dom_epi=[0.9 0.01 0.00;])
tebv = GBCore.analyse(trials, max_levels = 15, max_time_per_model = 2)
phenomes = tebv.phenomes[1]
# Extract genotype and phenotype data
G::Matrix{Float64} = genomes.allele_frequencies
y::Vector{Float64} = phenomes.phenotypes[:, trait_idx]
# Regress for just 200 iterations for demonstration purposes only. Use way way more iterations, e.g. 10,000.
rng::TaskLocalRNG = Random.seed!(123)
model = turing_bayesA(G, y)
niter::Int64 = 1_500
nburnin::Int64 = 500
# We use compile=true in AutoReverseDiff() because we do not have any if-statements in our Turing model below
@time chain = Turing.sample(rng, model, NUTS(nburnin, 0.5, max_depth=5, Δ_max=1000.0, init_ϵ=0.2; adtype=AutoReverseDiff(compile=true)), niter-nburnin, progress=true);
# Use the mean paramter values after 150 burn-in iterations
params = Turing.get_params(chain[150:end, :, :])
b_hat = vcat(mean(params.intercept), mean(stack(params.coefficients, dims=1)[:, :, 1], dims=2)[:,1])
# Assess prediction accuracy
y_pred::Vector{Float64} = hcat(ones(size(G,1)), G) * b_hat
performance::Dict{String, Float64} = metrics(y, y_pred)
```
"""
Turing.@model function turing_bayesA(G, y)
    # Set variance prior.
    σ² ~ Distributions.Exponential(1.0 / std(y))
    # Set intercept prior.
    intercept ~ Distributions.Normal(mean(y), std(y))
    # Set the priors on our coefficients.
    df ~ truncated(Distributions.Normal(0, 1.0); lower = 0)
    nfeatures = size(G, 2)
    coefficients = Vector{Float64}(undef, nfeatures)
    for j = 1:nfeatures
        coefficients[j] ~ Distributions.TDist(df)
    end
    # Calculate all the mu terms.
    mu = intercept .+ G * coefficients
    return y ~ Distributions.MvNormal(mu, σ² * I)
end


"""
Prior distribution for Bayes B model
"""
struct PriorBayesB <: ContinuousUnivariateDistribution
    π::Real
    df::Real
end

"""
Sampling method for PriorBayesB

# Examples
```
d = PriorBayesB(0.1, 1.0)
rand(d)
```
"""
function Distributions.rand(rng::AbstractRNG, d::PriorBayesB)::Real
    # d = PriorBayesB(0.1, 1.0)
    tdist = TDist(d.df)
    out::Real = 0.0
    if rand() > d.π
        out = rand(tdist)
    end
    out
end

"""
log(pdf) of PriorBayesB

# Examples
```
d = PriorBayesB(0.1, 1.0)
logpdf.(d, [-1.0, 0.0, 1.0])
```
"""
function Distributions.logpdf(d::PriorBayesB, x::Real)::Real
    # d = PriorBayesB(0.1, 1.0)
    tdist = TDist(d.df)
    if x == 0
        return log((1.0 + d.π) * pdf(tdist, 0.0))
    else
        return logpdf(tdist, 0)
    end
end

"""
Minimum value of the PriorBayesB distribution
"""
Distributions.minimum(d::PriorBayesB) = -Inf

"""
Maximum value of the PriorBayesB distribution
"""
Distributions.maximum(d::PriorBayesB) = Inf


"""
Turing specification of Bayes B linear regression

# Example usage
```julia
# Simulate data
genomes = GBCore.simulategenomes(n=10, l=100)
trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=3, f_add_dom_epi=[0.9 0.01 0.00;])
tebv = GBCore.analyse(trials, max_levels = 15, max_time_per_model = 2)
phenomes = tebv.phenomes[1]
# Extract genotype and phenotype data
G::Matrix{Float64} = genomes.allele_frequencies
y::Vector{Float64} = phenomes.phenotypes[:, 1]
# Regress for just 200 iterations for demonstration purposes only. Use way way more iterations, e.g. 10,000.
rng::TaskLocalRNG = Random.seed!(123)
model = turing_bayesB(G, y)
niter::Int64 = 1_500
nburnin::Int64 = 500
# We use compile=true in AutoReverseDiff() because we do not have any if-statements in our Turing model below
@time chain = Turing.sample(rng, model, NUTS(nburnin, 0.5, max_depth=5, Δ_max=1000.0, init_ϵ=0.2; adtype=AutoReverseDiff(compile=true)), niter-nburnin, progress=true);
# Use the mean paramter values after 150 burn-in iterations
params = Turing.get_params(chain[150:end, :, :])

UnicodePlots.scatterplot(stack(params.intercept)[:,1])
UnicodePlots.scatterplot(stack(params.coefficients, dims=1)[1,:,1])
UnicodePlots.scatterplot(stack(params.coefficients, dims=1)[2,:,1])
UnicodePlots.scatterplot(stack(params.coefficients, dims=1)[3,:,1])

b_hat = vcat(mean(params.intercept), mean(stack(params.coefficients, dims=1)[:, :, 1], dims=2)[:,1])
# Assess prediction accuracy
y_pred::Vector{Float64} = hcat(ones(size(G,1)), G) * b_hat
performance::Dict{String, Float64} = metrics(y, y_pred)
```
"""
Turing.@model function turing_bayesB(G, y)
    # Set variance prior.
    σ² ~ Distributions.Exponential(1.0 / std(y))
    # Set intercept prior.
    intercept ~ Distributions.Normal(mean(y), std(y))
    # Set the priors on our coefficients
    df ~ truncated(Distributions.Normal(0, 1.0); lower = 0)
    π ~ Distributions.Uniform(0.0, 1.0)
    nfeatures = size(G, 2)
    coefficients = Vector{Float64}(undef, nfeatures)
    for j = 1:nfeatures
        coefficients[j] ~ PriorBayesB(π, df)
    end
    # Calculate all the mu terms.
    mu = intercept .+ G * coefficients
    return y ~ Distributions.MvNormal(mu, σ² * I)
end
