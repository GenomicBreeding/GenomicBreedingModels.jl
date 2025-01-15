"""
Turing specification of Bayesian linear regression using a Gaussian prior with common variance

# Example usage
```julia
# Benchmarking
genomes = GBCore.simulategenomes(n=10, l=100)
trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=3, f_add_dom_epi=[0.9 0.01 0.00;])
tebv = GBCore.analyse(trials, max_levels = 15, max_time_per_model = 2)
phenomes = tebv.phenomes[1]
G::Matrix{Float64} = genomes.allele_frequencies
y::Vector{Float64} = phenomes.phenotypes[:, 1]
model = turing_bayesG(G, y)
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
# Check for uninferred types in the model
@code_warntype model = turing_bayesG(G, y)
# Fit
model = turing_bayesG(G, y)
# We use compile=true in AutoReverseDiff() because we do not have any if-statements in our Turing model below
rng::TaskLocalRNG = Random.seed!(123)
niter::Int64 = 1_500
nburnin::Int64 = 500
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
Turing.@model function turing_bayesG(G, y)
    # Set variance prior.
    σ² ~ Distributions.Exponential(1.0 / std(y))
    # σ² ~ truncated(Normal(0, 1.0); lower=0)
    # Set intercept prior.
    intercept ~ Turing.Flat()
    # intercept ~ Distributions.Normal(0.0, 1.0)
    # Set the priors on our coefficients.
    nfeatures = size(G, 2)
    coefficients ~ Distributions.MvNormal(Distributions.Zeros(nfeatures), I)
    # Calculate all the mu terms.
    mu = intercept .+ G * coefficients
    # Return the distrbution of the response variable, from which the likelihood will be derived
    return y ~ Distributions.MvNormal(mu, σ² * I)
end

"""
Turing specification of Bayesian linear regression using a Gaussian prior with varying variances

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
model = turing_bayesGs(G, y)
# We use compile=true in AutoReverseDiff() because we do not have any if-statements in our Turing model below
rng::TaskLocalRNG = Random.seed!(123)
niter::Int64 = 1_500
nburnin::Int64 = 500
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
Turing.@model function turing_bayesGs(G, y)
    # Set variance prior.
    σ² ~ Distributions.Exponential(1.0 / std(y))
    # Set intercept prior.
    intercept ~ Turing.Flat()
    # Set the priors on our coefficients.
    nfeatures = size(G, 2)
    s² ~ filldist(Distributions.Exponential(1.0), nfeatures)
    coefficients ~ Distributions.MvNormal(Distributions.Zeros(nfeatures), s²)
    # Calculate all the mu terms.
    mu = intercept .+ G * coefficients
    return y ~ Distributions.MvNormal(mu, σ² * I)
end

"""
Gaussian distribution with a point mass at 0.0
"""
struct NπDist <: ContinuousUnivariateDistribution
    π::Real
    μ::Real
    σ²::Real
end

"""
Sampling method for NπDist

# Examples
```
d = NπDist(0.1, 0.0, 1.0)
rand(d)
```
"""
function Distributions.rand(rng::AbstractRNG, d::NπDist)::Real
    # d = NπDist(0.1, 0.0, 1.0)
    gdist = Normal(d.μ, d.σ²)
    out::Real = 0.0
    if rand() > d.π
        out = rand(gdist)
    end
    out
end

"""
log(pdf) of NπDist

# Examples
```
d = NπDist(0.1, 0.0, 1.0)
logpdf.(d, [-1.0, 0.0, 1.0])
```
"""
function Distributions.logpdf(d::NπDist, x::Real)::Real
    # d = NπDist(0.1, 0.0, 1.0)
    gdist = Normal(d.μ, d.σ²)
    if x == 0
        return log((1.0 + d.π) * pdf(gdist, 0.0))
    else
        return logpdf(gdist, 0)
    end
end

"""
Minimum value of the NπDist distribution
"""
Distributions.minimum(d::NπDist) = -Inf

"""
Maximum value of the NπDist distribution
"""
Distributions.maximum(d::NπDist) = Inf


"""
Turing specification of Bayesian linear regression using a Gaussian prior with a point mass at zero and common variance

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
model = turing_bayesGπ(G, y)
# We use compile=true in AutoReverseDiff() because we do not have any if-statements in our Turing model below
rng::TaskLocalRNG = Random.seed!(123)
niter::Int64 = 1_500
nburnin::Int64 = 500
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
Turing.@model function turing_bayesGπ(G, y)
    # Set variance prior.
    σ² ~ Distributions.Exponential(1.0 / std(y))
    # Set intercept prior.
    intercept ~ Turing.Flat()
    # Set the priors on our coefficients
    nfeatures = size(G, 2)
    π ~ Distributions.Uniform(0.0, 1.0)
    coefficients ~ filldist(NπDist(π, 0.0, 1.0), nfeatures)
    # Calculate all the mu terms.
    mu = intercept .+ G * coefficients
    return y ~ Distributions.MvNormal(mu, σ² * I)
end

"""
Turing specification of Bayesian linear regression using a Gaussian prior with a point mass at zero and varying variances

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
model = turing_bayesGπs(G, y)
# We use compile=true in AutoReverseDiff() because we do not have any if-statements in our Turing model below
rng::TaskLocalRNG = Random.seed!(123)
niter::Int64 = 1_500
nburnin::Int64 = 500
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
Turing.@model function turing_bayesGπs(G, y, ::Type{T} = Float64) where {T<:Any}
    # Set variance prior.
    σ² ~ Distributions.Exponential(1.0 / std(y))
    # Set intercept prior.
    intercept ~ Turing.Flat()
    # Set the priors on our coefficients
    nfeatures = size(G, 2)
    π ~ Distributions.Uniform(0.0, 1.0)
    s² ~ filldist(Distributions.Exponential(1.0), nfeatures)
    coefficients = Vector{T}(undef, nfeatures)
    for i = 1:nfeatures
        coefficients[i] ~ NπDist(π, 0.0, s²[i])
    end
    # Calculate all the mu terms.
    mu = intercept .+ G * coefficients
    return y ~ Distributions.MvNormal(mu, σ² * I)
end

"""
Turing specification of Bayesian linear regression using a Laplacian prior with a common scale

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
model = turing_bayesL(G, y)
# We use compile=true in AutoReverseDiff() because we do not have any if-statements in our Turing model below
rng::TaskLocalRNG = Random.seed!(123)
niter::Int64 = 1_500
nburnin::Int64 = 500
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
Turing.@model function turing_bayesL(G, y)
    # Set variance prior.
    σ² ~ Distributions.Exponential(1.0 / std(y))
    # Set intercept prior.
    intercept ~ Turing.Flat()
    # Set the priors on our coefficients.
    nfeatures = size(G, 2)
    coefficients ~ filldist(Distributions.Laplace(0.0, 1.0), nfeatures)
    # Calculate all the mu terms.
    mu = intercept .+ G * coefficients
    return y ~ Distributions.MvNormal(mu, σ² * I)
end


"""
Turing specification of Bayesian linear regression using a Laplacian prior with varying scales

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
model = turing_bayesLs(G, y)
# We use compile=true in AutoReverseDiff() because we do not have any if-statements in our Turing model below
rng::TaskLocalRNG = Random.seed!(123)
niter::Int64 = 1_500
nburnin::Int64 = 500
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
Turing.@model function turing_bayesLs(G, y, ::Type{T} = Float64) where {T<:Any}
    # Set variance prior.
    σ² ~ Distributions.Exponential(1.0 / std(y))
    # Set intercept prior.
    intercept ~ Turing.Flat()
    # Set the priors on our coefficients.
    nfeatures = size(G, 2)
    b ~ filldist(Distributions.Exponential(1.0), nfeatures)
    coefficients = Vector{T}(undef, nfeatures)
    for i = 1:nfeatures
        coefficients[i] ~ Distributions.Laplace(0.0, b[i])
    end
    # Calculate all the mu terms.
    mu = intercept .+ G * coefficients
    return y ~ Distributions.MvNormal(mu, σ² * I)
end

"""
Laplace distribution with a point mass at 0.0
"""
struct LπDist <: ContinuousUnivariateDistribution
    π::Real
    μ::Real
    b::Real
end

"""
Sampling method for LπDist

# Examples
```
d = LπDist(0.1, 0.0, 1.0)
rand(d)
```
"""
function Distributions.rand(rng::AbstractRNG, d::LπDist)::Real
    # d = LπDist(0.1, 0.0, 1.0)
    gdist = Laplace(d.μ, d.b)
    out::Real = 0.0
    if rand() > d.π
        out = rand(gdist)
    end
    out
end

"""
log(pdf) of LπDist

# Examples
```
d = LπDist(0.1, 0.0, 1.0)
logpdf.(d, [-1.0, 0.0, 1.0])
```
"""
function Distributions.logpdf(d::LπDist, x::Real)::Real
    # d = LπDist(0.1, 0.0, 1.0)
    gdist = Laplace(d.μ, d.b)
    if x == 0
        return log((1.0 + d.π) * pdf(gdist, 0.0))
    else
        return logpdf(gdist, 0)
    end
end

"""
Minimum value of the LπDist distribution
"""
Distributions.minimum(d::LπDist) = -Inf

"""
Maximum value of the LπDist distribution
"""
Distributions.maximum(d::LπDist) = Inf


"""
Turing specification of Bayesian linear regression using a Laplacian prior with a point mass at zero and common scale

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
model = turing_bayesLπ(G, y)
# We use compile=true in AutoReverseDiff() because we do not have any if-statements in our Turing model below
rng::TaskLocalRNG = Random.seed!(123)
niter::Int64 = 1_500
nburnin::Int64 = 500
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
Turing.@model function turing_bayesLπ(G, y)
    # Set variance prior.
    σ² ~ Distributions.Exponential(1.0 / std(y))
    # Set intercept prior.
    intercept ~ Turing.Flat()
    # Set the priors on our coefficients
    nfeatures = size(G, 2)
    π ~ Distributions.Uniform(0.0, 1.0)
    coefficients ~ filldist(LπDist(π, 0.0, 1.0), nfeatures)
    # Calculate all the mu terms.
    mu = intercept .+ G * coefficients
    return y ~ Distributions.MvNormal(mu, σ² * I)
end

"""
Turing specification of Bayesian linear regression using a Laplacian prior with a point mass at zero and common scale

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
model = turing_bayesLπs(G, y)
# We use compile=true in AutoReverseDiff() because we do not have any if-statements in our Turing model below
rng::TaskLocalRNG = Random.seed!(123)
niter::Int64 = 1_500
nburnin::Int64 = 500
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
Turing.@model function turing_bayesLπs(G, y, ::Type{T} = Float64) where {T<:Any}
    # Set variance prior.
    σ² ~ Distributions.Exponential(1.0 / std(y))
    # Set intercept prior.
    intercept ~ Turing.Flat()
    # Set the priors on our coefficients
    nfeatures = size(G, 2)
    π ~ Distributions.Uniform(0.0, 1.0)
    b ~ filldist(Distributions.Exponential(1.0), nfeatures)
    coefficients = Vector{T}(undef, nfeatures)
    for i = 1:nfeatures
        coefficients[i] ~ LπDist(π, 0.0, b[i])
    end
    # Calculate all the mu terms.
    mu = intercept .+ G * coefficients
    return y ~ Distributions.MvNormal(mu, σ² * I)
end



"""
Turing specification of Bayesian linear regression using a T-distribution

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
model = turing_bayesT(G, y)
# We use compile=true in AutoReverseDiff() because we do not have any if-statements in our Turing model below
rng::TaskLocalRNG = Random.seed!(123)
niter::Int64 = 1_500
nburnin::Int64 = 500
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
Turing.@model function turing_bayesT(G, y)
    # Set variance prior.
    σ² ~ Distributions.Exponential(1.0 / std(y))
    # Set intercept prior.
    intercept ~ Turing.Flat()
    # Set the priors on our coefficients.
    nfeatures = size(G, 2)
    coefficients ~ filldist(Distributions.TDist(1.0), nfeatures)
    mu = intercept .+ G * coefficients
    return y ~ Distributions.MvNormal(mu, σ² * I)
end


"""
T-distribution with a point mass at 0.0
"""
struct TπDist <: ContinuousUnivariateDistribution
    π::Real
    df::Real
end

"""
Sampling method for TπDist

# Examples
```
d = TπDist(0.1, 1.0)
rand(d)
```
"""
function Distributions.rand(rng::AbstractRNG, d::TπDist)::Real
    # d = TπDist(0.1, 1.0)
    tdist = TDist(d.df)
    out::Real = 0.0
    if rand() > d.π
        out = rand(tdist)
    end
    out
end

"""
log(pdf) of TπDist

# Examples
```
d = TπDist(0.1, 1.0)
logpdf.(d, [-1.0, 0.0, 1.0])
```
"""
function Distributions.logpdf(d::TπDist, x::Real)::Real
    # d = TπDist(0.1, 1.0)
    tdist = TDist(d.df)
    if x == 0
        return log((1.0 + d.π) * pdf(tdist, 0.0))
    else
        return logpdf(tdist, 0)
    end
end

"""
Minimum value of the TπDist distribution
"""
Distributions.minimum(d::TπDist) = -Inf

"""
Maximum value of the TπDist distribution
"""
Distributions.maximum(d::TπDist) = Inf


"""
Turing specification of Bayesian linear regression using a T-distribution with a point mass at zero

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
model = turing_bayesTπ(G, y)
# We use compile=true in AutoReverseDiff() because we do not have any if-statements in our Turing model below
rng::TaskLocalRNG = Random.seed!(123)
niter::Int64 = 1_500
nburnin::Int64 = 500
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
Turing.@model function turing_bayesTπ(G, y)
    # Set variance prior.
    σ² ~ Distributions.Exponential(1.0 / std(y))
    # Set intercept prior.
    intercept ~ Turing.Flat()
    # Set the priors on our coefficients
    nfeatures = size(G, 2)
    π ~ Distributions.Uniform(0.0, 1.0)
    coefficients ~ filldist(TπDist(π, 1.0), nfeatures)
    # Calculate all the mu terms.
    mu = intercept .+ G * coefficients
    return y ~ Distributions.MvNormal(mu, σ² * I)
end
