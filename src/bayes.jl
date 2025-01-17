try
    R"library(BGLR)"
catch
    throw(ErrorException("Please install the BGLR package in R."))
end

"""
Bayesian models using BGLR, i.e. Bayes A, Bayes B and Bayes C
"""
function bglr(;
    G::Matrix{Float64},
    y::Vector{Float64},
    model::String = ["BayesA", "BayesB", "BayesC"][1],
    response_type::String = ["gaussian", "ordinal"][1],
    n_iter::Int64 = 1_500,
    n_burnin::Int64 = 500,
    verbose::Bool = false,
)::Vector{Float64}
    # genomes = GBCore.simulategenomes(n=1_000, l=1_750, verbose=true)
    # trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.1 0.01 0.01;], verbose=true)
    # phenomes = extractphenomes(trials)
    # G::Matrix{Float64} = genomes.allele_frequencies
    # y::Vector{Float64} = phenomes.phenotypes[:, 1]
    # model=["BayesA", "BayesB", "BayesC"][1]; response_type = ["gaussian", "ordinal"][1]; n_iter=10_500; n_burnin=100; verbose=true
    prefix_tmp_out = string(
        model,
        "-tmp-out-",
        hash(string.(vcat(y, [model, response_type, n_iter, n_burnin, verbose]))),
        "-",
        Int64(round(rand() * 1_000_000)),
    )
    @rput(G)
    @rput(y)
    @rput(model)
    @rput(response_type)
    @rput(n_iter)
    @rput(n_burnin)
    @rput(prefix_tmp_out)
    @rput(verbose)
    R"ETA = list(MRK=list(X=G, model=model, saveEffects=FALSE))"
    R"sol = BGLR::BGLR(y=y, ETA=ETA, response_type=response_type, nIter=n_iter, burnIn=n_burnin, saveAt=prefix_tmp_out, verbose=verbose)"
    @rget(sol)
    b_hat = vcat(sol[:mu], sol[:ETA][:MRK][:b])
    # Clean-up
    files = readdir()
    rm.(files[match.(Regex(string("^", prefix_tmp_out)), files).!=nothing])
    # Output
    b_hat
end


"""
    bayesian(
        bglr_model::String;
        X::Matrix{Float64},
        y::Vector{Float64},
        n_burnin::Int64 = 500,
        n_iter::Int64 = 1_500,
        verbose::Bool = false,
    )::Fit

Fit a Bayesian linear regression models via BGLR in R

## Examples
```jldoctest; setup = :(using GBCore, GBModels, Suppressor)
julia> genomes = GBCore.simulategenomes(verbose=false);

julia> trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.1 0.01 0.01;], verbose=false);;

julia> phenomes = extractphenomes(trials);

julia> sol = Suppressor.@suppress bayesian("BayesA", genomes=genomes, phenomes=phenomes);

julia> sol.metrics["cor"] > 0.5
true
```
"""
function bayesian(
    bglr_model::String;
    genomes::Genomes,
    phenomes::Phenomes,
    idx_entries::Union{Nothing,Vector{Int64}} = nothing,
    idx_loci_alleles::Union{Nothing,Vector{Int64}} = nothing,
    idx_trait::Int64 = 1,
    response_type::String = ["gaussian", "ordinal"][1],
    n_burnin::Int64 = 500,
    n_iter::Int64 = 1_500,
    verbose::Bool = false,
)::Fit
    # bglr_model = "BayesA"
    # genomes = GBCore.simulategenomes()
    # trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.1 0.01 0.01;], verbose=false);
    # phenomes = extractphenomes(trials); idx_trait = 1; response_type::String = ["gaussian", "ordinal"][1];
    # idx_entries = nothing, idx_loci_alleles = nothing
    # n_burnin = 500; n_iter = 1_500; verbose = true;
    # Check arguments and extract X, y, and loci-allele names
    X, y, entries, populations, loci_alleles = extractxyetc(
        genomes,
        phenomes,
        idx_entries = idx_entries,
        idx_loci_alleles = idx_loci_alleles,
        idx_trait = idx_trait,
        add_intercept = true,
    )
    # Instantiate output Fit
    fit = Fit(n = size(X, 1), l = size(X, 2))
    fit.model = bglr_model
    fit.b_hat_labels = vcat(["intercept"], loci_alleles)
    fit.trait = phenomes.traits[idx_trait]
    fit.entries = entries
    fit.populations = populations
    fit.y_true = y
    # R-BGLR
    b_hat = bglr(
        G = X[:, 2:end],
        y = y,
        model = bglr_model,
        response_type = response_type,
        n_iter = n_iter,
        n_burnin = n_burnin,
        verbose = verbose,
    )
    # Clean-up BGLR temp files
    files = readdir()
    for i in findall(match.(r".dat\$", files) .!= nothing)
        rm(files[i])
    end
    # Assess prediction accuracy
    y_pred::Vector{Float64} = X * b_hat
    performance::Dict{String,Float64} = metrics(y, y_pred)
    if verbose
        UnicodePlots.scatterplot(y, y_pred)
        UnicodePlots.histogram(y)
        UnicodePlots.histogram(y_pred)
        println(performance)
    end
    # Output
    fit.b_hat = b_hat
    fit.y_pred = y_pred
    fit.metrics = performance
    if !checkdims(fit)
        throw(ErrorException("Error fitting " * fit.model * "."))
    end
    fit
end


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


"""
Turing specification of Bayesian logistic regression using a Gaussian prior with common variance
"""
@model function turing_bayesG_logit(G, y)
    # Set intercept prior.
    intercept ~ Turing.Flat()
    # intercept ~ Distributions.Normal(0.0, 1.0)
    # Set the priors on our coefficients.
    n, nfeatures = size(G)
    coefficients ~ Distributions.MvNormal(Distributions.Zeros(nfeatures), I)
    v = 1 ./ (1 .+ exp.(-(intercept .+ G * coefficients)) .+ 1e-20)
    for i = 1:n
        y[i] ~ Bernoulli(v[i])
    end
end

"""
    bayesian(
        turing_model::Function;
        X::Matrix{Float64},
        y::Vector{Float64},
        sampler::String = ["NUTS", "HMC", "HMCDA", "MH", "PG"][1],
        sampling_method::Int64 = 1,
        seed::Int64 = 123,
        n_burnin::Int64 = 500,
        n_iter::Int64 = 1_500,
        verbose::Bool = false,
    )::Fit

Fit a Bayesian linear regression models via Turing.jl

## Examples
```jldoctest; setup = :(using GBCore, GBModels, Suppressor)
julia> genomes = GBCore.simulategenomes(verbose=false);

julia> trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.1 0.01 0.01;], verbose=false);;

julia> phenomes = extractphenomes(trials);

julia> sol = Suppressor.@suppress bayesian(turing_bayesG, genomes=genomes, phenomes=phenomes);

julia> sol.metrics["cor"] > 0.5
true
```
"""
function bayesian(
    turing_model::Function;
    genomes::Genomes,
    phenomes::Phenomes,
    idx_entries::Union{Nothing,Vector{Int64}} = nothing,
    idx_loci_alleles::Union{Nothing,Vector{Int64}} = nothing,
    idx_trait::Int64 = 1,
    sampler::String = ["NUTS", "HMC", "HMCDA", "MH", "PG"][1],
    sampling_method::Int64 = 1,
    seed::Int64 = 123,
    n_burnin::Int64 = 500,
    n_iter::Int64 = 1_500,
    verbose::Bool = false,
)::Fit
    # genomes = GBCore.simulategenomes()
    # trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.1 0.01 0.01;], verbose=false);
    # phenomes = extractphenomes(trials); idx_trait = 1;
    # idx_entries = nothing; idx_loci_alleles = nothing
    # sampler = ["NUTS", "HMC", "HMCDA", "MH", "PG"][1]; sampling_method = 1; seed = 123; n_burnin = 500; n_iter = 1_500; verbose = true;
    # Check arguments and extract X, y, and loci-allele names
    X, y, entries, populations, loci_alleles = extractxyetc(
        genomes,
        phenomes,
        idx_entries = idx_entries,
        idx_loci_alleles = idx_loci_alleles,
        idx_trait = idx_trait,
        add_intercept = false,
    )
    # Instantiate output Fit
    fit = Fit(n = size(X, 1), l = size(X, 2) + 1)
    fit.model = replace(string(turing_model), "turing_" => "")
    fit.b_hat_labels = vcat(["intercept"], loci_alleles)
    fit.trait = phenomes.traits[idx_trait]
    fit.entries = entries
    fit.populations = populations
    fit.y_true = y
    # MCMC
    rng::TaskLocalRNG = Random.seed!(seed)
    model = turing_model(X, y)
    if sampler == "NUTS"
        if sampling_method == 1
            sampling_function = NUTS(
                n_burnin,
                0.65,
                max_depth = 5,
                Δ_max = 1000.0,
                init_ϵ = 0.2;
                adtype = AutoReverseDiff(compile = true),
            )
        elseif sampling_method == 2
            sampling_function = NUTS(
                n_burnin,
                0.65,
                max_depth = 5,
                Δ_max = 1000.0,
                init_ϵ = 0.0;
                adtype = AutoReverseDiff(compile = true),
            )
        elseif sampling_method == 3
            sampling_function = NUTS(
                n_burnin,
                0.65,
                max_depth = 5,
                Δ_max = 1000.0,
                init_ϵ = 0.0;
                adtype = AutoReverseDiff(compile = false),
            )
        elseif sampling_method == 4
            sampling_function =
                NUTS(n_burnin, 0.65, max_depth = 5, Δ_max = 1000.0, init_ϵ = 0.0; adtype = AutoForwardDiff())
        elseif sampling_method == 5
            # May fail if the turing model has for-loop/s
            sampling_function = NUTS(n_burnin, 0.65, max_depth = 5, Δ_max = 1000.0, init_ϵ = 0.0; adtype = AutoZygote())
        else
            sampling_function = NUTS()
        end
    elseif sampler == "HMC"
        if sampling_method == 1
            sampling_function = HMC(0.01, 10)
        else
            sampling_function = HMC()
        end
    elseif sampler == "HMCDA"
        if sampling_method == 1
            sampling_function = HMCDA(n_burnin, 0.65, 0.3)
        else
            sampling_function = HMCDA()
        end
    elseif sampler == "MH"
        sampling_function = MH()
    elseif sampler == "PG"
        if sampling_method == 1
            sampling_function = PG(5)
        else
            sampling_function = PG()
        end
    else
        throw(ArgumentError("Unrecognised sampler: `" * sampler * "`. Please choose NUTS, HMC, HMCDA, HM or PG."))
    end
    chain = Turing.sample(rng, model, sampling_function, n_iter - n_burnin, progress = verbose)
    # Use the mean paramter values after 150 burn-in iterations
    params = Turing.get_params(chain[(n_burnin+1):end, :, :])
    b_hat = vcat(mean(params.intercept), mean(stack(params.coefficients, dims = 1)[:, :, 1], dims = 2)[:, 1])
    # Assess prediction accuracy
    y_pred::Vector{Float64} = hcat(ones(size(X, 1)), X) * b_hat
    performance::Dict{String,Float64} = metrics(y, y_pred)
    if verbose
        UnicodePlots.scatterplot(y, y_pred)
        UnicodePlots.histogram(y)
        UnicodePlots.histogram(y_pred)
        println(performance)
    end
    # Output
    fit.b_hat = b_hat
    fit.y_pred = y_pred
    fit.metrics = performance
    if !checkdims(fit)
        throw(ErrorException("Error fitting " * fit.model * "."))
    end
    fit
end
