"""
    ols(genomes::Genomes, phenomes::Phenomes, trait_idx::Int64=1, verbose::Bool=false)::Fit

Fit an ordinary least squares model

## Examples
```jldoctest; setup = :(using GBCore, GBModels)
julia> genomes = GBCore.simulategenomes();

julia> trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=3, f_add_dom_epi=[0.1 0.01 0.01;]);

julia> tebv = GBCore.analyse(trials, max_levels = 15, max_time_per_model = 2);

julia> phenomes = tebv.phenomes[1];

julia> fit = ols(genomes=genomes, phenomes=phenomes);

julia> fit.model == "ols"
true

julia> fit.metrics["cor"] > 0.50
true
```
"""
function ols(;genomes::Genomes, phenomes::Phenomes, trait_idx::Int64=1, verbose::Bool=false)::Fit
    # genomes = GBCore.simulategenomes()
    # trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=3, f_add_dom_epi=[0.1 0.01 0.01;])
    # tebv = GBCore.analyse(trials, max_levels = 15, max_time_per_model = 2)
    # phenomes = tebv.phenomes[1]
    # trait_idx = 1; verbose = true
    # Merge genomes and phenomes keeping only common the entries
    g, p = GBCore.merge(genomes, phenomes, keep_all=false)
    # Omit entries missing phenotype data
    idx::Vector{Int64} = findall(ismissing.(p.phenotypes[:, trait_idx]) .== false)
    if length(idx) < 2
        throw(ArgumentError("There are less than 2 entries with non-missing phenotype data after merging with the genotype data."))
    end
    X::Matrix{Float64} = hcat(ones(length(idx)), g.allele_frequencies[idx, :])
    y::Vector{Float64} = p.phenotypes[idx, trait_idx]
    # Instantiate output Fit
    fit::Fit = Fit(l=size(X,2))
    fit.model = "ols"
    fit.b_hat_labels = vcat(["intercept"], genomes.loci_alleles)
    # Ordinary least squares regression
    b_hat::Vector{Float64} = X \ y
    # Assess prediction accuracy
    y_pred::Vector{Float64} = X*b_hat
    performance::Dict{String, Float64} = metrics(y, y_pred)
    if verbose
        UnicodePlots.scatterplot(y, y_pred)
        UnicodePlots.histogram(y)
        UnicodePlots.histogram(y_pred)
        println(performance)
    end
    # Output
    fit.b_hat = b_hat
    fit.metrics = performance
    if !checkdims(fit)
        throw(ErrorException("Error fitting ols."))
    end
    fit
end


"""
    ridge(genomes::Genomes, phenomes::Phenomes, trait_idx::Int64=1, verbose::Bool=false)::Fit

Fit a ridge (L2) regression model

## Examples
```jldoctest; setup = :(using GBCore, GBModels)
julia> genomes = GBCore.simulategenomes();

julia> trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=3, f_add_dom_epi=[0.1 0.01 0.01;]);

julia> tebv = GBCore.analyse(trials, max_levels = 15, max_time_per_model = 2);

julia> phenomes = tebv.phenomes[1];

julia> fit = ridge(genomes=genomes, phenomes=phenomes);

julia> fit.model == "ridge"
true

julia> fit.metrics["cor"] > 0.50
true
```
"""
function ridge(;genomes::Genomes, phenomes::Phenomes, trait_idx::Int64=1, verbose::Bool=false)::Fit
    # genomes = GBCore.simulategenomes()
    # trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=3, f_add_dom_epi=[0.1 0.01 0.01;])
    # tebv = GBCore.analyse(trials, max_levels = 15, max_time_per_model = 2)
    # phenomes = tebv.phenomes[1]
    # trait_idx = 1; verbose = true
    # Merge genomes and phenomes keeping only common the entries
    g, p = GBCore.merge(genomes, phenomes, keep_all=false)
    # Omit entries missing phenotype data
    idx::Vector{Int64} = findall(ismissing.(p.phenotypes[:, trait_idx]) .== false)
    if length(idx) < 2
        throw(ArgumentError("There are less than 2 entries with non-missing phenotype data after merging with the genotype data."))
    end
    X::Matrix{Float64} = hcat(ones(length(idx)), g.allele_frequencies[idx, :])
    y::Vector{Float64} = p.phenotypes[idx, trait_idx]
    # Instantiate output Fit
    fit::Fit = Fit(l=size(X,2))
    fit.model = "ridge"
    fit.b_hat_labels = vcat(["intercept"], genomes.loci_alleles)
    # Ridge regression using the glmnet package
    glmnet_fit = GLMNet.glmnetcv(X, y, alpha=0.0, standardize=false, nlambda=100, lambda_min_ratio=0.01, tol=1e-7, intercept=true, maxit=1_000_000)
    if verbose
        UnicodePlots.histogram(glmnet_fit.meanloss)
        UnicodePlots.scatterplot(glmnet_fit.meanloss)
        UnicodePlots.scatterplot(glmnet_fit.lambda, glmnet_fit.meanloss)
        UnicodePlots.scatterplot(log10.(glmnet_fit.lambda), glmnet_fit.meanloss)
        println(string("agrmin = ", argmin(glmnet_fit.meanloss)))
    end
    b_hat::Vector{Float64} = GLMNet.coef(glmnet_fit)
    # Assess prediction accuracy
    y_pred::Vector{Float64} = X*b_hat
    performance::Dict{String, Float64} = metrics(y, y_pred)
    if verbose
        UnicodePlots.scatterplot(y, y_pred)
        UnicodePlots.histogram(y)
        UnicodePlots.histogram(y_pred)
        println(performance)
    end
    # Output
    fit.b_hat = b_hat
    fit.metrics = performance
    if !checkdims(fit)
        throw(ErrorException("Error fitting ridge."))
    end
    fit
end


"""
    lasso(genomes::Genomes, phenomes::Phenomes, trait_idx::Int64=1, verbose::Bool=false)::Fit

Fit a LASSO (least absolute shrinkage and selection operator; L1) regression model

## Examples
```jldoctest; setup = :(using GBCore, GBModels)
julia> genomes = GBCore.simulategenomes();

julia> trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=3, f_add_dom_epi=[0.1 0.01 0.01;]);

julia> tebv = GBCore.analyse(trials, max_levels = 15, max_time_per_model = 2);

julia> phenomes = tebv.phenomes[1];

julia> fit = lasso(genomes=genomes, phenomes=phenomes);

julia> fit.model == "lasso"
true

julia> fit.metrics["cor"] > 0.50
true
```
"""
function lasso(;genomes::Genomes, phenomes::Phenomes, trait_idx::Int64=1, verbose::Bool=false)::Fit
    # genomes = GBCore.simulategenomes()
    # trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=3, f_add_dom_epi=[0.1 0.01 0.01;])
    # tebv = GBCore.analyse(trials, max_levels = 15, max_time_per_model = 2)
    # phenomes = tebv.phenomes[1]
    # trait_idx = 1; verbose = true
    # Merge genomes and phenomes keeping only common the entries
    g, p = GBCore.merge(genomes, phenomes, keep_all=false)
    # Omit entries missing phenotype data
    idx::Vector{Int64} = findall(ismissing.(p.phenotypes[:, trait_idx]) .== false)
    if length(idx) < 2
        throw(ArgumentError("There are less than 2 entries with non-missing phenotype data after merging with the genotype data."))
    end
    X::Matrix{Float64} = hcat(ones(length(idx)), g.allele_frequencies[idx, :])
    y::Vector{Float64} = p.phenotypes[idx, trait_idx]
    # Instantiate output Fit
    fit::Fit = Fit(l=size(X,2))
    fit.model = "lasso"
    fit.b_hat_labels = vcat(["intercept"], genomes.loci_alleles)
    # Lasso regression using the glmnet package
    glmnet_fit = GLMNet.glmnetcv(X, y, alpha=1.0, standardize=false, nlambda=100, lambda_min_ratio=0.01, tol=1e-7, intercept=true, maxit=1_000_000)
    if verbose
        UnicodePlots.histogram(glmnet_fit.meanloss)
        UnicodePlots.scatterplot(glmnet_fit.meanloss)
        UnicodePlots.scatterplot(glmnet_fit.lambda, glmnet_fit.meanloss)
        UnicodePlots.scatterplot(log10.(glmnet_fit.lambda), glmnet_fit.meanloss)
        println(string("agrmin = ", argmin(glmnet_fit.meanloss)))
    end
    b_hat::Vector{Float64} = GLMNet.coef(glmnet_fit)
    # Assess prediction accuracy
    y_pred::Vector{Float64} = X*b_hat
    performance::Dict{String, Float64} = metrics(y, y_pred)
    if verbose
        UnicodePlots.scatterplot(y, y_pred)
        UnicodePlots.histogram(y)
        UnicodePlots.histogram(y_pred)
        println(performance)
    end
    # Output
    fit.b_hat = b_hat
    fit.metrics = performance
    if !checkdims(fit)
        throw(ErrorException("Error fitting lasso."))
    end
    fit
end


"""
    bayesA(genomes::Genomes, phenomes::Phenomes, trait_idx::Int64=1, verbose::Bool=false)::Fit

Fit a Bayes A regression model

## Examples
```jldoctest; setup = :(using GBCore, GBModels)
julia> genomes = GBCore.simulategenomes();

julia> trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=3, f_add_dom_epi=[0.1 0.01 0.01;]);

julia> tebv = GBCore.analyse(trials, max_levels = 15, max_time_per_model = 2);

julia> phenomes = tebv.phenomes[1];

julia> fit = bayesA(genomes=genomes, phenomes=phenomes);

julia> fit.model == "bayesA"
true
```
"""
function bayesA(;genomes::Genomes, phenomes::Phenomes, trait_idx::Int64=1, verbose::Bool=false, seed::Int64 = 123, nburnin::Int64 = 1_000, niter::Int64 = 5_000, nthreads::Int64 = 1)
    # genomes = GBCore.simulategenomes(n=10, l=100)
    # trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=3, f_add_dom_epi=[0.9 0.01 0.00;])
    # tebv = GBCore.analyse(trials, max_levels = 15, max_time_per_model = 2)
    # phenomes = tebv.phenomes[1]
    # trait_idx::Int64=1; verbose::Bool=false
    # seed::Int64 = 123; nburnin::Int64 = 1_000; niter::Int64 = 5_000; nthreads::Int64 = 5
    # Merge genomes and phenomes keeping only common the entries
    g, p = GBCore.merge(genomes, phenomes, keep_all=false)
    # Omit entries missing phenotype data
    idx::Vector{Int64} = findall(ismissing.(p.phenotypes[:, trait_idx]) .== false)
    if length(idx) < 2
        throw(ArgumentError("There are less than 2 entries with non-missing phenotype data after merging with the genotype data."))
    end
    G::Matrix{Float64} = g.allele_frequencies[idx, :]
    y::Vector{Float64} = p.phenotypes[idx, trait_idx]
    n, p = size(G)
    # Instantiate output Fit
    fit::Fit = Fit(l=size(G,2)+1)
    fit.model = "bayesA"
    fit.b_hat_labels = vcat(["intercept"], genomes.loci_alleles)
    # Number of samples per thread or per chain
    nsamples_per_thread::Int64 = Int64(ceil(niter / nthreads))
    # MCMC
    rng::TaskLocalRNG = Random.seed!(seed)
    model = turing_bayesA(G, y)
    # @time chain = Turing.sample(rng, model, NUTS(), niter, progress=true)
    @time chain = Turing.sample(rng, model, NUTS(), MCMCThreads(), nsamples_per_thread, nthreads; verbose=true, progress=true)
    # Use the mean paramter values after burn-in
    b_hat::Vector{Float64} = zeros(p+1)
    weight::Float64 = 1.00 / nthreads
    for i in 1:nthreads
        params = Turing.get_params(chain[nburnin:end, :, i])
        b_hat[1] += weight * params.intercept[1]
        b_hat[2:end] .+= weight .* reduce(hcat, params.coefficients)[1, :]
        y_pred += mean(reduce(hcat, params.intercept) .+ G * reduce(hcat, params.coefficients)', dims=2)[:,1] .* (1.00/nthreads)
    end
    # Assess prediction accuracy
    y_pred::Vector{Float64} = hcat(ones(n), G) * b_hat
    performance::Dict{String, Float64} = metrics(y, y_pred)
    if verbose
        UnicodePlots.scatterplot(y, y_pred)
        UnicodePlots.histogram(y)
        UnicodePlots.histogram(y_pred)
        println(performance)
    end
    # Output
    fit.b_hat = b_hat
    fit.metrics = performance
    if !checkdims(fit)
        throw(ErrorException("Error fitting ridge."))
    end
    fit
end