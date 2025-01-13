"""
    ols(genomes::Genomes, phenomes::Phenomes, trait_idx::Int64=1, verbose::Bool=false)::Fit

Fit an ordinary least squares model

## Examples
```jldoctest; setup = :(using GBCore, GBModels)
julia> genomes = GBCore.simulategenomes(n=10, l=100, verbose=false);

julia> trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=3, f_add_dom_epi=[0.1 0.01 0.01;], verbose=false);

julia> phenomes = Phenomes(n=10, t=1);

julia> phenomes.entries = trials.entries[1:10]; phenomes.populations = trials.populations[1:10]; phenomes.traits = trials.traits; phenomes.phenotypes = trials.phenotypes[1:10, :];

julia> fit = ols(genomes=genomes, phenomes=phenomes);

julia> fit.model == "ols"
true

julia> fit.metrics["cor"] > 0.50
true
```
"""
function ols(; genomes::Genomes, phenomes::Phenomes, trait_idx::Int64 = 1, verbose::Bool = false)::Fit
    # genomes = GBCore.simulategenomes()
    # trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=3, f_add_dom_epi=[0.1 0.01 0.01;])
    # tebv = GBCore.analyse(trials, max_levels = 15, max_time_per_model = 2)
    # phenomes = tebv.phenomes[1]
    # trait_idx = 1; verbose = true
    # Merge genomes and phenomes keeping only common the entries
    if genomes.entries != phenomes.entries
        genomes, phenomes = GBCore.merge(genomes, phenomes, keep_all = false)
    end
    # Omit entries missing phenotype data and apply mask (for cross-validation purposes)
    idx::Vector{Int64} = findall(ismissing.(phenomes.phenotypes[:, trait_idx]) .== false)
    if length(idx) < 2
        throw(
            ArgumentError(
                "There are less than 2 entries with non-missing phenotype data after merging with the genotype data.",
            ),
        )
    end
    X::Matrix{Float64} = hcat(ones(length(idx)), genomes.allele_frequencies[idx, :])
    y::Vector{Float64} = phenomes.phenotypes[idx, trait_idx]
    # Instantiate output Fit
    fit::Fit = Fit(l = size(X, 2))
    fit.model = "ols"
    fit.b_hat_labels = vcat(["intercept"], genomes.loci_alleles)
    # Ordinary least squares regression
    b_hat::Vector{Float64} = X \ y
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
julia> genomes = GBCore.simulategenomes(n=10, l=100, verbose=false);

julia> trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=3, f_add_dom_epi=[0.1 0.01 0.01;], verbose=false);

julia> phenomes = Phenomes(n=10, t=1);

julia> phenomes.entries = trials.entries[1:10]; phenomes.populations = trials.populations[1:10]; phenomes.traits = trials.traits; phenomes.phenotypes = trials.phenotypes[1:10, :];

julia> fit = ridge(genomes=genomes, phenomes=phenomes);

julia> fit.metrics["cor"] > 0.50
true
```
"""
function ridge(; genomes::Genomes, phenomes::Phenomes, trait_idx::Int64 = 1, verbose::Bool = false)::Fit
    # genomes = GBCore.simulategenomes()
    # trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=3, f_add_dom_epi=[0.1 0.01 0.01;])
    # tebv = GBCore.analyse(trials, max_levels = 15, max_time_per_model = 2)
    # phenomes = tebv.phenomes[1]
    # trait_idx = 1; verbose = true
    # Merge genomes and phenomes keeping only common the entries
    if genomes.entries != phenomes.entries
        genomes, phenomes = GBCore.merge(genomes, phenomes, keep_all = false)
    end
    # Omit entries missing phenotype data and apply mask (for cross-validation purposes)
    idx::Vector{Int64} = findall(ismissing.(phenomes.phenotypes[:, trait_idx]) .== false)
    if length(idx) < 2
        throw(
            ArgumentError(
                "There are less than 2 entries with non-missing phenotype data after merging with the genotype data.",
            ),
        )
    end
    X::Matrix{Float64} = hcat(ones(length(idx)), genomes.allele_frequencies[idx, :])
    y::Vector{Float64} = phenomes.phenotypes[idx, trait_idx]
    # Instantiate output Fit
    fit::Fit = Fit(l = size(X, 2))
    fit.model = "ridge"
    fit.b_hat_labels = vcat(["intercept"], genomes.loci_alleles)
    # Ridge regression using the glmnet package
    glmnet_fit = GLMNet.glmnetcv(
        X,
        y,
        alpha = 0.0,
        standardize = false,
        nlambda = 100,
        lambda_min_ratio = 0.01,
        tol = 1e-7,
        intercept = true,
        maxit = 1_000_000,
    )
    if verbose
        UnicodePlots.histogram(glmnet_fit.meanloss)
        UnicodePlots.scatterplot(glmnet_fit.meanloss)
        UnicodePlots.scatterplot(glmnet_fit.lambda, glmnet_fit.meanloss)
        UnicodePlots.scatterplot(log10.(glmnet_fit.lambda), glmnet_fit.meanloss)
        println(string("argmin = ", argmin(glmnet_fit.meanloss)))
    end
    b_hat::Vector{Float64} = GLMNet.coef(glmnet_fit)
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
julia> genomes = GBCore.simulategenomes(n=10, l=100, verbose=false);

julia> trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=3, f_add_dom_epi=[0.1 0.01 0.01;], verbose=false);

julia> phenomes = Phenomes(n=10, t=1);

julia> phenomes.entries = trials.entries[1:10]; phenomes.populations = trials.populations[1:10]; phenomes.traits = trials.traits; phenomes.phenotypes = trials.phenotypes[1:10, :];

julia> fit = lasso(genomes=genomes, phenomes=phenomes);

julia> fit.metrics["cor"] > 0.0
true
```
"""
function lasso(; genomes::Genomes, phenomes::Phenomes, trait_idx::Int64 = 1, verbose::Bool = false)::Fit
    # genomes = GBCore.simulategenomes()
    # trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=3, f_add_dom_epi=[0.1 0.01 0.01;])
    # tebv = GBCore.analyse(trials, max_levels = 15, max_time_per_model = 2)
    # phenomes = tebv.phenomes[1]
    # trait_idx = 1; verbose = true
    # Merge genomes and phenomes keeping only common the entries
    if genomes.entries != phenomes.entries
        genomes, phenomes = GBCore.merge(genomes, phenomes, keep_all = false)
    end
    # Omit entries missing phenotype data and apply mask (for cross-validation purposes)
    idx::Vector{Int64} = findall(ismissing.(phenomes.phenotypes[:, trait_idx]) .== false)
    if length(idx) < 2
        throw(
            ArgumentError(
                "There are less than 2 entries with non-missing phenotype data after merging with the genotype data.",
            ),
        )
    end
    X::Matrix{Float64} = hcat(ones(length(idx)), genomes.allele_frequencies[idx, :])
    y::Vector{Float64} = phenomes.phenotypes[idx, trait_idx]
    # Instantiate output Fit
    fit::Fit = Fit(l = size(X, 2))
    fit.model = "lasso"
    fit.b_hat_labels = vcat(["intercept"], genomes.loci_alleles)
    # Lasso regression using the glmnet package
    glmnet_fit = GLMNet.glmnetcv(
        X,
        y,
        alpha = 1.0,
        standardize = false,
        nlambda = 100,
        lambda_min_ratio = 0.01,
        tol = 1e-7,
        intercept = true,
        maxit = 1_000_000,
    )
    if verbose
        UnicodePlots.histogram(glmnet_fit.meanloss)
        UnicodePlots.scatterplot(glmnet_fit.meanloss)
        UnicodePlots.scatterplot(glmnet_fit.lambda, glmnet_fit.meanloss)
        UnicodePlots.scatterplot(log10.(glmnet_fit.lambda), glmnet_fit.meanloss)
        println(string("argmin = ", argmin(glmnet_fit.meanloss)))
    end
    b_hat::Vector{Float64} = GLMNet.coef(glmnet_fit)
    i = 2
    idx_sort = sortperm(glmnet_fit.meanloss)
    BETAs = glmnet_fit.path.betas[:, idx_sort]
    while var(b_hat) < 1e-10
        b_hat = BETAs[:, i]
        i += 1
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
    fit.metrics = performance
    if !checkdims(fit)
        throw(ErrorException("Error fitting lasso."))
    end
    fit
end

"""
    bayesian(;
        genomes::Genomes,
        phenomes::Phenomes,
        trait_idx::Int64 = 1,
        turing_model::Function = turing_bayesG,
        sampling_method::Int64 = 1,
        seed::Int64 = 123,
        nburnin::Int64 = 500,
        niter::Int64 = 1_500,
        verbose::Bool = false,
    )::Fit

Fit a Bayesian ridge regression model

## Examples
```jldoctest; setup = :(using GBCore, GBModels, Suppressor)
julia> genomes = GBCore.simulategenomes(n=10, l=100, verbose=false);

julia> trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=3, f_add_dom_epi=[0.1 0.01 0.01;], verbose=false);

julia> tebv = GBCore.analyse(trials, max_levels=20, max_time_per_model=2, verbose=false);

julia> phenomes = tebv.phenomes[1];

julia> sol = Suppressor.@suppress bayesian(genomes=genomes, phenomes=phenomes);

julia> sol.metrics["cor"] > 0.5
true
```
"""
function bayesian(;
    genomes::Genomes,
    phenomes::Phenomes,
    trait_idx::Int64 = 1,
    turing_model::Function = turing_bayesG,
    sampling_method::Int64 = 1,
    seed::Int64 = 123,
    nburnin::Int64 = 500,
    niter::Int64 = 1_500,
    verbose::Bool = false,
)::Fit
    # genomes = GBCore.simulategenomes(n=10, l=100)
    # trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=3, f_add_dom_epi=[0.9 0.01 0.00;])
    # tebv = GBCore.analyse(trials, max_levels = 15, max_time_per_model = 2)
    # phenomes = tebv.phenomes[1]
    # trait_idx=1; turing_model::Function=turing_bayesG; sampling_method = 1
    # seed = 123; nburnin = 500; niter = 1_500; verbose=true
    # Merge genomes and phenomes keeping only common the entries
    if genomes.entries != phenomes.entries
        genomes, phenomes = GBCore.merge(genomes, phenomes, keep_all = false)
    end
    # Omit entries missing phenotype data and apply mask (for cross-validation purposes)
    idx::Vector{Int64} = findall(ismissing.(phenomes.phenotypes[:, trait_idx]) .== false)
    if length(idx) < 2
        throw(
            ArgumentError(
                "There are less than 2 entries with non-missing phenotype data after merging with the genotype data.",
            ),
        )
    end
    # Normalise explanatory and response variables
    G::Matrix{Float64} = genomes.allele_frequencies[idx, :]
    y::Vector{Float64} = phenomes.phenotypes[idx, trait_idx]
    G = (G .- mean(G, dims = 2)) ./ std(G, dims = 2)
    y = (y .- mean(y)) ./ std(y)
    # Instantiate output Fit
    fit = Fit(l = size(G, 2) + 1)
    fit.model = replace(string(turing_model), "turing_" => "")
    fit.b_hat_labels = vcat(["intercept"], genomes.loci_alleles)
    # MCMC
    rng::TaskLocalRNG = Random.seed!(seed)
    model = turing_model(G, y)
    if sampling_method == 1
        sampling_function =
            NUTS(nburnin, 0.65, max_depth = 5, Δ_max = 1000.0, init_ϵ = 0.2; adtype = AutoReverseDiff(compile = true))
    elseif sampling_method == 2
        sampling_function =
            NUTS(nburnin, 0.65, max_depth = 5, Δ_max = 1000.0, init_ϵ = 0.0; adtype = AutoReverseDiff(compile = true))
    elseif sampling_method == 3
        sampling_function =
            NUTS(nburnin, 0.65, max_depth = 5, Δ_max = 1000.0, init_ϵ = 0.0; adtype = AutoReverseDiff(compile = false))
    elseif sampling_method == 4
        sampling_function = NUTS(nburnin, 0.65, max_depth = 5, Δ_max = 1000.0, init_ϵ = 0.0; adtype = AutoForwardDiff())
    elseif sampling_method == 5
        # May fail
        sampling_function = NUTS(nburnin, 0.65, max_depth = 5, Δ_max = 1000.0, init_ϵ = 0.0; adtype = AutoZygote())
    else
        sampling_function = NUTS()
    end
    chain = Turing.sample(rng, model, sampling_function, niter - nburnin, progress = verbose)
    # Use the mean paramter values after 150 burn-in iterations
    params = Turing.get_params(chain[(nburnin+1):end, :, :])
    b_hat = vcat(mean(params.intercept), mean(stack(params.coefficients, dims = 1)[:, :, 1], dims = 2)[:, 1])
    # Assess prediction accuracy
    y_pred::Vector{Float64} = hcat(ones(size(G, 1)), G) * b_hat
    performance::Dict{String,Float64} = metrics(y, y_pred)
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

# using RCall, StatsBase, UnicodePlots
# R"library(BGLR)"
# X = genomes.allele_frequencies
# y = phenomes.phenotypes[:,1]
# model = "BayesA"
# @rput(X)
# @rput(y)
# @rput(model)
# @time R"sol = BGLR::BGLR(y=y, ETA=list(MRK=list(X=X, model=model, saveEffects=FALSE)), verbose=TRUE)"
# R"bhat = sol$ETA$MRK$b"

# @time sol = bayesian(genomes=genomes, phenomes=phenomes, turing_model=turing_bayesT, verbose=true)

# @rget bhat

# cor(X * bhat, y)
# cor(X * sol.b_hat[2:end], y)

# UnicodePlots.histogram(bhat)
# UnicodePlots.histogram(sol.b_hat)
