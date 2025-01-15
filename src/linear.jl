"""
    extractxy(;
        genomes::Genomes,
        phenomes::Phenomes,
        idx_entries::Vector{Int64},
        idx_loci_alleles::Vector{Int64},
        idx_trait::Int64 = 1,
        add_intercept::Bool = true,
    )::Tuple{Matrix{Float64},Vector{Float64},Vector{String}}

Extract explanatory `X` matrix, and response `y` vector from genomes and phenomes.

# Examples
```jldoctest; setup = :(using GBCore, GBModels)
julia> genomes = GBCore.simulategenomes(verbose=false);

julia> trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.1 0.01 0.01;], verbose=false);;

julia> phenomes = extractphenomes(trials);

julia> X, y, loci_alleles = extractxy(genomes=genomes, phenomes=phenomes);

julia> X == hcat(ones(length(phenomes.entries)), genomes.allele_frequencies)
true

julia> y == phenomes.phenotypes[:, 1]
true
```
"""
function extractxy(;
    genomes::Genomes,
    phenomes::Phenomes,
    idx_trait::Int64 = 1,
    add_intercept::Bool = true,
)::Tuple{Matrix{Float64},Vector{Float64},Vector{String}}
    # genomes = GBCore.simulategenomes()
    # trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.1 0.01 0.01;], verbose=false);
    # phenomes = extractphenomes(trials)
    # idx_trait = 1; add_intercept = true
    # Check arguments
    if genomes.entries != phenomes.entries
        throw(ArgumentError("The genomes and phenomes input need to have been merged to have consitent entries."))
    end
    if findall(mean(genomes.mask, dims = 2)[:, 1] .== 1.0) != findall(mean(phenomes.mask, dims = 2)[:, 1] .== 1.0)
        throw(ArgumentError("The masks in genomes and phenomes do not match."))
    end
    # Apply mask
    genomes = filter(genomes)
    phenomes = filter(phenomes)
    # Extract the response variable
    y::Vector{Float64} = phenomes.phenotypes[:, idx_trait]
    # Omit entries missing phenotype data
    idx::Vector{Int64} = findall(ismissing.(y) .== false)
    if length(idx) < 2
        throw(
            ArgumentError(
                "There are less than 2 entries with non-missing phenotype data after merging with the genotype data.",
            ),
        )
    end
    y = y[idx]
    if var(y) < 1e-20
        throw(ErrorException("Very low or zero variance in trait: `" * phenomes.traits[idx_trait] * "`."))
    end
    # Extract the explanatory matrix
    G::Matrix{Float64} = genomes.allele_frequencies[idx, :]
    # Output X with/without intercept, and y
    if add_intercept
        return (hcat(ones(length(idx)), G), y, genomes.loci_alleles)
    else
        return (G, y, genomes.loci_alleles)
    end
end

"""
    ols(; genomes::Genomes, phenomes::Phenomes, idx_trait::Int64 = 1, verbose::Bool = false)::Fit

Fit an ordinary least squares model

## Examples
```jldoctest; setup = :(using GBCore, GBModels)
julia> genomes = GBCore.simulategenomes(verbose=false);

julia> trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.1 0.01 0.01;], verbose=false);;

julia> phenomes = extractphenomes(trials);

julia> fit = ols(genomes=genomes, phenomes=phenomes);

julia> fit.model == "ols"
true

julia> fit.metrics["cor"] > 0.50
true
```
"""
function ols(; genomes::Genomes, phenomes::Phenomes, idx_trait::Int64 = 1, verbose::Bool = false)::Fit
    # genomes = GBCore.simulategenomes()
    # trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.1 0.01 0.01;], verbose=false);
    # phenomes = extractphenomes(trials)
    # idx_trait = 1; verbose = true
    # Check arguments and extract X, y, and loci-allele names
    X, y, loci_alleles = extractxy(genomes = genomes, phenomes = phenomes, idx_trait = idx_trait, add_intercept = true)
    # Instantiate output Fit
    fit::Fit = Fit(l = size(X, 2))
    fit.model = "ols"
    fit.b_hat_labels = vcat(["intercept"], loci_alleles)
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
        throw(ErrorException("Error fitting " * fit.model * "."))
    end
    fit
end


"""
    ridge(; genomes::Genomes, phenomes::Phenomes, idx_trait::Int64 = 1, verbose::Bool = false)::Fit

Fit a ridge (L2) regression model

## Examples
```jldoctest; setup = :(using GBCore, GBModels)
julia> genomes = GBCore.simulategenomes(verbose=false);

julia> trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.1 0.01 0.01;], verbose=false);;

julia> phenomes = extractphenomes(trials);

julia> fit = ridge(genomes=genomes, phenomes=phenomes);

julia> fit.model == "ridge"
true

julia> fit.metrics["cor"] > 0.50
true
```
"""
function ridge(; genomes::Genomes, phenomes::Phenomes, idx_trait::Int64 = 1, verbose::Bool = false)::Fit
    # genomes = GBCore.simulategenomes()
    # trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.1 0.01 0.01;], verbose=false);
    # phenomes = extractphenomes(trials)
    # idx_trait = 1; verbose = true
    # Check arguments and extract X, y, and loci-allele names
    X, y, loci_alleles = extractxy(genomes = genomes, phenomes = phenomes, idx_trait = idx_trait, add_intercept = true)
    # Instantiate output Fit
    fit::Fit = Fit(l = size(X, 2))
    fit.model = "ridge"
    fit.b_hat_labels = vcat(["intercept"], loci_alleles)
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
    # Use the coefficients with variance
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
        throw(ErrorException("Error fitting " * fit.model * "."))
    end
    fit
end


"""
    lasso(; genomes::Genomes, phenomes::Phenomes, idx_trait::Int64 = 1, verbose::Bool = false)::Fit

Fit a LASSO (least absolute shrinkage and selection operator; L1) regression model

## Examples
```jldoctest; setup = :(using GBCore, GBModels)
julia> genomes = GBCore.simulategenomes(verbose=false);

julia> trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.1 0.01 0.01;], verbose=false);;

julia> phenomes = extractphenomes(trials);

julia> fit = lasso(genomes=genomes, phenomes=phenomes);

julia> fit.model == "lasso"
true

julia> fit.metrics["cor"] > 0.50
true
```
"""
function lasso(; genomes::Genomes, phenomes::Phenomes, idx_trait::Int64 = 1, verbose::Bool = false)::Fit
    # genomes = GBCore.simulategenomes()
    # trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.1 0.01 0.01;], verbose=false);
    # phenomes = extractphenomes(trials)
    # idx_trait = 1; verbose = true
    # Check arguments and extract X, y, and loci-allele names
    X, y, loci_alleles = extractxy(genomes = genomes, phenomes = phenomes, idx_trait = idx_trait, add_intercept = true)
    # Instantiate output Fit
    fit::Fit = Fit(l = size(X, 2))
    fit.model = "lasso"
    fit.b_hat_labels = vcat(["intercept"], loci_alleles)
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
    # Use the coefficients with variance
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
        throw(ErrorException("Error fitting " * fit.model * "."))
    end
    fit
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
    # sampler = ["NUTS", "HMC", "HMCDA", "MH", "PG"][1]; sampling_method = 1; seed = 123; n_burnin = 500; n_iter = 1_500; verbose = true;
    # Check arguments and extract X, y, and loci-allele names
    X, y, loci_alleles = extractxy(genomes = genomes, phenomes = phenomes, idx_trait = idx_trait, add_intercept = false)
    # Instantiate output Fit
    fit = Fit(l = size(X, 2) + 1)
    fit.model = replace(string(turing_model), "turing_" => "")
    fit.b_hat_labels = vcat(["intercept"], loci_alleles)
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
    fit.metrics = performance
    if !checkdims(fit)
        throw(ErrorException("Error fitting " * fit.model * "."))
    end
    fit
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
    idx_trait::Int64 = 1,
    n_burnin::Int64 = 500,
    n_iter::Int64 = 1_500,
    verbose::Bool = false,
)::Fit
    # bglr_model = "BayesA"
    # genomes = GBCore.simulategenomes()
    # trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.1 0.01 0.01;], verbose=false);
    # phenomes = extractphenomes(trials); idx_trait = 1
    # n_burnin = 500; n_iter = 1_500; verbose = true;
    # Check arguments and extract X, y, and loci-allele names
    X, y, loci_alleles = extractxy(genomes = genomes, phenomes = phenomes, idx_trait = idx_trait, add_intercept = true)
    # Instantiate output Fit
    fit = Fit(l = size(X, 2))
    fit.model = bglr_model
    fit.b_hat_labels = vcat(["intercept"], loci_alleles)
    # R-BGLR
    b_hat = bglr(G = X[:, 2:end], y = y, model = bglr_model, n_iter = n_iter, n_burnin = n_burnin, verbose = verbose)
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
    fit.metrics = performance
    if !checkdims(fit)
        throw(ErrorException("Error fitting " * fit.model * "."))
    end
    fit
end

"""
    logistic(
        model::Function;
        X::Matrix{Float64},
        y::Vector{Float64},
        verbose::Bool = false,
    )::Fit

Fit a linear model with binary response variable, using ols, ridge or lasso functions.

## Examples
```jldoctest; setup = :(using GBCore, GBModels, StatsBase)
julia> genomes = GBCore.simulategenomes(verbose=false);

julia> trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.1 0.01 0.01;], verbose=false);;

julia> phenomes = extractphenomes(trials); y = phenomes.phenotypes[:, 1];

julia> phenomes.phenotypes[:, 1] = (y .- minimum(y)) ./ (maximum(y) - minimum(y));

julia> fit_ols = logistic(ols, genomes=genomes, phenomes=phenomes);

julia> fit_ols.model == "logistic-ols"
true

julia> fit_ols.metrics["cor"] > 0.50
true

julia> fit_ridge = logistic(ridge, genomes=genomes, phenomes=phenomes);

julia> fit_ridge.model == "logistic-ridge"
true

julia> fit_ridge.metrics["cor"] > 0.00
true

julia> fit_lasso = logistic(lasso, genomes=genomes, phenomes=phenomes);

julia> fit_lasso.model == "logistic-lasso"
true

julia> fit_lasso.metrics["cor"] > 0.00
true
```
"""
function logistic(
    model::Function;
    genomes::Genomes,
    phenomes::Phenomes,
    idx_trait::Int64 = 1,
    verbose::Bool = false,
)::Fit
    # model = ridge
    # genomes = GBCore.simulategenomes()
    # trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.1 0.01 0.01;], verbose=false);
    # phenomes = extractphenomes(trials)
    # idx_trait = 1; verbose = true
    # Check arguments and extract X, y, and loci-allele names
    X, y, _loci_alleles = extractxy(genomes = genomes, phenomes = phenomes, idx_trait = idx_trait, add_intercept = true)
    # Log the binary response variable
    ϵ = 1e-20
    ϕ = clone(phenomes)
    if sum((ϕ.phenotypes[:, idx_trait] .< 0.0) .&& (ϕ.phenotypes[:, idx_trait] .> 1.0)) > 0
        throw(
            ArgumentError(
                "The trait: `" * ϕ.traits[idx_trait] * "` is not binary or at least does not range between 0 and 1.",
            ),
        )
    end
    ϕ.phenotypes[:, idx_trait] = log.(ϵ .+ ϕ.phenotypes[:, idx_trait])
    # Fit
    fit = model(genomes = genomes, phenomes = ϕ, idx_trait = idx_trait, verbose = verbose)
    # Extract the logisitc predictions and update Fit struct
    y_hat::Vector{Float64} = exp.(X * fit.b_hat)
    fit.model = string("logistic-", fit.model)
    fit.metrics = metrics(y, y_hat)
    # Output
    if !checkdims(fit)
        throw(ErrorException("Error fitting " * fit.model * "."))
    end
    fit
end
