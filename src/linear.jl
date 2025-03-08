"""
    ols(;
        genomes::Genomes,
        phenomes::Phenomes,
        idx_entries::Union{Nothing,Vector{Int64}} = nothing,
        idx_loci_alleles::Union{Nothing,Vector{Int64}} = nothing,
        idx_trait::Int64 = 1,
        verbose::Bool = false
    )::Fit

Fits an ordinary least squares (OLS) regression model to genomic and phenotypic data.

# Arguments
- `genomes::Genomes`: Genomic data containing genetic markers
- `phenomes::Phenomes`: Phenotypic data containing trait measurements
- `idx_entries::Union{Nothing,Vector{Int64}}`: Optional indices to select specific entries (default: all entries)
- `idx_loci_alleles::Union{Nothing,Vector{Int64}}`: Optional indices to select specific loci-alleles (default: all loci-alleles)
- `idx_trait::Int64`: Index of the trait to analyze (default: 1)
- `verbose::Bool`: If true, displays diagnostic plots and performance metrics (default: false)

# Returns
- `Fit`: A fitted model object containing:
  - `model`: Model identifier ("ols")
  - `b_hat`: Estimated regression coefficients
  - `b_hat_labels`: Labels for the coefficients
  - `y_true`: Observed phenotypic values
  - `y_pred`: Predicted phenotypic values
  - `metrics`: Dictionary of performance metrics
  - `trait`: Name of the analyzed trait
  - `entries`: Entry identifiers
  - `populations`: Population identifiers

# Description
Performs ordinary least squares regression on genomic data to predict phenotypic values.
The model includes an intercept term and estimates effects for each locus-allele combination.

# Examples
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
function ols(;
    genomes::Genomes,
    phenomes::Phenomes,
    idx_entries::Union{Nothing,Vector{Int64}} = nothing,
    idx_loci_alleles::Union{Nothing,Vector{Int64}} = nothing,
    idx_trait::Int64 = 1,
    verbose::Bool = false,
)::Fit
    # genomes = GBCore.simulategenomes()
    # trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.1 0.01 0.01;], verbose=false);
    # phenomes = extractphenomes(trials)
    # idx_entries = nothing; idx_loci_alleles = nothing
    # idx_trait = 1; verbose = true
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
    fit::Fit = Fit(n = size(X, 1), l = size(X, 2))
    fit.model = "ols"
    fit.b_hat_labels = vcat(["intercept"], loci_alleles)
    fit.trait = phenomes.traits[idx_trait]
    fit.entries = entries
    fit.populations = populations
    fit.y_true = y
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
    fit.y_pred = y_pred
    if !checkdims(fit)
        throw(ErrorException("Error fitting " * fit.model * "."))
    end
    fit
end


"""
    ridge(;
        genomes::Genomes,
        phenomes::Phenomes,
        idx_entries::Union{Nothing,Vector{Int64}} = nothing,
        idx_loci_alleles::Union{Nothing,Vector{Int64}} = nothing,
        idx_trait::Int64 = 1,
        verbose::Bool = false
    )::Fit

Fit a ridge (L2) regression model to genomic data. Ridge regression adds an L2 regularization term 
to the ordinary least squares objective function, which helps prevent overfitting and handles 
multicollinearity in the predictors.

# Arguments
- `genomes::Genomes`: Genomic data structure containing genetic markers
- `phenomes::Phenomes`: Phenotypic data structure containing trait measurements
- `idx_entries::Union{Nothing,Vector{Int64}}`: Optional indices to subset specific entries/individuals
- `idx_loci_alleles::Union{Nothing,Vector{Int64}}`: Optional indices to subset specific loci-alleles
- `idx_trait::Int64`: Index of the trait to analyze (default: 1)
- `verbose::Bool`: If true, prints diagnostic plots and additional information (default: false)

# Returns
- `Fit`: A structure containing:
  - `model`: Model name ("ridge")
  - `b_hat`: Estimated coefficients (including intercept)
  - `b_hat_labels`: Labels for the coefficients
  - `metrics`: Performance metrics including correlation and error measures
  - `y_true`: Observed phenotypic values
  - `y_pred`: Predicted phenotypic values
  - Other model metadata

# Notes
- Uses cross-validation to select the optimal regularization parameter (λ)
- Standardizes predictors before fitting
- Includes an intercept term in the model

# Examples
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
function ridge(;
    genomes::Genomes,
    phenomes::Phenomes,
    idx_entries::Union{Nothing,Vector{Int64}} = nothing,
    idx_loci_alleles::Union{Nothing,Vector{Int64}} = nothing,
    idx_trait::Int64 = 1,
    verbose::Bool = false,
)::Fit
    # genomes = GBCore.simulategenomes()
    # trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.1 0.01 0.01;], verbose=false);
    # phenomes = extractphenomes(trials)
    # idx_entries = nothing; idx_loci_alleles = nothing
    # idx_trait = 1; verbose = true
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
    fit::Fit = Fit(n = size(X, 1), l = size(X, 2))
    fit.model = "ridge"
    fit.b_hat_labels = vcat(["intercept"], loci_alleles)
    fit.trait = phenomes.traits[idx_trait]
    fit.entries = entries
    fit.populations = populations
    fit.y_true = y
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
        display(UnicodePlots.histogram(glmnet_fit.meanloss))
        display(UnicodePlots.scatterplot(glmnet_fit.meanloss))
        display(UnicodePlots.scatterplot(glmnet_fit.lambda, glmnet_fit.meanloss))
        display(UnicodePlots.scatterplot(log10.(glmnet_fit.lambda), glmnet_fit.meanloss))
        println(string("argmin = ", argmin(glmnet_fit.meanloss)))
    end
    # Use the coefficients with variance
    b_hat::Vector{Float64} = zeros(size(X, 2) + 1)
    idx_sort = sortperm(glmnet_fit.meanloss)
    INTERCEPTSs = glmnet_fit.path.a0[idx_sort]
    # BETAs = glmnet_fit.path.betas[:, idx_sort]
    BETAs = glmnet_fit.path.betas
    i = 1
    while (var(b_hat[2:end]) < 1e-10) || (i == size(BETAs, 2))
        b_hat = vcat(INTERCEPTSs[idx_sort[i]], BETAs[:, idx_sort[i]])
        i += 1
    end
    # Assess prediction accuracy
    y_pred::Vector{Float64} = b_hat[1] .+ (X * b_hat[2:end])
    performance::Dict{String,Float64} = metrics(y, y_pred)
    if verbose
        display(UnicodePlots.scatterplot(y, y_pred))
        display(UnicodePlots.histogram(y))
        display(UnicodePlots.histogram(y_pred))
        println(performance)
    end
    # Output
    fit.b_hat = b_hat
    fit.metrics = performance
    fit.y_pred = y_pred
    if !checkdims(fit)
        throw(ErrorException("Error fitting " * fit.model * "."))
    end
    fit
end


"""
    lasso(;
        genomes::Genomes,
        phenomes::Phenomes,
        idx_entries::Union{Nothing,Vector{Int64}} = nothing,
        idx_loci_alleles::Union{Nothing,Vector{Int64}} = nothing,
        idx_trait::Int64 = 1,
        verbose::Bool = false,
    )::Fit

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
function lasso(;
    genomes::Genomes,
    phenomes::Phenomes,
    idx_entries::Union{Nothing,Vector{Int64}} = nothing,
    idx_loci_alleles::Union{Nothing,Vector{Int64}} = nothing,
    idx_trait::Int64 = 1,
    verbose::Bool = false,
)::Fit
    # genomes = GBCore.simulategenomes()
    # trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.1 0.01 0.01;], verbose=false);
    # phenomes = extractphenomes(trials)
    # idx_entries = nothing; idx_loci_alleles = nothing
    # idx_trait = 1; verbose = true
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
    fit::Fit = Fit(n = size(X, 1), l = size(X, 2))
    fit.model = "lasso"
    fit.b_hat_labels = vcat(["intercept"], loci_alleles)
    fit.trait = phenomes.traits[idx_trait]
    fit.entries = entries
    fit.populations = populations
    fit.y_true = y
    # Ridge regression using the glmnet package
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
    b_hat::Vector{Float64} = zeros(size(X, 2) + 1)
    idx_sort = sortperm(glmnet_fit.meanloss)
    INTERCEPTSs = glmnet_fit.path.a0[idx_sort]
    BETAs = glmnet_fit.path.betas[:, idx_sort]
    i = 1
    while var(b_hat[2:end]) < 1e-10
        b_hat = vcat(INTERCEPTSs[i], BETAs[:, i])
        i += 1
    end
    # Assess prediction accuracy
    y_pred::Vector{Float64} = b_hat[1] .+ (X * b_hat[2:end])
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
    fit.y_pred = y_pred
    if !checkdims(fit)
        throw(ErrorException("Error fitting " * fit.model * "."))
    end
    fit
end


"""
    bayesa(;
        genomes::Genomes,
        phenomes::Phenomes,
        idx_entries::Union{Nothing,Vector{Int64}} = nothing,
        idx_loci_alleles::Union{Nothing,Vector{Int64}} = nothing,
        idx_trait::Int64 = 1,
        n_iter::Int64 = 1_500,
        n_burnin::Int64 = 500,
        verbose::Bool = false,
    )::Fit

Fit a Bayes A model via BGLR

## Examples
```jldoctest; setup = :(using GBCore, GBModels)
julia> genomes = GBCore.simulategenomes(verbose=false);

julia> trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.1 0.01 0.01;], verbose=false);;

julia> phenomes = extractphenomes(trials);

julia> fit = bayesa(genomes=genomes, phenomes=phenomes);

julia> fit.model == "bayesa"
true

julia> fit.metrics["cor"] > 0.50
true
```
"""
function bayesa(;
    genomes::Genomes,
    phenomes::Phenomes,
    idx_entries::Union{Nothing,Vector{Int64}} = nothing,
    idx_loci_alleles::Union{Nothing,Vector{Int64}} = nothing,
    idx_trait::Int64 = 1,
    n_iter::Int64 = 1_500,
    n_burnin::Int64 = 500,
    verbose::Bool = false,
)::Fit
    fit = bayesian(
        "BayesA",
        genomes = genomes,
        phenomes = phenomes,
        idx_entries = idx_entries,
        idx_loci_alleles = idx_loci_alleles,
        idx_trait = idx_trait,
        n_iter = n_iter,
        n_burnin = n_burnin,
        response_type = "gaussian",
        verbose = verbose,
    )
    fit.model = "bayesa"
    fit
end


"""
    bayesb(;
        genomes::Genomes,
        phenomes::Phenomes,
        idx_entries::Union{Nothing,Vector{Int64}} = nothing,
        idx_loci_alleles::Union{Nothing,Vector{Int64}} = nothing,
        idx_trait::Int64 = 1,
        n_iter::Int64 = 1_500,
        n_burnin::Int64 = 500,
        verbose::Bool = false,
    )::Fit

Fit a Bayes A model via BGLR

## Examples
```jldoctest; setup = :(using GBCore, GBModels)
julia> genomes = GBCore.simulategenomes(verbose=false);

julia> trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.1 0.01 0.01;], verbose=false);;

julia> phenomes = extractphenomes(trials);

julia> fit = bayesb(genomes=genomes, phenomes=phenomes);

julia> fit.model == "bayesb"
true

julia> fit.metrics["cor"] > 0.50
true
```
"""
function bayesb(;
    genomes::Genomes,
    phenomes::Phenomes,
    idx_entries::Union{Nothing,Vector{Int64}} = nothing,
    idx_loci_alleles::Union{Nothing,Vector{Int64}} = nothing,
    idx_trait::Int64 = 1,
    n_iter::Int64 = 1_500,
    n_burnin::Int64 = 500,
    verbose::Bool = false,
)::Fit
    fit = bayesian(
        "BayesB",
        genomes = genomes,
        phenomes = phenomes,
        idx_entries = idx_entries,
        idx_loci_alleles = idx_loci_alleles,
        idx_trait = idx_trait,
        n_iter = n_iter,
        n_burnin = n_burnin,
        response_type = "gaussian",
        verbose = verbose,
    )
    fit.model = "bayesb"
    fit
end


"""
    bayesc(;
        genomes::Genomes,
        phenomes::Phenomes,
        idx_entries::Union{Nothing,Vector{Int64}} = nothing,
        idx_loci_alleles::Union{Nothing,Vector{Int64}} = nothing,
        idx_trait::Int64 = 1,
        n_iter::Int64 = 1_500,
        n_burnin::Int64 = 500,
        verbose::Bool = false,
    )::Fit

Fit a Bayes A model via BGLR

## Examples
```jldoctest; setup = :(using GBCore, GBModels)
julia> genomes = GBCore.simulategenomes(verbose=false);

julia> trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.1 0.01 0.01;], verbose=false);;

julia> phenomes = extractphenomes(trials);

julia> fit = bayesc(genomes=genomes, phenomes=phenomes);

julia> fit.model == "bayesc"
true

julia> fit.metrics["cor"] > 0.50
true
```
"""
function bayesc(;
    genomes::Genomes,
    phenomes::Phenomes,
    idx_entries::Union{Nothing,Vector{Int64}} = nothing,
    idx_loci_alleles::Union{Nothing,Vector{Int64}} = nothing,
    idx_trait::Int64 = 1,
    n_iter::Int64 = 1_500,
    n_burnin::Int64 = 500,
    verbose::Bool = false,
)::Fit
    fit = bayesian(
        "BayesC",
        genomes = genomes,
        phenomes = phenomes,
        idx_entries = idx_entries,
        idx_loci_alleles = idx_loci_alleles,
        idx_trait = idx_trait,
        n_iter = n_iter,
        n_burnin = n_burnin,
        response_type = "gaussian",
        verbose = verbose,
    )
    fit.model = "bayesc"
    fit
end
