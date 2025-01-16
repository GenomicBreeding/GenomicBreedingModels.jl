"""
    extractxyetc(;
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

julia> X, y, loci_alleles = extractxyetc(genomes, phenomes);

julia> X == hcat(ones(length(phenomes.entries)), genomes.allele_frequencies)
true

julia> y == phenomes.phenotypes[:, 1]
true
```
"""
function extractxyetc(
    genomes::Genomes,
    phenomes::Phenomes;
    idx_entries::Union{Nothing,Vector{Int64}} = nothing,
    idx_loci_alleles::Union{Nothing,Vector{Int64}} = nothing,
    idx_trait::Int64 = 1,
    add_intercept::Bool = true,
)::Tuple{
    Matrix{Float64},
    Vector{Float64},
    Vector{String},
    Vector{String},
    Vector{String}}
    # genomes = GBCore.simulategenomes()
    # trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.1 0.01 0.01;], verbose=false);
    # phenomes = extractphenomes(trials)
    # idx_entries = nothing, idx_loci_alleles = nothing
    # idx_trait = 1; add_intercept = true
    # Check arguments
    if !checkdims(genomes) && !checkdims(phenomes)
        throw(ArgumentError("The Genomes and Phenomes structs are corrupted."))
    end
    if !checkdims(genomes)
        throw(ArgumentError("The Genomes struct is corrupted."))
    end
    if !checkdims(phenomes)
        throw(ArgumentError("The Phenomes struct is corrupted."))
    end
    if genomes.entries != phenomes.entries
        throw(ArgumentError("The genomes and phenomes input need to have been merged to have consitent entries."))
    end
    if isnothing(idx_entries)
        idx_entries = collect(1:length(genomes.entries))
    else
        if (minimum(idx_entries) < 1) .|| maximum(idx_entries) > length(genomes.entries)
            throw(
                ArgumentError(
                    "The indexes of the entries, `idx_entries` are out of bounds. Expected range: from 1 to " *
                    string(length(genomes.entries)) *
                    " while the supplied range is from " *
                    string(minimum(idx_entries)) *
                    " to " *
                    string(maximum(idx_entries)) *
                    ".",
                ),
            )
        end
    end
    if isnothing(idx_loci_alleles)
        idx_loci_alleles = collect(1:length(genomes.loci_alleles))
    else
        if (minimum(idx_loci_alleles) < 1) .|| maximum(idx_loci_alleles) > length(genomes.loci_alleles)
            throw(
                ArgumentError(
                    "The indexes of the loci_alleles, `idx_loci_alleles` are out of bounds. Expected range: from 1 to " *
                    string(length(genomes.loci_alleles)) *
                    " while the supplied range is from " *
                    string(minimum(idx_loci_alleles)) *
                    " to " *
                    string(maximum(idx_loci_alleles)) *
                    ".",
                ),
            )
        end
    end
    # Extract the response variable
    y::Vector{Float64} = phenomes.phenotypes[idx_entries, idx_trait]
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
    G::Matrix{Float64} = genomes.allele_frequencies[idx_entries[idx], idx_loci_alleles]
    # Output X with/without intercept, y, names of the entries, populations, and loci-alleles
    entries = genomes.entries[idx_entries[idx]]
    populations = genomes.populations[idx_entries[idx]]
    loci_alleles = genomes.loci_alleles[idx_loci_alleles]
    if add_intercept
        return (hcat(ones(length(idx)), G), y, entries, populations, loci_alleles)
    else
        return (G, y, entries, populations, loci_alleles)
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
    fit = Fit(n=size(X,1), l = size(X, 2) + 1)
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
    fit = Fit(n=size(X,1), l = size(X, 2))
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