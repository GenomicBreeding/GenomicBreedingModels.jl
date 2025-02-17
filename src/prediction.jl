"""
    extractxyetc(
        genomes::Genomes,
        phenomes::Phenomes;
        idx_entries::Union{Nothing,Vector{Int64}} = nothing,
        idx_loci_alleles::Union{Nothing,Vector{Int64}} = nothing,
        idx_trait::Int64 = 1,
        add_intercept::Bool = true,
    )::Tuple{Matrix{Float64},Vector{Float64},Vector{String},Vector{String},Vector{String}}

Extract explanatory `X` matrix, response `y` vector, names of the entries, populations and loci-alleles from genomes and phenomes.

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
)::Tuple{Matrix{Float64},Vector{Float64},Vector{String},Vector{String},Vector{String}}
    # genomes = GBCore.simulategenomes()
    # trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.1 0.01 0.01;], verbose=false);
    # phenomes = extractphenomes(trials)
    # idx_entries = nothing; idx_loci_alleles = nothing
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
    # Omit entries missing, nan and infinite phenotype data
    idx::Vector{Int64} = findall(.!ismissing.(y) .&& .!isnan.(y) .&& .!isinf.(y))
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
    predict(; fit::Fit, genomes::Genomes, idx_entries::Vector{Int64})::Vector{Float64}

Predict the phenotypes given a genomic prediction model fit, a genomes and the corresponding entries indexes

# Examples
```jldoctest; setup = :(using GBCore, GBModels, StatsBase)
julia> genomes = GBCore.simulategenomes(verbose=false);

julia> trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.1 0.01 0.01;], verbose=false);

julia> phenomes = extractphenomes(trials);

julia> fit = ridge(genomes=genomes, phenomes=phenomes, idx_entries=collect(1:90));

julia> y_hat = GBModels.predict(fit=fit, genomes=genomes, idx_entries=collect(91:100));

julia> cor(phenomes.phenotypes[91:100, 1], y_hat) > 0.5
true
```
"""
function predict(; fit::Fit, genomes::Genomes, idx_entries::Vector{Int64})::Vector{Float64}
    # genomes = GBCore.simulategenomes(n=300, verbose=false); genomes.populations = StatsBase.sample(string.("pop_", 1:3), length(genomes.entries), replace=true);
    # trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, verbose=false);
    # phenomes = extractphenomes(trials)
    # fit = ridge(genomes=genomes, phenomes=phenomes, idx_entries=collect(1:200))
    # idx_entries = collect(201:300)
    # Check arguments
    if !checkdims(fit)
        throw(ArgumentError("The Fit struct is corrupted."))
    end
    if !checkdims(genomes)
        throw(ArgumentError("The Genomes struct is corrupted."))
    end
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
    idx_loci_alleles = try
        [findall(genomes.loci_alleles .== x)[1] for x in fit.b_hat_labels[2:end]]
    catch
        throw(
            ArgumentError(
                "The loci-alleles in the fitted genomic prediction model do not match the loci-alleles in the requested validation set.",
            ),
        )
    end
    # Linear or non-linear model prediction
    linear_models = ["ols", "ridge", "lasso", "bayesa", "bayesb", "bayesc"]
    non_linear_models = [""]
    if sum(linear_models .== fit.model) == 1
        return fit.b_hat[1] .+ (genomes.allele_frequencies[idx_entries, idx_loci_alleles] * fit.b_hat[2:end])
    elseif sum(non_linear_models .== fit.model) == 1
        # TODO: once we have non-linear models
        return NaN
    else
        throw(ArgumentError("Unrecognised genomic prediction model: `" * fit.model * "`."))
    end
end
