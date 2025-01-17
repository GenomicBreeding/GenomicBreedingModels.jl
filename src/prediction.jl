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
