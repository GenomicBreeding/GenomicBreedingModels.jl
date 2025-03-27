"""
    validate(
        fit::Fit,
        genomes::Genomes,
        phenomes::Phenomes;
        idx_validation::Vector{Int64},
        replication::String="",
        fold::String=""
    )::CV

Evaluate the predictive accuracy of a genomic prediction model on a validation dataset.

# Arguments
- `fit::Fit`: A fitted genomic prediction model
- `genomes::Genomes`: Genomic data structure containing allele frequencies
- `phenomes::Phenomes`: Phenotypic data structure containing trait measurements
- `idx_validation::Vector{Int64}`: Indices of entries to use for validation
- `replication::String`: Optional identifier for the validation replication
- `fold::String`: Optional identifier for the cross-validation fold

# Returns
- `CV`: A cross-validation result object containing:
  - Validation metrics (correlation, RMSE, etc.)
  - True and predicted values
  - Entry and population information
  - Model specifications

# Notes
- Performs checks for data leakage between training and validation sets
- Handles missing, NaN, and Inf values in phenotypic data
- Validates dimensions of output CV struct

# Examples
```jldoctest; setup = :(using GenomicBreedingCore, GenomicBreedingModels)
julia> genomes = GenomicBreedingCore.simulategenomes(verbose=false);

julia> trials, _ = GenomicBreedingCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.1 0.01 0.01;], verbose=false);

julia> phenomes = extractphenomes(trials);

julia> fit = ridge(genomes=genomes, phenomes=phenomes, idx_entries=collect(1:90));

julia> cv = validate(fit, genomes, phenomes, idx_validation=collect(91:100));

julia> cv.metrics["cor"] > 0.50
true
```
"""
function validate(
    fit::Fit,
    genomes::Genomes,
    phenomes::Phenomes;
    idx_validation::Vector{Int64},
    replication::String = "",
    fold::String = "",
)::CV
    # genomes = GenomicBreedingCore.simulategenomes(n=300, verbose=false); genomes.populations = StatsBase.sample(string.("pop_", 1:3), length(genomes.entries), replace=true);
    # trials, _ = GenomicBreedingCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, verbose=false);
    # phenomes = extractphenomes(trials)
    # fit = ridge(genomes=genomes, phenomes=phenomes, idx_entries=collect(1:200))
    # idx_validation = collect(201:300); replication = ""; fold = "";
    idx_trait = findall(phenomes.traits .== fit.trait)[1]
    data_leakage = intersect(fit.entries, phenomes.entries[idx_validation])
    if length(data_leakage) > 0
        throw(
            ArgumentError(
                "Data leakage between training and validation sets, i.e. entries:\n\t‣ " * join(data_leakage, "\n\t‣ "),
            ),
        )
    end
    ϕ = phenomes.phenotypes[idx_validation, idx_trait]
    idx = findall(.!ismissing.(ϕ) .&& .!isnan.(ϕ) .&& .!isinf.(ϕ))
    populations = phenomes.populations[idx_validation[idx]]
    entries = phenomes.entries[idx_validation[idx]]
    y_true::Vector{Float64} = ϕ[idx]
    y_pred::Vector{Float64} =
        GenomicBreedingModels.predict(fit = fit, genomes = genomes, idx_entries = idx_validation[idx])
    performance = metrics(y_true, y_pred)
    cv = CV(replication, fold, fit, populations, entries, y_true, y_pred, performance)
    if !checkdims(cv)
        throw(ArgumentError("CV struct is corrupted."))
    end
    cv
end

"""
    cvmultithread!(cvs::Vector{CV}; genomes::Genomes, phenomes::Phenomes, models_vector, verbose::Bool = true)::Vector{CV}

Perform multi-threaded genomic prediction cross-validation using specified models.

# Arguments
- `cvs::Vector{CV}`: Vector of cross-validation objects to be processed
- `genomes::Genomes`: Genomic data structure containing allele frequencies
- `phenomes::Phenomes`: Phenotypic data structure containing trait measurements
- `models_vector`: Vector of model functions to be used for prediction (e.g., [ridge, bayesa])
- `verbose::Bool=true`: Whether to display progress bar during computation

# Returns
- Modified `cvs` vector with updated cross-validation results

# Threading
Requires Julia to be started with multiple threads to utilize parallel processing.
Example startup command: `julia --threads 7,1` (7 worker threads, 1 runtime thread)

# Details
The function performs cross-validation in parallel for each CV object using the corresponding model
from the models_vector. For each fold:
1. Extracts training and validation indices
2. Fits the specified model using training data
3. Validates the model using validation data
4. Updates the CV object with prediction results

# Example
```jldoctest; setup = :(using GenomicBreedingCore, GenomicBreedingModels, StatsBase)
julia> genomes = GenomicBreedingCore.simulategenomes(verbose=false);

julia> trials, _ = GenomicBreedingCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.1 0.01 0.01;], verbose=false);

julia> phenomes = extractphenomes(trials);

julia> idx_training = collect(1:50);

julia> idx_validation_1 = collect(51:75);

julia> idx_validation_2 = collect(76:100);

julia> fit = Fit(n = length(idx_training), l = length(genomes.loci_alleles) + 1); fit.model = "ridge"; fit.trait = "trait_1"; 

julia> fit.entries = genomes.entries[idx_training]; fit.populations = genomes.populations[idx_training]; 

julia> fit.b_hat_labels = vcat(["intercept"], genomes.loci_alleles);

julia> cv_1 = CV("replication_1", "fold_1", fit, genomes.populations[idx_validation_1], genomes.entries[idx_validation_1], zeros(length(idx_validation_1)), zeros(length(idx_validation_1)), fit.metrics);

julia> cv_2 = CV("replication_1", "fold_2", fit, genomes.populations[idx_validation_2], genomes.entries[idx_validation_2], zeros(length(idx_validation_2)), zeros(length(idx_validation_2)), fit.metrics);

julia> cvs = [cv_1, cv_2]; models = [ridge, ridge];

julia> cvmultithread!(cvs, genomes=genomes, phenomes=phenomes, models_vector=[ridge, bayesa], verbose=false);

julia> df_across_entries, df_per_entry = tabularise(cvs);

julia> idx_across = findall(df_across_entries.fold .== "fold_2");

julia> idx_per = findall(df_per_entry.fold .== "fold_2");

julia> abs(df_across_entries.cor[idx_across][1] - cor(df_per_entry.y_true[idx_per], df_per_entry.y_pred[idx_per])) < 1e-10
true
```
"""
function cvmultithread!(cvs::Vector{CV}; genomes::Genomes, phenomes::Phenomes, models_vector, verbose::Bool = true)
    # Cross-validate using all thread/s available to Julia (set at startup via julia --threads 2,1 --load test/interactive_prelude.jl)
    m = length(cvs)
    # Multi-threaded CV for non-BGLR models
    if verbose
        pb = Progress(m; desc = "Multi-threaded genomic prediction replicated cross-validation: ")
    end
    thread_lock::ReentrantLock = ReentrantLock()
    Threads.@threads for i = 1:m
        # i = 1
        model = models_vector[i]
        idx_training = [findall(genomes.entries .== x)[1] for x in cvs[i].fit.entries]
        idx_validation = [findall(genomes.entries .== x)[1] for x in cvs[i].validation_entries]
        idx_loci_alleles = [findall(genomes.loci_alleles .== x)[1] for x in cvs[i].fit.b_hat_labels[2:end]]
        idx_trait = findall(phenomes.traits .== cvs[i].fit.trait)[1]
        replication = cvs[i].replication
        fold = cvs[i].fold
        try
            fit = model(
                genomes = genomes,
                phenomes = phenomes,
                idx_entries = idx_training,
                idx_loci_alleles = idx_loci_alleles,
                idx_trait = idx_trait,
                verbose = false,
            )
            cv = validate(
                fit,
                genomes,
                phenomes,
                idx_validation = idx_validation,
                replication = replication,
                fold = fold,
            )
            @lock thread_lock cvs[i] = cv
        catch
            @warn string(
                "Oh naur! This is unexpected multi-threaded model fitting error! Model: ",
                model,
                "; i: ",
                i,
                ". If you're a dev, please inspect the `",
                model,
                "(...)` and the `validate(...)` functions.",
            )
            continue
        end
        if verbose
            next!(pb)
        end
    end
    if verbose
        finish!(pb)
    end
    cvs
end


"""
    cvbulk(;
        genomes::Genomes,
        phenomes::Phenomes,
        models::Vector{Function}=[ridge],
        n_replications::Int64=5,
        n_folds::Int64=5,
        seed::Int64=42,
        verbose::Bool=true
    )::Tuple{Vector{CV}, Vector{String}}

Perform replicated k-fold cross-validation of genomic prediction model(s) across all available traits
and entries, ignoring populations.

# Arguments
- `genomes::Genomes`: Genomic data structure containing allele frequencies
- `phenomes::Phenomes`: Phenotypic data structure containing trait measurements
- `models::Vector{Function}`: Vector of genomic prediction model functions to evaluate
- `n_replications::Int64`: Number of times to repeat k-fold cross-validation randomising k-fold partitioning each time (default: 5)
- `n_folds::Int64`: Number of cross-validation folds (default: 5) 
- `seed::Int64`: Random seed for reproducibility (default: 42)
- `verbose::Bool`: Whether to display progress information (default: true)

# Returns
- Tuple containing:
  - Vector of CV objects with cross-validation results
  - Vector of warning messages about skipped cases

# Threading
Uses multiple threads if Julia is started with threading enabled.
Example startup command: `julia --threads 7,1` (7 worker threads, 1 runtime thread)

# Notes
- Performs random k-fold partitioning of entries ignoring population structure
- Handles missing/invalid phenotype values
- Validates model inputs and data dimensions
- Returns warnings for cases with insufficient data or zero variance

# Example
```jldoctest; setup = :(using GenomicBreedingCore, GenomicBreedingModels, StatsBase)
julia> genomes = GenomicBreedingCore.simulategenomes(verbose=false); genomes.populations = StatsBase.sample(string.("pop_", 1:3), length(genomes.entries), replace=true);

julia> trials, _ = GenomicBreedingCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, verbose=false);

julia> phenomes = extractphenomes(trials);

julia> cvs, notes = cvbulk(genomes=genomes, phenomes=phenomes, models=[ols, ridge], n_replications=2, n_folds=2, verbose=false);

julia> df_across_entries, df_per_entry = tabularise(cvs);

julia> idx_across = findall((df_across_entries.trait .== "trait_1") .&& (df_across_entries.model .== "ridge") .&& (df_across_entries.replication .== "replication_1") .&& (df_across_entries.fold .== "fold_1"));

julia> idx_per = findall((df_per_entry.trait .== "trait_1") .&& (df_per_entry.model .== "ridge") .&& (df_per_entry.replication .== "replication_1") .&& (df_per_entry.fold .== "fold_1"));

julia> abs(df_across_entries.cor[idx_across][1] - cor(df_per_entry.y_true[idx_per], df_per_entry.y_pred[idx_per])) < 1e-10
true
```
"""
function cvbulk(;
    genomes::Genomes,
    phenomes::Phenomes,
    models = [ridge], ### Puting type annotation of Vector{Function} fails when the vector contains a single function
    n_replications::Int64 = 5,
    n_folds::Int64 = 5,
    seed::Int64 = 42,
    verbose::Bool = true,
)::Tuple{Vector{CV},Vector{String}}
    # genomes = GenomicBreedingCore.simulategenomes(n=300, verbose=false); genomes.populations = StatsBase.sample(string.("pop_", 1:3), length(genomes.entries), replace=true);
    # trials, _ = GenomicBreedingCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, verbose=false);
    # phenomes = extractphenomes(trials)
    # models = [ridge, lasso]
    # n_folds = 2; n_replications = 2; seed = 42; verbose = true
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
    mask_genomes = findall(mean(genomes.mask, dims = 2)[:, 1] .== 1.0)
    mask_phenomes = findall(mean(phenomes.mask, dims = 2)[:, 1] .== 1.0)
    if mask_genomes != mask_phenomes
        throw(ArgumentError("The masks in genomes and phenomes do not match."))
    end
    if length(models) < 1
        throw(ArgumentError("No models were specified."))
    end
    valid_models = [ols, ridge, lasso, bayesa, bayesb, bayesc]
    for model in models
        # model = models[1]
        if !isa(model, Function) && sum(valid_models .== model) > 0
            throw(
                ArgumentError(
                    "The supplied model: `" *
                    string(model) *
                    "` is not a valid genomic prediction model function. Please choose from:\n\t‣ " *
                    join(string.(valid_models), "\n\t‣ "),
                ),
            )
        end
    end
    n, p = size(genomes.allele_frequencies) # dimensions prior to potential filtering via masks below
    if (n_folds < 1) || (n_folds > n)
        throw(
            ArgumentError(
                "The number of folds, `n_folds = " *
                string(n_folds) *
                "` is out of bounds. Please use values from 1 to " *
                string(n) *
                ".",
            ),
        )
    end
    if (n_replications < 1) || (n_replications > 100)
        throw(
            ArgumentError(
                "The number of replications, `n_replications = " *
                string(n_replications) *
                "` is out of bounds. Please use values from 1 to 100.",
            ),
        )
    end
    # Apply mask
    mask_loci_alleles = findall(mean(genomes.mask, dims = 1)[1, :] .== 1.0)
    if (mask_genomes != collect(1:n)) || (mask_loci_alleles != collect(1:p))
        genomes = filter(genomes)
        phenomes = filter(phenomes)
    end
    # Set randomisation seed
    rng::TaskLocalRNG = Random.seed!(seed)
    # Instantiate the entire replicated cross-validation vector of output
    n, p = size(genomes.allele_frequencies)
    cvs::Vector{CV} = []
    notes::Vector{String} = []
    models_vector::Vector{Function} = [] # for ease of function calling in multi-threaded function below, i.e. `cvmultithread!(...)`
    if verbose
        k = length(phenomes.traits) * n_replications * n_folds * length(models)
        pb = Progress(k; desc = "Setting up cross-validation job/s: ")
    end
    for trait in phenomes.traits
        # trait = phenomes.traits[1]
        for i = 1:n_replications
            # i = 1
            idx_permutation = StatsBase.sample(rng, 1:n_folds, n, replace = true)
            for j = 1:n_folds
                # j = 1
                # Check the phenotypes early before we partition them into threads. Much more efficient eh!
                ϕ = phenomes.phenotypes[:, phenomes.traits.==trait][:, 1]
                idx_training = findall((idx_permutation .!= j) .&& .!ismissing.(ϕ) .&& .!isnan.(ϕ) .&& .!isinf.(ϕ))
                idx_validation = findall((idx_permutation .== j) .&& .!ismissing.(ϕ) .&& .!isnan.(ϕ) .&& .!isinf.(ϕ))
                if (length(idx_training) < 2 || length(idx_validation) < 1)
                    push!(notes, join(["too_many_missing", trait, string("replication_", i), string("fold_", j)], ";"))
                    continue
                end
                if var(ϕ[idx_training]) < 1e-20
                    push!(notes, join(["zero_variance", trait, string("replication_", i), string("fold_", j)], ";"))
                    continue
                end
                for model in models
                    # model = models[1]
                    fit = Fit(n = length(idx_training), l = p + 1)
                    fit.model = string(model)
                    fit.trait = trait
                    fit.entries = genomes.entries[idx_training]
                    fit.populations = genomes.populations[idx_training]
                    fit.b_hat_labels = vcat(["intercept"], genomes.loci_alleles)
                    cv = CV(
                        string("replication_", i),
                        string("fold_", j),
                        fit,
                        genomes.populations[idx_validation],
                        genomes.entries[idx_validation],
                        zeros(length(idx_validation)),
                        zeros(length(idx_validation)),
                        fit.metrics,
                    )
                    # Update the vectors of CVs and models
                    push!(cvs, cv)
                    push!(models_vector, model)
                    if verbose
                        next!(pb)
                    end
                end
            end
        end
    end
    if verbose
        finish!(pb)
    end
    # Multi-threaded cross-validation
    if verbose
        println("Setup ", length(cvs), " cross-validation job/s.")
        println("Skipping ", length(notes), " cross-validation job/s.")
    end
    try
        cvmultithread!(cvs, genomes = genomes, phenomes = phenomes, models_vector = models_vector, verbose = verbose)
    catch
        throw(
            ErrorException(
                "Error performing bulk cross-validation across population/s:\n\t‣ " *
                join(sort(unique(phenomes.populations)), "\n\t‣ "),
            ),
        )
    end
    # output
    (cvs, notes)
end

"""
    cvperpopulation(;
        genomes::Genomes,
        phenomes::Phenomes,
        models::Vector{Function}=[ridge],
        n_replications::Int64=5,
        n_folds::Int64=5,
        seed::Int64=42,
        verbose::Bool=true
    )::Tuple{Vector{CV}, Vector{String}}

Performs within-population replicated cross-validation of genomic prediction models across all available traits.

# Arguments
- `genomes::Genomes`: Genomic data structure containing allele frequencies
- `phenomes::Phenomes`: Phenotypic data structure containing trait measurements
- `models::Vector{Function}=[ridge]`: Vector of genomic prediction model functions to evaluate
- `n_replications::Int64=5`: Number of replications for cross-validation
- `n_folds::Int64=5`: Number of folds for k-fold cross-validation
- `seed::Int64=42`: Random seed for reproducibility
- `verbose::Bool=true`: Whether to print progress information

# Returns
- `Tuple{Vector{CV}, Vector{String}}`: A tuple containing:
    - Vector of CV objects with cross-validation results
    - Vector of notes/warnings generated during the process

# Details
The function performs separate cross-validations for each unique population in the dataset.
Supports multiple genomic prediction models including:
- `ols`: Ordinary Least Squares
- `ridge`: Ridge Regression
- `lasso`: Lasso Regression
- `bayesa`: Bayes A
- `bayesb`: Bayes B
- `bayesc`: Bayes C

# Threading
To use multiple threads, invoke Julia with: `julia --threads n,1` where n is the desired number of threads
for multi-threaded processes and 1 is reserved for the Julia runtime.

# Examples
```jldoctest; setup = :(using GenomicBreedingCore, GenomicBreedingModels, StatsBase; import GenomicBreedingModels: ridge)
julia> genomes = GenomicBreedingCore.simulategenomes(l=1_000, verbose=false); genomes.populations = StatsBase.sample(string.("pop_", 1:3), length(genomes.entries), replace=true);

julia> trials, _ = GenomicBreedingCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.1 0.01 0.01;], verbose=false);

julia> phenomes = extractphenomes(trials);

julia> cvs, notes = cvperpopulation(genomes=genomes, phenomes=phenomes, models=[ridge, bayesa], n_replications=2, n_folds=2, verbose=false);

julia> df_across_entries, df_per_entry = tabularise(cvs);

julia> sort(unique(df_across_entries.training_population))
3-element Vector{String}:
 "pop_1"
 "pop_2"
 "pop_3"

julia> df_across_entries.training_population == df_across_entries.validation_population
true

julia> idx_across = findall((df_across_entries.validation_population .== "pop_1") .&& (df_across_entries.trait .== "trait_1") .&& (df_across_entries.model .== "bayesa") .&& (df_across_entries.replication .== "replication_1") .&& (df_across_entries.fold .== "fold_1"));

julia> idx_per = findall((df_per_entry.validation_population .== "pop_1") .&& (df_per_entry.trait .== "trait_1") .&& (df_per_entry.model .== "bayesa") .&& (df_per_entry.replication .== "replication_1") .&& (df_per_entry.fold .== "fold_1"));

julia> abs(df_across_entries.cor[idx_across][1] - cor(df_per_entry.y_true[idx_per], df_per_entry.y_pred[idx_per])) < 1e-10
true

julia> summary_across, summary_per_entry = summarise(cvs);

julia> size(summary_across)
(6, 8)

julia> size(summary_per_entry)
(200, 8)
```
"""
function cvperpopulation(;
    genomes::Genomes,
    phenomes::Phenomes,
    models = [ridge], ### Puting type annotation of Vector{Function} fails when the vector contains a single function
    n_replications::Int64 = 5,
    n_folds::Int64 = 5,
    seed::Int64 = 42,
    verbose::Bool = true,
)::Tuple{Vector{CV},Vector{String}}
    # genomes = GenomicBreedingCore.simulategenomes(n=300, verbose=false); genomes.populations = StatsBase.sample(string.("pop_", 1:3), length(genomes.entries), replace=true);
    # trials, _ = GenomicBreedingCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, verbose=false);
    # phenomes = extractphenomes(trials)
    # models = [ridge, lasso]
    # n_folds = 2; n_replications = 2; seed = 42; verbose = true
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
    mask_genomes = findall(mean(genomes.mask, dims = 2)[:, 1] .== 1.0)
    mask_phenomes = findall(mean(phenomes.mask, dims = 2)[:, 1] .== 1.0)
    if mask_genomes != mask_phenomes
        throw(ArgumentError("The masks in genomes and phenomes do not match."))
    end
    if length(models) < 1
        throw(ArgumentError("No models were specified."))
    end
    valid_models = [ols, ridge, lasso, bayesa, bayesb, bayesc]
    for model in models
        # model = models[1]
        if !isa(model, Function) && sum(valid_models .== model) > 0
            throw(
                ArgumentError(
                    "The supplied model: `" *
                    string(model) *
                    "` is not a valid genomic prediction model function. Please choose from:\n\t‣ " *
                    join(string.(valid_models), "\n\t‣ "),
                ),
            )
        end
    end
    if verbose
        println("Cross-validations per population")
    end
    populations = sort(unique(genomes.populations))
    cvs::Vector{CV} = []
    notes::Vector{String} = []
    for population in populations
        # population = populations[1]
        idx_entries = findall(phenomes.populations .== population)
        idx_loci_alleles = collect(1:length(genomes.loci_alleles))
        idx_traits = collect(1:length(phenomes.traits))
        if verbose
            println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            println(
                string(
                    "Population: ",
                    population,
                    "; n = ",
                    length(idx_entries),
                    "; p = ",
                    length(idx_loci_alleles),
                    "; t = ",
                    length(idx_traits),
                ),
            )
            println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        end
        try
            cvs_per_pop, notes_per_pop = cvbulk(
                genomes = slice(genomes, idx_entries = idx_entries, idx_loci_alleles = idx_loci_alleles),
                phenomes = slice(phenomes, idx_entries = idx_entries, idx_traits = idx_traits),
                models = models,
                n_replications = n_replications,
                n_folds = n_folds,
                seed = seed,
                verbose = verbose,
            )
            append!(cvs, cvs_per_pop)
            append!(notes, notes_per_pop)
        catch
            @warn string("Oh naur! This is unexpected per population cross-validation error! Population: ", population)
            continue
        end
    end
    (cvs, notes)
end

"""
    cvpairwisepopulation(;
        genomes::Genomes,
        phenomes::Phenomes,
        models::Vector{Function}=[ridge],
        n_replications::Int64=5,
        n_folds::Int64=5, 
        seed::Int64=42,
        verbose::Bool=true
    )::Tuple{Vector{CV}, Vector{String}}

Performs pairwise cross-validation between populations for genomic prediction models.

# Arguments
- `genomes::Genomes`: Genomic data structure containing allele frequencies
- `phenomes::Phenomes`: Phenotypic data structure containing trait measurements
- `models::Vector{Function}`: Vector of genomic prediction model functions to evaluate (default: [ridge])
- `n_replications::Int64`: Number of replications (unused, maintained for API consistency)
- `n_folds::Int64`: Number of folds (unused, maintained for API consistency)  
- `seed::Int64`: Random seed (unused, maintained for API consistency)
- `verbose::Bool`: Whether to display progress information

# Returns
- `Tuple{Vector{CV}, Vector{String}}`: 
    - Vector of CV objects containing cross-validation results
    - Vector of warning messages for skipped validations

# Details
For each pair of populations (pop1, pop2):
1. Uses pop1 as training set and pop2 as validation set
2. Skips within-population validation (pop1==pop2)
3. Evaluates each model on all available traits
4. Handles missing/invalid phenotype values
5. Validates model inputs and data dimensions

# Threading
Requires Julia to be started with multiple threads:
`julia --threads n,1` where n is number of worker threads

# Examples
```jldoctest; setup = :(using GenomicBreedingCore, GenomicBreedingModels, StatsBase)
julia> genomes = GenomicBreedingCore.simulategenomes(verbose=false); genomes.populations = StatsBase.sample(string.("pop_", 1:3), length(genomes.entries), replace=true);

julia> trials, _ = GenomicBreedingCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.1 0.01 0.01;], verbose=false);

julia> phenomes = extractphenomes(trials);

julia> cvs, notes = cvpairwisepopulation(genomes=genomes, phenomes=phenomes, models=[ols, ridge], n_replications=2, n_folds=2, verbose=false);

julia> df_across_entries, df_per_entry = tabularise(cvs);

julia> sum(df_across_entries.training_population .!= df_across_entries.validation_population) == size(df_across_entries, 1)
true

julia> idx_across = findall((df_across_entries.training_population .== "pop_1") .&& (df_across_entries.validation_population .== "pop_2") .&& (df_across_entries.trait .== "trait_1") .&& (df_across_entries.model .== "ridge"));

julia> idx_per = findall((df_per_entry.training_population .== "pop_1") .&& (df_per_entry.validation_population .== "pop_2") .&& (df_per_entry.trait .== "trait_1") .&& (df_per_entry.model .== "ridge"));

julia> abs(df_across_entries.cor[idx_across][1] - cor(df_per_entry.y_true[idx_per], df_per_entry.y_pred[idx_per])) < 1e-10
true
```
"""
function cvpairwisepopulation(;
    genomes::Genomes,
    phenomes::Phenomes,
    models = [ridge], ### Puting type annotation of Vector{Function} fails when the vector contains a single function
    n_replications::Int64 = 5, ### Unused as no replication is required because we train and validate on the entire respective populations
    n_folds::Int64 = 5, ### Unused as no replication is required because we train and validate on the entire respective populations
    seed::Int64 = 42, ### Unused as no replication is required because we train and validate on the entire respective populations
    verbose::Bool = true,
)::Tuple{Vector{CV},Vector{String}}
    # genomes = GenomicBreedingCore.simulategenomes(n=300, verbose=false); genomes.populations = StatsBase.sample(string.("pop_", 1:3), length(genomes.entries), replace=true);
    # trials, _ = GenomicBreedingCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, verbose=false);
    # phenomes = extractphenomes(trials)
    # models = [ridge, lasso]
    # n_folds = 2; n_replications = 2; seed = 42; verbose = true
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
    mask_genomes = findall(mean(genomes.mask, dims = 2)[:, 1] .== 1.0)
    mask_phenomes = findall(mean(phenomes.mask, dims = 2)[:, 1] .== 1.0)
    if mask_genomes != mask_phenomes
        throw(ArgumentError("The masks in genomes and phenomes do not match."))
    end
    if length(models) < 1
        throw(ArgumentError("No models were specified."))
    end
    valid_models = [ols, ridge, lasso, bayesa, bayesb, bayesc]
    for model in models
        # model = models[1]
        if !isa(model, Function) && sum(valid_models .== model) > 0
            throw(
                ArgumentError(
                    "The supplied model: `" *
                    string(model) *
                    "` is not a valid genomic prediction model function. Please choose from:\n\t‣ " *
                    join(string.(valid_models), "\n\t‣ "),
                ),
            )
        end
    end
    # Apply mask
    n, p = size(genomes.allele_frequencies) # dimensions prior to potential filtering via masks below
    mask_loci_alleles = findall(mean(genomes.mask, dims = 1)[1, :] .== 1.0)
    if (mask_genomes != collect(1:n)) || (mask_loci_alleles != collect(1:p))
        genomes = filter(genomes)
        phenomes = filter(phenomes)
    end
    # Instantiate the entire replicated cross-validation vector of output
    if verbose
        println("Cross-validation per pair of populations")
    end
    n, p = size(genomes.allele_frequencies)
    populations = sort(unique(genomes.populations))
    n_populations = length(populations)
    cvs::Vector{CV} = []
    notes::Vector{String} = []
    models_vector::Vector{Function} = [] # for ease of function calling in multi-threaded function below, i.e. `cvmultithread!(...)`
    if verbose
        k = length(phenomes.traits) * (n_populations^2 - n_populations) * length(models)
        pb = Progress(k; desc = "Setting up pairwise population cross-validation job/s: ")
    end
    for trait in phenomes.traits
        # trait = phenomes.traits[1]
        for training_population in populations
            # training_population = populations[1]
            for validation_population in populations
                # validation_population = populations[2]
                # Skip within population cross-validation
                if training_population == validation_population
                    continue
                end
                # Check the phenotypes early before we partition them into threads. Much more efficient eh!
                ϕ = phenomes.phenotypes[:, phenomes.traits.==trait][:, 1]
                idx_training = findall(
                    (phenomes.populations .== training_population) .&& .!ismissing.(ϕ) .&& .!isnan.(ϕ) .&& .!isinf.(ϕ),
                )
                idx_validation = findall(
                    (phenomes.populations .== validation_population) .&&
                    .!ismissing.(ϕ) .&&
                    .!isnan.(ϕ) .&&
                    .!isinf.(ϕ),
                )
                if (length(idx_training) < 2 || length(idx_validation) < 1)
                    push!(
                        notes,
                        join(
                            [
                                "too_many_missing",
                                trait,
                                string("training: ", training_population),
                                string("validation: ", validation_population),
                            ],
                            ";",
                        ),
                    )
                    continue
                end
                if var(ϕ[idx_training]) < 1e-20
                    push!(
                        notes,
                        join(
                            [
                                "zero_variance",
                                trait,
                                string("training: ", training_population),
                                string("validation: ", validation_population),
                            ],
                            ";",
                        ),
                    )
                    continue
                end
                for model in models
                    # model = models[1]
                    fit = Fit(n = length(idx_training), l = p + 1)
                    fit.model = string(model)
                    fit.trait = trait
                    fit.entries = genomes.entries[idx_training]
                    fit.populations = genomes.populations[idx_training]
                    fit.b_hat_labels = vcat(["intercept"], genomes.loci_alleles)
                    cv = CV(
                        "",
                        "",
                        fit,
                        genomes.populations[idx_validation],
                        genomes.entries[idx_validation],
                        zeros(length(idx_validation)),
                        zeros(length(idx_validation)),
                        fit.metrics,
                    )
                    push!(cvs, cv)
                    push!(models_vector, model)
                    if verbose
                        next!(pb)
                    end
                end
            end
        end
    end
    if verbose
        finish!(pb)
    end
    # Multi-threaded cross-validation
    if verbose
        println("Setup ", length(cvs), " pairwise population cross-validation job/s.")
        println("Skipping ", length(notes), " pairwise population cross-validation job/s.")
    end
    try
        cvmultithread!(cvs, genomes = genomes, phenomes = phenomes, models_vector = models_vector, verbose = verbose)
    catch
        throw(
            ErrorException(
                "Error performing pairwise population cross-validation across population/s: " *
                join(sort(unique(phenomes.populations))) *
                ".",
            ),
        )
    end
    # output
    (cvs, notes)
end

"""
    cvleaveonepopulationout(;
        genomes::Genomes,
        phenomes::Phenomes,
        models::Vector{Function}=[ridge],
        n_replications::Int64=5,
        n_folds::Int64=5,
        seed::Int64=42,
        verbose::Bool=true
    )::Tuple{Vector{CV}, Vector{String}}

Performs leave-one-population-out cross-validation for genomic prediction models across all available traits.

# Arguments
- `genomes::Genomes`: Genomic data structure containing allele frequencies
- `phenomes::Phenomes`: Phenotypic data structure containing trait measurements
- `models::Vector{Function}`: Vector of model functions to evaluate (default: [ridge])
- `n_replications::Int64`: Number of replications (not used in this implementation)
- `n_folds::Int64`: Number of folds (not used in this implementation)
- `seed::Int64`: Random seed (not used in this implementation)
- `verbose::Bool`: If true, displays progress information during execution

# Returns
- `Tuple{Vector{CV}, Vector{String}}`: Returns a tuple containing:
  - Vector of CV objects with cross-validation results
  - Vector of warning/error messages for skipped validations

# Details
The function implements a leave-one-population-out cross-validation strategy where:
1. For each trait and population combination:
   - Uses one population as validation set
   - Uses remaining populations as training set
2. Evaluates multiple genomic prediction models
3. Handles missing data and variance checks
4. Supports parallel processing via Julia's multi-threading

# Threading
To utilize multiple threads, start Julia with: `julia --threads n,1` where n is the desired number of threads
for computation and 1 is reserved for the runtime.

# Supported Models
- ols (Ordinary Least Squares)
- ridge (Ridge Regression)
- lasso (Lasso Regression)
- bayesa (Bayes A)
- bayesb (Bayes B)
- bayesc (Bayes C)

# Example
```jldoctest; setup = :(using GenomicBreedingCore, GenomicBreedingModels, StatsBase)
julia> genomes = GenomicBreedingCore.simulategenomes(verbose=false); genomes.populations = StatsBase.sample(string.("pop_", 1:3), length(genomes.entries), replace=true);

julia> trials, _ = GenomicBreedingCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.1 0.01 0.01;], verbose=false);

julia> phenomes = extractphenomes(trials);

julia> cvs, notes = cvleaveonepopulationout(genomes=genomes, phenomes=phenomes, models=[ridge, bayesa], n_replications=2, n_folds=2, verbose=false);

julia> df_across_entries, df_per_entry = tabularise(cvs);

julia> sum([sum(split(df_across_entries.training_population[i], ";") .== df_across_entries.validation_population[i]) for i in 1:size(df_across_entries, 1)]) == 0
true

julia> idx_across = findall((df_across_entries.validation_population .== "pop_1") .&& (df_across_entries.trait .== "trait_1") .&& (df_across_entries.model .== "ridge"));

julia> idx_per = findall((df_per_entry.validation_population .== "pop_1") .&& (df_per_entry.trait .== "trait_1") .&& (df_per_entry.model .== "ridge"));

julia> abs(df_across_entries.cor[idx_across][1] - cor(df_per_entry.y_true[idx_per], df_per_entry.y_pred[idx_per])) < 1e-10
true
```
"""
function cvleaveonepopulationout(;
    genomes::Genomes,
    phenomes::Phenomes,
    models = [ridge], ### Puting type annotation of Vector{Function} fails when the vector contains a single function
    n_replications::Int64 = 5, ### Unused as no replication is required because we train and validate on the entire respective populations
    n_folds::Int64 = 5, ### Unused as no replication is required because we train and validate on the entire respective populations
    seed::Int64 = 42, ### Unused as no replication is required because we train and validate on the entire respective populations
    verbose::Bool = true,
)::Tuple{Vector{CV},Vector{String}}
    # genomes = GenomicBreedingCore.simulategenomes(n=300, verbose=false); genomes.populations = StatsBase.sample(string.("pop_", 1:3), length(genomes.entries), replace=true);
    # trials, _ = GenomicBreedingCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, verbose=false);
    # phenomes = extractphenomes(trials)
    # models = [ridge, lasso]
    # n_folds = 2; n_replications = 2; seed = 42; verbose = true
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
    mask_genomes = findall(mean(genomes.mask, dims = 2)[:, 1] .== 1.0)
    mask_phenomes = findall(mean(phenomes.mask, dims = 2)[:, 1] .== 1.0)
    if mask_genomes != mask_phenomes
        throw(ArgumentError("The masks in genomes and phenomes do not match."))
    end
    if length(models) < 1
        throw(ArgumentError("No models were specified."))
    end
    valid_models = [ols, ridge, lasso, bayesa, bayesb, bayesc]
    for model in models
        # model = models[1]
        if !isa(model, Function) && sum(valid_models .== model) > 0
            throw(
                ArgumentError(
                    "The supplied model: `" *
                    string(model) *
                    "` is not a valid genomic prediction model function. Please choose from:\n\t‣ " *
                    join(string.(valid_models), "\n\t‣ "),
                ),
            )
        end
    end
    # Apply mask
    n, p = size(genomes.allele_frequencies) # dimensions prior to potential filtering via masks below
    mask_loci_alleles = findall(mean(genomes.mask, dims = 1)[1, :] .== 1.0)
    if (mask_genomes != collect(1:n)) || (mask_loci_alleles != collect(1:p))
        genomes = filter(genomes)
        phenomes = filter(phenomes)
    end
    # Instantiate the entire replicated cross-validation vector of output
    if verbose
        println("Leave-one-population-out cross-validation")
    end
    n, p = size(genomes.allele_frequencies)
    populations = sort(unique(genomes.populations))
    n_populations = length(populations)
    cvs::Vector{CV} = []
    notes::Vector{String} = []
    models_vector::Vector{Function} = [] # for ease of function calling in multi-threaded function below, i.e. `cvmultithread!(...)`
    if verbose
        k = length(phenomes.traits) * (n_populations^2 - n_populations) * length(models)
        pb = Progress(k; desc = "Setting up leave-one-population-out cross-validation job/s: ")
    end
    for trait in phenomes.traits
        # trait = phenomes.traits[1]
        for validation_population in populations
            # validation_population = populations[1]
            # Check the phenotypes early before we partition them into threads. Much more efficient eh!
            ϕ = phenomes.phenotypes[:, phenomes.traits.==trait][:, 1]
            idx_training = findall(
                phenomes.populations .!= validation_population .&& .!ismissing.(ϕ) .&& .!isnan.(ϕ) .&& .!isinf.(ϕ),
            )
            idx_validation = findall(
                phenomes.populations .== validation_population .&& .!ismissing.(ϕ) .&& .!isnan.(ϕ) .&& .!isinf.(ϕ),
            )
            training_population = join(populations[populations.!=validation_population], ";")
            if (length(idx_training) < 2 || length(idx_validation) < 1)
                push!(
                    notes,
                    join(
                        [
                            "too_many_missing",
                            trait,
                            string("training: ", join(sort(unique(training_population)), ";")),
                            string("validation: ", validation_population),
                        ],
                        ";",
                    ),
                )
                continue
            end
            if var(ϕ[idx_training]) < 1e-20
                push!(
                    notes,
                    join(
                        [
                            "zero_variance",
                            trait,
                            string("training: ", join(sort(unique(training_population)), ";")),
                            string("validation: ", validation_population),
                        ],
                        ";",
                    ),
                )
                continue
            end
            for model in models
                # model = models[1]
                fit = Fit(n = length(idx_training), l = p + 1)
                fit.model = string(model)
                fit.trait = trait
                fit.entries = genomes.entries[idx_training]
                fit.populations = genomes.populations[idx_training]
                fit.b_hat_labels = vcat(["intercept"], genomes.loci_alleles)
                cv = CV(
                    "",
                    "",
                    fit,
                    genomes.populations[idx_validation],
                    genomes.entries[idx_validation],
                    zeros(length(idx_validation)),
                    zeros(length(idx_validation)),
                    fit.metrics,
                )
                push!(cvs, cv)
                push!(models_vector, model)
                if verbose
                    next!(pb)
                end
            end
        end
    end
    if verbose
        finish!(pb)
    end
    # Multi-threaded cross-validation
    if verbose
        println("Setup ", length(cvs), " leave-out-population-out cross-validation job/s.")
        println("Skipping ", length(notes), " leave-out-population-out cross-validation job/s.")
    end
    try
        cvmultithread!(cvs, genomes = genomes, phenomes = phenomes, models_vector = models_vector, verbose = verbose)
    catch
        throw(
            ErrorException(
                "Error performing leave-out-population-out cross-validation across population/s: " *
                join(sort(unique(phenomes.populations))) *
                ".",
            ),
        )
    end
    # output
    (cvs, notes)
end
