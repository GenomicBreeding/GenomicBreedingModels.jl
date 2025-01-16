# """
#     crossvalidate(;
#         genomes::Genomes,
#         phenomes::Phenomes,
#         models::Vector{Function}=[ridge],
#         n_folds::Int64=5,
#         n_replications::Int64=5,
#         n_threads::Int64=1,
#         seed::Int64=42,
#         verbose::Bool=true
#     )::Tuple{Vector{CV}, Vector{String}}

# Replicated cross-validation of genomic prediction model/s across all available traits. 

# Note that to use multiple threads, please invoke Julia as: `julia --threads 7,1 --load test/interactive_prelude.jl`,
# where `--threads 7,1` means use 7 threads for multi-threaded processes while reserving 1 thread for the Julia runtime itself.

# ## Examples
# ```jldoctest; setup = :(using GBCore, GBModels; StatsBase)
# julia> genomes = GBCore.simulategenomes(n=300, verbose=false); genomes.populations = StatsBase.sample(string.("pop_", 1:3), length(genomes.entries), replace=true);

# julia> trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, verbose=false);

# julia> phenomes = extractphenomes(trials);

# julia> cv = crossvalidate(genomes=genomes, phenomes=phenomes, models=[ols, ridge], verbose=false);

# julia> summarise(cv)
# ```
# """
# function cvbulk(;
#     genomes::Genomes,
#     phenomes::Phenomes,
#     models::Vector{Function}=[ridge],
#     n_folds::Int64=5,
#     n_replications::Int64=5,
#     seed::Int64=42,
#     verbose::Bool=true
# )::Tuple{Vector{CV}, Vector{String}}
#     # genomes = GBCore.simulategenomes(n=300, verbose=false); genomes.populations = StatsBase.sample(string.("pop_", 1:3), length(genomes.entries), replace=true);
#     # trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, verbose=false);
#     # phenomes = extractphenomes(trials)
#     # models = [ridge, lasso]
#     # n_folds = 2; n_replications = 2; seed = 42; verbose = true
#     # Check arguments
#     if !checkdims(genomes) && !checkdims(phenomes)
#         throw(ArgumentError("The Genomes and Phenomes structs are corrupted."))
#     end
#     if !checkdims(genomes)
#         throw(ArgumentError("The Genomes struct is corrupted."))
#     end
#     if !checkdims(phenomes)
#         throw(ArgumentError("The Phenomes struct is corrupted."))
#     end
#     if genomes.entries != phenomes.entries
#         throw(ArgumentError("The genomes and phenomes input need to have been merged to have consitent entries."))
#     end
#     mask_genomes = findall(mean(genomes.mask, dims = 2)[:, 1] .== 1.0)
#     mask_phenomes = findall(mean(phenomes.mask, dims = 2)[:, 1] .== 1.0)
#     if mask_genomes != mask_phenomes
#         throw(ArgumentError("The masks in genomes and phenomes do not match."))
#     end
#     valid_models = [
#         ols,
#         ridge,
#         lasso,
#         bayesa,
#         bayesb,
#         bayesc,
#     ]
#     for model in models
#         # model = models[1]
#         if !isa(model, Function)
#             throw(ArgumentError("The supplied model: `" * string(model) * "` is not a valid genomic prediction model function. Please choose from:\n\t‣ " *
#             join(string.(valid_models), "\n\t‣ ")
#             ))
#         end
#     end
#     n, p = size(genomes.allele_frequencies) # dimensions prior to potential filtering via masks below
#     if (n_folds < 1) || (n_folds > n)
#         throw(ArgumentError("The number of folds, `n_folds = " * string(n_folds) * "` is out of bounds. Please use values from 1 to " * string(n) * "."))
#     end
#     if (n_replications < 1) || (n_replications > 100)
#         throw(ArgumentError("The number of replications, `n_replications = " * string(n_replications) * "` is out of bounds. Please use values from 1 to 100."))
#     end
#     # Apply mask
#     mask_loci_alleles = findall(mean(genomes.mask, dims = 1)[1, :] .== 1.0)
#     if (mask_genomes != collect(1:n)) || (mask_loci_alleles != collect(1:p))
#         genomes = filter(genomes)
#         phenomes = filter(phenomes)
#     end
#     # Set randomisation seed
#     rng::TaskLocalRNG = Random.seed!(seed)
#     # Instantiate the entire replicated cross-validation vector of output
#     n, p = size(genomes.allele_frequencies)
#     cvs::Vector{CV} = []
#     notes::Vector{String} = []
#     models_vector::Vector{Function} = [] # for ease of function calling in multi-threaded loop below
#     for trait in phenomes.traits
#         # trait = phenomes.traits[1]
#         for i in 1:n_replications
#             # i = 1
#             idx_permutation = StatsBase.sample(rng, 1:n_folds, n, replace=true)
#             for j in 1:n_folds
#                 # j = 1
#                 # Check the phenotypes early before we partition them into threads. Much more efficient eh!
#                 ϕ = phenomes.phenotypes[:, phenomes.traits .== trait][:, 1]
#                 idx = findall((idx_permutation .== j) .&& (ismissing.(ϕ) .== false))
#                 if length(idx) < 2
#                     push!(notes, join(["too_many_missing", trait, string("replication_", i), string("fold_", j)], ";"))
#                     continue
#                 end
#                 if var(ϕ[idx]) < 1e-20
#                     push!(notes, join(["zero_variance", trait, string("replication_", i), string("fold_", j)], ";"))
#                     continue
#                 end
#                 for model in models
#                     # model = models[1]
#                     fit = Fit(n=length(idx), l=p+1)
#                     fit.trait = trait
#                     fit.entries = genomes.entries[idx]
#                     fit.populations = genomes.populations[idx]
#                     fit.b_hat_labels = vcat(["intercept"], genomes.loci_alleles)
#                     cv = CV(
#                         string(model),
#                         string("replication_", i),
#                         string("fold_", j),
#                         fit,
#                         join(sort(unique(fit.populations)), ";"),
#                         fit.entries,
#                         join(sort(unique(fit.populations)), ";"),
#                         fit.entries,
#                         fit.y_true,
#                         fit.y_pred,
#                         fit.metrics
#                     )
#                     push!(cvs, cv)
#                     # push!(fits, fit)
#                     # push!(replications, string("replication_", i))
#                     # push!(folds, string("fold_", j))
#                     push!(models_vector, model)
#                 end
#             end
#         end
#     end
#     # Cross-validate using all thread/s available to Julia (set at startup via julia --threads 2,1 --load test/interactive_prelude.jl)
    
#     m = length(cvs)
#     if verbose
#         pb = Progress(m; desc = "Genomic prediction replicated cross-validation: ")
#     end
#     thread_lock::ReentrantLock = ReentrantLock()
#     Threads.@threads for i in 1:m
#         # i = 1
#         model = models_vector[i]
#         idx_entries = [findall(genomes.entries .== x)[1] for x in cvs[i].fits.entries]
#         idx_loci_alleles = [findall(genomes.loci_alleles .== x)[1] for x in cvs[i].fits.b_hat_labels[2:end]]
#         idx_trait = findall(phenomes.traits .== cvs[i].fits.trait)[1]
#         try
#             fit = model(
#                 genomes=genomes,
#                 phenomes=phenomes,
#                 idx_entries=idx_entries,
#                 idx_loci_alleles=idx_loci_alleles,
#                 idx_trait=idx_trait
#             )
#             @lock thread_lock cvs[i].fits = fit
#         catch
#             @warn string("Oh naur! This is unexpected model fitting error! Model: ", model, "; i: ", i)
#             continue
#         end
#         if verbose
#             next!(pb)
#         end
#     end
#     if verbose
#         finish!(pb)
#     end
#     # output
#     (cvs, notes)
# end





# # # TODO: Move to GBCore.jl
# # function summarise(cv::CV, metric::String="cor")
# #     # genomes = GBCore.simulategenomes(verbose=false); genomes.populations = StatsBase.sample(string.("pop_", 1:3), length(genomes.entries), replace=true);
# #     # trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, verbose=false);;
# #     # phenomes = extractphenomes(trials)
# #     # cv = crossvalidate(genomes=genomes, phenomes=phenomes, models = [ridge, bayesa], n_replications=2, n_folds=2)
# #     # metric = "cor"
# #     # Check arguments
# #     c = length(cv[i].fits     if c < 1
# #         throw(ArgumentError("Input vector of CV structs is empty."))
# #     end
# #     if sum(names(cv.fits[1].metrics) .== metric) != 1
# #         throw(ArgumentError("Unrecognised metric `" * metric * "`. Please choose from:\n\t‣ " * 
# #         join(names(cv.fits[1].metrics), "\n\t‣ ")
# #         ))
# #     end
# #     # Extract the metrics calculated across entries per trait, replication, fold and model
# #     df_across_entries = DataFrames.DataFrame(
# #         trait = fill("", c),
# #         replication = fill("", c),
# #         fold = fill("", c),
# #         model = fill("", c),
# #         θ = fill(0.0, c),
# #     )
# #     # At the same time extract individual entry predictions
# #     traits::Vector{String} = []
# #     models::Vector{String} = []
# #     entries::Vector{String} = []
# #     populations::Vector{String} = []
# #     y_trues::Vector{Float64} = []
# #     y_preds::Vector{Float64} = []
# #     for i in 1:c
# #         # i = 1
# #         df_across_entries.trait[i] = cv.fits[i].trait
# #         df_across_entries.replication[i] = cv.replications[i]
# #         df_across_entries.fold[i] = cv.folds[i]
# #         df_across_entries.model[i] = cv.fits[i].model
# #         df_across_entries.θ[i] = cv.fits[i].metrics[metric]
# #         n = length(cv.fits[i].entries)
# #         append!(traits, repeat([cv.fits[i].trait], n))
# #         append!(models, repeat([cv.fits[i].model], n))
# #         append!(entries, cv.fits[i].entries)
# #         append!(populations, cv.fits[i].populations)
# #         append!(y_trues, cv.fits[i].y_true)
# #         append!(y_preds, cv.fits[i].y_pred)
# #     end
# #     # Metrics per entry
# #     df_per_entry = DataFrames.DataFrame(
# #         trait=traits,
# #         model=models,
# #         entry=entries,
# #         population=populations,
# #         y_true=y_trues,
# #         y_pred=y_preds,
# #     )


# #     combine(groupby(df_across_entries, :trait), :θ => mean)
# #     combine(groupby(df_across_entries, :model), :θ => mean)
# #     combine(groupby(df_across_entries, [:trait, :model]), :θ => mean)
        
# #     # Using metric calculated per entry
# #     t = length(unique(df_across_entries.trait))
# #     n = length(unique(df_across_entries.entry))
# #     m = length(unique(df_across_entries.model))

# #     for i in 1:c
# #         # i = 1
# #         entries = cv[i].entries

# #         df_across_entries.trait[i] = cv[i].trait
# #         df_across_entries.entry[i] = cv[i].entries
# #         df_across_entries.model[i] = cv[i].fit.model
        
# #     end





# # end