function gwasprep(
    genomes::Genomes,
    phenomes::Phenomes;
    idx_entries::Union{Nothing,Vector{Int64}} = nothing,
    idx_loci_alleles::Union{Nothing,Vector{Int64}} = nothing,
    idx_trait::Int64 = 1,
    GRM_type::String = ["simple", "ploidy-aware"][1],
    standardise::Bool = true,
    verbose::Bool = false,
)::Tuple{Matrix{Float64},Vector{Float64},Matrix{Float64},Fit}
    # genomes = GBCore.simulategenomes(); ploidy = 4; genomes.allele_frequencies = round.(genomes.allele_frequencies .* ploidy) ./ ploidy
    # proportion_of_variance = zeros(9, 1); proportion_of_variance[1, 1] = 0.5
    # trials, effects = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.05 0.00 0.00;], proportion_of_variance = proportion_of_variance, verbose=false);
    # phenomes = extractphenomes(trials)
    # idx_entries = nothing; idx_loci_alleles = nothing; idx_trait = 1; GRM_type = "ploidy-aware"; verbose = true
    # Check arguments while extracting the allele frequencies
    G, y, entries, populations, loci_alleles = extractxyetc(
        genomes,
        phenomes,
        idx_entries = idx_entries,
        idx_loci_alleles = idx_loci_alleles,
        idx_trait = idx_trait,
        add_intercept = false,
    )
    if sum(["simple", "ploidy-aware"] .== GRM_type) == 0
        throw(
            ArgumentError(
                "Unrecognised `GRM_type`. Please select from:\n\t‣ " * join(["simple", "ploidy-aware"], "\n\t‣ "),
            ),
        )
    end
    # Standardise G and y and remove fixed loci-alleles
    if standardise
        y = (y .- mean(y)) ./ std(y)
    end
    if var(y) < eps(Float64)
        throw(ArgumentError("No variance in the trait: " * phenomes.traits[idx_trait] * "."))
    end
    v = std(G, dims = 1)[1, :]
    idx_cols = findall((v .> eps(Float64)) .&& .!ismissing.(v) .&& .!isnan.(v) .&& .!isinf.(v))
    G = G[:, idx_cols]
    loci_alleles = loci_alleles[idx_cols]
    if standardise
        G = (G .- mean(G, dims = 1)) ./ v[idx_cols]'
    end
    # Extract the GRM to correct for population structure
    if GRM_type == "ploidy-aware"
        # Infer ploidy level
        ploidy = Int(round(1 / minimum(G[G.!=0.0])))
        GRM = grmploidyaware(genomes, ploidy = ploidy)
    else
        # Simple GRM
        GMR = grmsimple(genomes)
    end
    if standardise
        GRM = (GRM .- mean(GRM, dims = 1)) ./ std(GRM, dims = 1)
    end
    # Instantiate output Fit struct
    n, l = size(G)
    fit = Fit(n = n, l = l)
    fit.model = "GWAS_OLS"
    fit.trait = phenomes.traits[idx_trait]
    fit.b_hat_labels = loci_alleles
    fit.entries = entries
    fit.populations = populations
    fit.metrics = Dict("" => 0.0)
    (G, y, GRM, fit)
end

function gwasols(
    genomes::Genomes,
    phenomes::Phenomes;
    idx_entries::Union{Nothing,Vector{Int64}} = nothing,
    idx_loci_alleles::Union{Nothing,Vector{Int64}} = nothing,
    idx_trait::Int64 = 1,
    GRM_type::String = ["simple", "ploidy-aware"][1],
    verbose::Bool = false,
)::Fit
    # genomes = GBCore.simulategenomes(); ploidy = 4; genomes.allele_frequencies = round.(genomes.allele_frequencies .* ploidy) ./ ploidy
    # proportion_of_variance = zeros(9, 1); proportion_of_variance[1, 1] = 0.5
    # trials, effects = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.05 0.00 0.00;], proportion_of_variance = proportion_of_variance, verbose=false);
    # phenomes = extractphenomes(trials)
    # idx_entries = nothing; idx_loci_alleles = nothing; idx_trait = 1; GRM_type = "ploidy-aware"; verbose = true
    # Check arguments while preparing the G, y, GRM, and Fit struct, vector and matrices
    G, y, GRM, fit = gwasprep(
        genomes,
        phenomes,
        idx_entries = idx_entries,
        idx_loci_alleles = idx_loci_alleles,
        idx_trait = idx_trait,
        GRM_type = GRM_type,
        standardise = true,
        verbose = false,
    )
    # Iterative GWAS
    X = hcat(ones(n), GRM, G[:, 1])
    if verbose
        pb = ProgressMeter.Progress(l; desc = "GWAS via OLS using " * GRM_type * " GRM:")
    end
    for j = 1:l
        # j = 1
        X[:, end] = G[:, j]
        Vinv = pinv(X' * X)
        b = Vinv * X' * y
        fit.b_hat[j] = b[end] / sqrt(Vinv[end, end])
        if verbose
            ProgressMeter.next!(pb)
        end
    end
    if verbose
        ProgressMeter.finish!(pb)
        # Histogram of t-values
        UnicodePlots.histogram(fit.b_hat, title = "Distribution of t-values")
        # Manhattan plot
        tdist = Distributions.TDist(n - 1)
        pval = ccdf.(tdist, abs.(fit.b_hat))
        lod = -log10.(pval)
        threshold = -log10(0.05 / l)
        p1 = UnicodePlots.scatterplot(lod, title = "Manhattan plot", xlabel = "Loci-alleles", ylabel = "-log10(pval)")
        UnicodePlots.lineplot!(p1, [0, l], [threshold, threshold])
        @show p1
        # QQ plot
        lod_expected = reverse(-log10.(collect(range(0, 1, l))))
        p2 = UnicodePlots.scatterplot(sort(lod), lod_expected, xlabel = "Observed LOD", ylabel = "Expected LOD")
        UnicodePlots.lineplot!(p2, [0, lod_expected[end-1]], [0, lod_expected[end-1]])
        @show p2
    end
    # Output
    if !checkdims(fit)
        throw(ErrorException("Error performing GWAS via OLS using the " * GRM_type * " GRM."))
    end
    fit
end

function gwasreml(
    genomes::Genomes,
    phenomes::Phenomes;
    idx_entries::Union{Nothing,Vector{Int64}} = nothing,
    idx_loci_alleles::Union{Nothing,Vector{Int64}} = nothing,
    idx_trait::Int64 = 1,
    GRM_type::String = ["simple", "ploidy-aware"][1],
    verbose::Bool = false,
)::Fit
    # genomes = GBCore.simulategenomes(); ploidy = 4; genomes.allele_frequencies = round.(genomes.allele_frequencies .* ploidy) ./ ploidy
    # proportion_of_variance = zeros(9, 1); proportion_of_variance[1, 1] = 0.5
    # trials, effects = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.05 0.00 0.00;], proportion_of_variance = proportion_of_variance, verbose=false);
    # phenomes = extractphenomes(trials)
    # idx_entries = nothing; idx_loci_alleles = nothing; idx_trait = 1; GRM_type = "ploidy-aware"; verbose = true
    # Check arguments while preparing the G, y, GRM, and Fit struct, vector and matrices
    G, y, GRM, fit = gwasprep(
        genomes,
        phenomes,
        idx_entries = idx_entries,
        idx_loci_alleles = idx_loci_alleles,
        idx_trait = idx_trait,
        GRM_type = GRM_type,
        standardise = true,
        verbose = false,
    )
    # Iterative GWAS
    df = DataFrames.DataFrame(x = G[:, 1])
    df.y = (GRM * y)
    if abs(mean(df.y)) > 0.0001
        throw(ErrorException("E[Ky] ≠ 0"))
    end
    formula = string("y ~ 1 + (1|x)")
    if verbose
        pb = ProgressMeter.Progress(l; desc = "GWAS via OLS using " * GRM_type * " GRM:")
    end
    for j = 1:l
        # j = 1
        df.x = G[:, j]
        f = @eval(@string2formula $(formula))
        model = try
            MixedModel(f, df)
        catch
            continue
        end
        model.optsum.REML = true
        model.optsum.maxtime = 60
        try
            fit!(model, progress = false)
        catch
            try
                fit!(model, progress = false)
            catch
                continue
            end
        end
        unique(df.x)
        DataFrame(raneftables(model)[1])
        condVar(model)

        if verbose
            ProgressMeter.next!(pb)
        end
    end
    if verbose
        ProgressMeter.finish!(pb)
        # Histogram of t-values
        UnicodePlots.histogram(out.b_hat, title = "Distribution of t-values")
        # Manhattan plot
        tdist = Distributions.TDist(n - 1)
        pval = ccdf.(tdist, abs.(out.b_hat))
        lod = -log10.(pval)
        threshold = -log10(0.05 / l)
        p1 = UnicodePlots.scatterplot(lod, title = "Manhattan plot", xlabel = "Loci-alleles", ylabel = "-log10(pval)")
        UnicodePlots.lineplot!(p1, [0, l], [threshold, threshold])
        @show p1
        # QQ plot
        lod_expected = reverse(-log10.(collect(range(0, 1, l))))
        p2 = UnicodePlots.scatterplot(sort(lod), lod_expected, xlabel = "Observed LOD", ylabel = "Expected LOD")
        UnicodePlots.lineplot!(p2, [0, lod_expected[end-1]], [0, lod_expected[end-1]])
        @show p2
    end
    # Output
    if !checkdims(out)
        throw(ErrorException("Error performing GWAS via OLS using the " * GRM_type * " GRM."))
    end
    out
end
