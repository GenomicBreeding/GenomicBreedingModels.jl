try
    R"library(BGLR)"
catch
    throw(ErrorException("Please install the BGLR package in R."))
end

function bglr(;
    G::Matrix{Float64},
    y::Vector{Float64},
    model::String = ["BayesA", "BayesB", "BayesC"][1],
    n_iter::Int64 = 1_500,
    n_burnin::Int64 = 500,
    verbose::Bool = false,
)::Vector{Float64}
    # genomes = GBCore.simulategenomes(n=1_000, l=1_750, verbose=true)
    # trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.1 0.01 0.01;], verbose=true)
    # phenomes = extractphenomes(trials)
    # G::Matrix{Float64} = genomes.allele_frequencies
    # y::Vector{Float64} = phenomes.phenotypes[:, 1]
    # model::String=["BayesA", "BayesB", "BayesC"][1]; n_iter::Int64=1_500; n_burnin::Int64=500; verbose::Bool=true
    @rput(G)
    @rput(y)
    @rput(model)
    @rput(n_iter)
    @rput(n_burnin)
    @rput(verbose)
    R"ETA = list(MRK=list(X=G, model=model, saveEffects=FALSE))"
    @time R"sol = BGLR::BGLR(y=y, ETA=ETA, nIter=n_iter, burnIn=n_burnin, verbose=verbose)"
    @rget(sol)
    sol[:ETA][:MRK][:b]
end
