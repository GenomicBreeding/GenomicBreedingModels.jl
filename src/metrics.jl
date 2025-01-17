# NOTE: We set Pearson's correlation and narrow-sense heritability metrics to zero if either/both observed and predicted values have nil/zero variance

function pearsonscorrelation(y_true::Vector{Float64}, y_pred::Vector{Float64})::Float64
    # y_true::Vector{Float64} = rand(100); y_pred::Vector{Float64} = rand(100)
    if (var(y_true) < 1e-10) || (var(y_pred) < 1e-10)
        return 0.0
    end
    1.00 - Distances.corr_dist(y_true, y_pred)
end

function heritabilitynarrow_sense(y_true::Vector{Float64}, y_pred::Vector{Float64})::Float64
    # y_true::Vector{Float64} = rand(100); y_pred::Vector{Float64} = rand(100)
    if (var(y_true) < 1e-10) || (var(y_pred) < 1e-10)
        return 0.0
    end
    s²a = var(y_pred)
    s²e = var(y_true - y_pred)
    h² = 0.0
    if (s²a + s²e) >= 1e-20
        h² = s²a / (s²a + s²e)
    end
    if h² < 0.0
        h² = 0.0
    elseif h² > 1.0
        h² = 1.0
    end
    h²
end

function metrics(y_true::Vector{Float64}, y_pred::Vector{Float64})::Dict{String,Float64}
    Dict(
        "cor" => pearsonscorrelation(y_true, y_pred),
        "mad" => Distances.meanad(y_true, y_pred),
        "msd" => Distances.msd(y_true, y_pred),
        "rmsd" => Distances.rmsd(y_true, y_pred),
        "nrmsd" => Distances.nrmsd(y_true, y_pred),
        "euc" => Distances.euclidean(y_true, y_pred),
        "jac" => Distances.jaccard(y_true, y_pred),
        "tvar" => Distances.totalvariation(y_true, y_pred),
        "h²" => heritabilitynarrow_sense(y_true, y_pred),
    )
end
