function heritability_narrow_sense(y_obs::Vector{Float64}, y_pred::Vector{Float64})::Float64
    # y_obs::Vector{Float64} = rand(100); y_pred::Vector{Float64} = rand(100)
    s²a = var(y_pred)
    s²e = var(y_obs - y_pred)
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

function metrics(y_obs::Vector{Float64}, y_pred::Vector{Float64})::Dict{String,Float64}
    Dict(
        "cor" => 1.00 - Distances.corr_dist(y_obs, y_pred),
        "mad" => Distances.meanad(y_obs, y_pred),
        "msd" => Distances.msd(y_obs, y_pred),
        "rmsd" => Distances.rmsd(y_obs, y_pred),
        "nrmsd" => Distances.nrmsd(y_obs, y_pred),
        "euc" => Distances.euclidean(y_obs, y_pred),
        "jac" => Distances.jaccard(y_obs, y_pred),
        "tvar" => Distances.totalvariation(y_obs, y_pred),
        "h²" => heritability_narrow_sense(y_obs, y_pred),
    )
end
