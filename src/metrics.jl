function metrics(y_obs::Vector{Float64}, y_pred::Vector{Float64})::Dict{String, Float64}
    # y_obs::Vector{Float64} = rand(100); y_pred::Vector{Float64} = rand(100)
    # UnicodePlots.scatterplot(y_obs, y_pred)
    Dict(
        "cor" => 1.00 - Distances.corr_dist(y_obs, y_pred),
        "mad" => Distances.meanad(y_obs, y_pred),
        "msd" => Distances.msd(y_obs, y_pred),
        "rmsd" => Distances.rmsd(y_obs, y_pred),
        "nrmsd" => Distances.nrmsd(y_obs, y_pred),
        "euc" => Distances.euclidean(y_obs, y_pred),
        "jac" => Distances.jaccard(y_obs, y_pred),
        "tvar" => Distances.totalvariation(y_obs, y_pred),
    )
end