"""
    pearsonscorrelation(y_true::Vector{Float64}, y_pred::Vector{Float64})::Float64

Calculate the Pearson correlation coefficient between two vectors.

The Pearson correlation coefficient measures the linear correlation between two variables,
giving a value between -1 and +1, where:
- +1 represents perfect positive correlation
- 0 represents no linear correlation
- -1 represents perfect negative correlation

# Arguments
- `y_true::Vector{Float64}`: Vector of true/actual values
- `y_pred::Vector{Float64}`: Vector of predicted/estimated values

# Returns
- `Float64`: Pearson correlation coefficient

# Notes
- Returns 0.0 if the variance of either input vector is less than 1e-10
- Uses `1 - correlation distance` from Distances.jl package
"""
function pearsonscorrelation(y_true::Vector{Float64}, y_pred::Vector{Float64})::Float64
    # y_true::Vector{Float64} = rand(100); y_pred::Vector{Float64} = rand(100)
    if (var(y_true) < 1e-10) || (var(y_pred) < 1e-10)
        return 0.0
    end
    1.00 - Distances.corr_dist(y_true, y_pred)
end


"""
    heritabilitynarrow_sense(y_true::Vector{Float64}, y_pred::Vector{Float64})::Float64

Calculate narrow-sense heritability (h²) from true and predicted phenotypic values.

Narrow-sense heritability is the proportion of phenotypic variance that can be attributed to additive genetic effects.
It is calculated as the ratio of additive genetic variance (s²a) to total phenotypic variance (s²a + s²e).

# Arguments
- `y_true::Vector{Float64}`: Vector of observed/true phenotypic values
- `y_pred::Vector{Float64}`: Vector of predicted genetic values

# Returns
- `Float64`: Narrow-sense heritability (h²) value between 0 and 1

# Details
- Returns 0.0 if variance of either input vector is near zero (< 1e-10)
- Additive genetic variance (s²a) is estimated from variance of predictions
- Environmental variance (s²e) is estimated from variance of residuals
- Result is bounded between 0 and 1
"""
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

"""
    metrics(y_true::Vector{Float64}, y_pred::Vector{Float64})::Dict{String,Float64}

Calculate various metrics comparing predicted vs true values.

Returns a dictionary containing the following metrics:
- `cor`: Pearson correlation coefficient
- `mad`: Mean absolute deviation
- `msd`: Mean squared deviation
- `rmsd`: Root mean squared deviation  
- `nrmsd`: Normalized root mean squared deviation
- `euc`: Euclidean distance
- `jac`: Jaccard distance
- `tvar`: Total variation distance
- `h²`: Narrow-sense heritability

# Arguments
- `y_true::Vector{Float64}`: Vector of true/observed values
- `y_pred::Vector{Float64}`: Vector of predicted values

# Returns
- `Dict{String,Float64}`: Dictionary mapping metric names to their calculated values
"""
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
