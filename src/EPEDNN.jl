module EPEDNN

import Flux
import Dates
import Memoize
import BSON
import JSON

#= ===================================== =#
#  structs/constructors for the EPEDmodel
#= ===================================== =#
# EPEDmodel abstract type, since we could have different models
abstract type EPEDmodel end

# EPED1NN
struct EPED1NNmodel <: EPEDmodel
    fluxmodel::Flux.Chain
    name::String
    date::Dates.DateTime
    xnames::Vector{String}
    ynames::Vector{String}
    xm::Vector{Float64}
    xσ::Vector{Float64}
    ym::Vector{Float64}
    yσ::Vector{Float64}
    xbounds::Array{Float64}
    ybounds::Array{Float64}
    yp::Array{Float64}
end

#= ========================================== =#
#  functions for saving/loading the EPEDmodel
#= ========================================== =#
function savemodel(model::EPEDmodel, filename::String)
    savedict = Dict()
    for name in fieldnames(EPED1NNmodel)
        if name == :fluxmodel
            savedict[:fluxstate] = Flux.state(model.fluxmodel)
        else
            savedict[name] = getproperty(model, name)
        end
    end
    fullpath = dirname(dirname(@__FILE__)) * "/data/" * filename
    BSON.bson(fullpath, savedict)
    return fullpath
end

Memoize.@memoize function loadmodelonce(filename::String)
    return loadmodel(filename)
end

function loadmodel(filename::String)
    savedict = BSON.load(dirname(dirname(@__FILE__)) * "/data/" * filename, @__MODULE__)
    args = []
    for name in fieldnames(EPED1NNmodel)
        if name == :fluxmodel
            fluxmodel = Flux.Chain(
                Flux.Dense(10 => 32, Flux.gelu),
                Flux.Dense(32 => 32, Flux.gelu),
                Flux.Dense(32 => 32, Flux.gelu),
                Flux.Dense(32 => 18)
            ) |> Flux.f64 # use a 64bit model
            Flux.loadmodel!(fluxmodel, savedict[:fluxstate])
            push!(args, fluxmodel)
        else
            push!(args, savedict[name])
        end
    end
    return EPED1NNmodel(args...)
end

#= ====================================== =#
#  functions to get the pedestal solution
#= ====================================== =#
function pedestal_array(pedmodel::EPED1NNmodel, x::AbstractMatrix{<:Real}; only_powerlaw::Bool=false, warn_nn_train_bounds::Bool=true)
    return hcat(collect(map(x0 -> pedestal_array(pedmodel, x0; only_powerlaw, warn_nn_train_bounds), eachslice(x; dims=2)))...)
end

function pedestal_array(pedmodel::EPED1NNmodel, x::AbstractVector{<:Real}; only_powerlaw::Bool=false, warn_nn_train_bounds::Bool=true)
    xx = deepcopy(x)

    if warn_nn_train_bounds # training bounds are on the original data
        for ix in eachindex(xx)
            if any(xx[ix] .< pedmodel.xbounds[ix, 1])
                @warn("Extrapolation warning on $(pedmodel.xnames[ix])=$(minimum(xx[ix])) is below bound of $(pedmodel.xbounds[ix,1])")
            elseif any(xx[ix] .> pedmodel.xbounds[ix, 2])
                @warn("Extrapolation warning on $(pedmodel.xnames[ix])=$(maximum(xx[ix])) is above bound of $(pedmodel.xbounds[ix,2])")
            end
        end
    end

    xx[4] += 1.0 # delta + 1
    xx .= abs.(xx) # to make Bt and Ip always positive
    y0 = power_law_fit_eval(pedmodel.yp, xx)
    if !only_powerlaw
        xn = (xx .- pedmodel.xm) ./ pedmodel.xσ
        yn = pedmodel.fluxmodel(xn)
        y1 = yn .* pedmodel.yσ .+ pedmodel.ym
        y = y0 .+ y1
    else
        y = y0
    end
    y .^= 2 # quare of the outputs
    y[1:9] .*= [xx[8] for k in 1:9] # multiply by density
    return y
end

function pedestal_array(
    pedmodel::EPED1NNmodel,
    a::T,
    betan::T,
    bt::T,
    delta::T,
    ip::T,
    kappa::T,
    m::T,
    neped::T,
    r::T,
    zeffped::T;
    only_powerlaw::Bool=false,
    warn_nn_train_bounds::Bool=true
) where {T<:Real}
    x = [a, betan, bt, delta, ip, kappa, m, neped, r, zeffped]
    return pedestal_array(pedmodel, x; only_powerlaw, warn_nn_train_bounds)
end

#= ================================================== =#
#  structs/constructors to interpret PedestalSolution
#= ================================================== =#
struct ModeSolution
    H
    meta
    superH
end

struct DiamagneticSolution
    GH::ModeSolution
    G::ModeSolution
    H::ModeSolution
end

struct PedestalSolution
    pressure::DiamagneticSolution
    width::DiamagneticSolution
end

function Base.Dict(pedsol::PedestalSolution)
    out = Dict()
    for field1 in fieldnames(PedestalSolution)
        out[field1] = Dict()
        for field2 in fieldnames(DiamagneticSolution)
            out[field1][field2] = Dict()
            for field3 in fieldnames(ModeSolution)
                out[field1][field2][field3] = getproperty(getproperty(getproperty(pedsol, field1), field2), field3)
            end
        end
    end
    return out
end

function PedestalSolution(
    pedmodel::EPED1NNmodel,
    a::Real,
    betan::Real,
    bt::Real,
    delta::Real,
    ip::Real,
    kappa::Real,
    m::Real,
    neped::Real,
    r::Real,
    zeffped::Real;
    only_powerlaw::Bool=false,
    warn_nn_train_bounds::Bool=true
)
    a, betan, bt, delta, ip, kappa, m, neped, r, zeffped = promote(a, betan, bt, delta, ip, kappa, m, neped, r, zeffped)
    x = [a, betan, bt, delta, ip, kappa, m, neped, r, zeffped]
    return PedestalSolution(pedmodel, x; only_powerlaw, warn_nn_train_bounds)
end

function PedestalSolution(pedmodel::EPED1NNmodel, x::AbstractVector; only_powerlaw::Bool=false, warn_nn_train_bounds::Bool=true)
    y = pedestal_array(pedmodel, x; only_powerlaw, warn_nn_train_bounds)
    return PedestalSolution(
        # pressure
        DiamagneticSolution(
            ModeSolution(y[1], y[2], y[3]),
            ModeSolution(y[4], y[5], y[6]),
            ModeSolution(y[7], y[8], y[9])
        ),
        # width
        DiamagneticSolution(
            ModeSolution(y[10], y[11], y[12]),
            ModeSolution(y[13], y[14], y[15]),
            ModeSolution(y[16], y[17], y[18])
        )
    )
end

#= ================================= =#
#  functors for EPED1NNmodel objects
#= ================================= =#
function (pedmodel::EPED1NNmodel)(x::Array; only_powerlaw::Bool=false, warn_nn_train_bounds::Bool=true)
    return pedestal_array(pedmodel, x; only_powerlaw, warn_nn_train_bounds)
end

function (pedmodel::EPED1NNmodel)(a, betan, bt, delta, ip, kappa, m, neped, r, zeffped; only_powerlaw::Bool=false, warn_nn_train_bounds::Bool=true)
    return PedestalSolution(pedmodel, a, betan, bt, delta, ip, kappa, m, neped, r, zeffped; only_powerlaw, warn_nn_train_bounds)
end

mutable struct InputEPED{T<:Real}
    a::Union{T,Missing}
    betan::Union{T,Missing}
    bt::Union{T,Missing}
    delta::Union{T,Missing}
    ip::Union{T,Missing}
    kappa::Union{T,Missing}
    m::Union{T,Missing}
    neped::Union{T,Missing}
    r::Union{T,Missing}
    zeffped::Union{T,Missing}

    function InputEPED()
        return InputEPED{Float64}()
    end
    function InputEPED{T}() where {T<:Real}
        return new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    end
end

function Base.show(io::IO, input::InputEPED)
    return print(io,
        "\n" *
        "           a : $(input.a)\n" *
        "       betan : $(input.betan)\n" *
        "          bt : $(input.bt)\n" *
        "       delta : $(input.delta)\n" *
        "          ip : $(input.ip)\n" *
        "       kappa : $(input.kappa)\n" *
        "           m : $(input.m)\n" *
        "       neped : $(input.neped)\n" *
        "           r : $(input.r)\n" *
        "     zeffped : $(input.zeffped)")
end

function (pedmodel::EPED1NNmodel)(input::InputEPED; only_powerlaw::Bool=false, warn_nn_train_bounds::Bool=true)
    return PedestalSolution(
        pedmodel,
        input.a,
        input.betan,
        input.bt,
        input.delta,
        input.ip,
        input.kappa,
        input.m,
        input.neped,
        input.r,
        input.zeffped;
        only_powerlaw,
        warn_nn_train_bounds
    )
end

"""
    run_epednn(input_eped::InputEPED; model_filename::String="EPED1NNmodel.bson", warn_nn_train_bounds::Bool)

Run EPEDNN starting from a InputEPED, using a specific `model_filename`.

The warn_nn_train_bounds checks against the standard deviation of the inputs to warn if evaluation is likely outside of training bounds.

Returns a `PedestalSolution` structure
"""
function run_epednn(input_eped::InputEPED; model_filename::String="EPED1NNmodel.bson", warn_nn_train_bounds::Bool)
    epedmod = EPEDNN.loadmodelonce(model_filename)
    return epedmod(input_eped...; warn_nn_train_bounds)
end

export run_epednn

#= ============= =#
#  power law fit
#= ============= =#
function power_law_fit(A, b, λ=0)
    A = vcat(transpose(b .* 0.0 .+ 1), log10.(abs.(A)))
    b = log10.(abs.(b))
    if λ > 0
        A = transpose(A)
        reg_solve(A, b, λ) = inv(A' * A + λ * I) * A' * b
        p = reg_solve(A, b, λ)
    else
        b = transpose(b)
        p = transpose(b / A)
    end
    return p
end

function power_law_fit_eval(py::AbstractMatrix, x::AbstractMatrix)
    yy = zeros(size(py)[1], size(x)[2])
    for k in 1:size(py)[1]
        yy[k, :] .= power_law_fit_eval(py[k, :], x)[1, :]
    end
    return yy
end

function power_law_fit_eval(py::AbstractVector, x::AbstractMatrix)
    return hcat(collect(map(x0 -> power_law_fit_eval(py, x0), eachslice(x; dims=2)))...)
end

function power_law_fit_eval(py::AbstractMatrix, x0::AbstractVector)
    yy = zeros(eltype(x0), size(py)[1])
    for k in 1:size(py)[1]
        yy[k, :] .= power_law_fit_eval(py[k, :], x0)
    end
    return yy
end

function power_law_fit_eval(p::AbstractVector, x0::AbstractVector)
    y = p[1]
    for i in eachindex(x0)
        y += (p[i+1] * log10(abs(x0[i])))
    end
    return 10.0^y
end

"""
    extrapolation_distance(pedmodel::EPED1NNmodel, x::AbstractVector{<:Real})

Compute a normalized extrapolation distance for each input relative to the training bounds.

Returns a named tuple with:
  - `per_input`: Dict mapping input name to its normalized distance (0 = within bounds)
  - `max_distance`: worst-case distance across all inputs
  - `worst_input`: name of the input furthest outside bounds

Distance is measured as fraction of the training range: `(x - bound) / (bound_max - bound_min)`.
A value of 0.5 means the input is half a training-range-width outside the bounds.
"""
function extrapolation_distance(pedmodel::EPED1NNmodel, x::AbstractVector{<:Real})
    per_input = Dict{String,Float64}()
    max_dist = 0.0
    worst = ""
    for ix in eachindex(x)
        xmin = pedmodel.xbounds[ix, 1]
        xmax = pedmodel.xbounds[ix, 2]
        range_ix = xmax - xmin
        if range_ix <= 0.0
            continue
        end
        dist = max(0.0, (xmin - x[ix]) / range_ix, (x[ix] - xmax) / range_ix)
        per_input[pedmodel.xnames[ix]] = dist
        if dist > max_dist
            max_dist = dist
            worst = pedmodel.xnames[ix]
        end
    end
    return (per_input=per_input, max_distance=max_dist, worst_input=worst)
end

function extrapolation_distance(pedmodel::EPED1NNmodel, input::InputEPED)
    x = [input.a, input.betan, input.bt, input.delta, input.ip, input.kappa, input.m, input.neped, input.r, input.zeffped]
    return extrapolation_distance(pedmodel, x)
end

export extrapolation_distance

"""
    effective_triangularity(tri_lo::T, tri_up::T) where {T<:Real}a

Effective triangularity to be used as an EPED input. Defined as:
tri_eff = (2/3)*tri_min + (1/3)*tri_max
where tri_min is the minimum of upper and lower triangularity, and tri_max is the maximum
"""
function effective_triangularity(tri_lo::T, tri_up::T) where {T<:Real}
    tri_min = min(tri_lo, tri_up)
    tri_max = max(tri_lo, tri_up)
    return (2.0 / 3.0) * tri_min + (1.0 / 3.0) * tri_max
end

#= ====================================================== =#
#  EPED-NN deep ensemble (12-input, height-only, with UQ)
#= ====================================================== =#
# 5-model deep ensemble predicting the pedestal HEIGHT p_E1 (MPa), with an ensemble σ for UQ.
# Adds nesep_ratio and tesep over the legacy 10-input EPED1NNmodel. Trained on the old
# multi-machine DB + new ITER omega-star scans (iter_claude/eped/train_ensemble.py). The forward
# pass is a faithful port of eped/eped-api/server.py (validated to reproduce its predictions).

const ENSEMBLE_INPUT_COLS = ["a", "betan", "bt", "delta", "ip", "kappa", "m", "neped", "nesep_ratio", "r", "tesep", "zeffped"]

struct EPEDNNEnsemble <: EPEDmodel
    models::Vector{Vector{Tuple{Matrix{Float64},Vector{Float64}}}}  # [model][layer] = (W (out,in), b)
    xm::Vector{Float64}
    xs::Vector{Float64}
    ym::Vector{Float64}
    ys::Vector{Float64}
    yp::Matrix{Float64}       # power-law coeffs (n_out, 13): [bias, 12 log-input coeffs]
    xbounds::Matrix{Float64}  # (12, 2) on the transformed (delta+1, abs) inputs
    n_inputs::Int
    n_outputs::Int
end

_ensemble_gelu(x) = 0.5 * x * (1.0 + tanh(sqrt(2.0 / pi) * (x + 0.044715 * x^3)))

function _ensemble_layers(raw)
    layers = Tuple{Matrix{Float64},Vector{Float64}}[]
    for li in (0, 2, 4, 6)  # Linear layers in the Sequential (GELU at 1,3,5)
        wrows = raw["net.$(li).weight"]                              # PyTorch [out][in]
        W = permutedims(reduce(hcat, [Float64.(r) for r in wrows])) # -> (out, in)
        b = Float64.(raw["net.$(li).bias"])
        push!(layers, (W, b))
    end
    return layers
end

_rows_to_matrix(rows) = reduce(vcat, [permutedims(Float64.(r)) for r in rows])

"""
    loadensemble(dirname="eped_ensemble")

Load the 5-model EPED-NN ensemble from `data/<dirname>/` (`preprocessing.json` + `model_{0-4}_weights.json`).
"""
Memoize.@memoize function loadensemble(dirname::String="eped_ensemble")
    base = joinpath(Base.dirname(Base.dirname(@__FILE__)), "data", dirname)
    pp = JSON.parsefile(joinpath(base, "preprocessing.json"))
    models = [_ensemble_layers(JSON.parsefile(joinpath(base, "model_$(m)_weights.json"))) for m in 0:4]
    return EPEDNNEnsemble(
        models,
        Float64.(pp["xm"]), Float64.(pp["xs"]),
        Float64.(pp["ym"]), Float64.(pp["ys"]),
        _rows_to_matrix(pp["yp"]), _rows_to_matrix(pp["xbounds"]),
        Int(pp["n_inputs"]), Int(pp["n_outputs"]))
end

function _ensemble_forward(model, x::Vector{Float64})
    n = length(model)
    for (i, (W, b)) in enumerate(model)
        x = W * x .+ b
        i < n && (x = _ensemble_gelu.(x))
    end
    return x
end

# Transform a raw 12-input vector the way training did: delta+1, then abs() of everything.
_ensemble_xabs(x12::AbstractVector{<:Real}) = (xabs = collect(float.(x12)); xabs[4] += 1.0; abs.(xabs))

"""
    ensemble_predict(ens::EPEDNNEnsemble, x12) -> (mean, std)

Pedestal height p_E1 (MPa): ensemble mean and population std over the 5 nets.
`x12` is in `ENSEMBLE_INPUT_COLS` order.
"""
function ensemble_predict(ens::EPEDNNEnsemble, x12::AbstractVector{<:Real})
    @assert length(x12) == 12 "ensemble expects 12 inputs in ENSEMBLE_INPUT_COLS order"
    xabs = _ensemble_xabs(x12)
    xnorm = (xabs .- ens.xm) ./ ens.xs
    logx = log.(max.(xabs, 1e-30))
    ypl = [exp(ens.yp[k, 1] + sum(@view(ens.yp[k, 2:end]) .* logx)) for k in 1:ens.n_outputs]
    preds = Vector{Float64}(undef, length(ens.models))
    for (j, model) in enumerate(ens.models)
        resid = _ensemble_forward(model, xnorm) .* ens.ys .+ ens.ym
        y = (ypl .+ resid) .^ 2
        preds[j] = y[1] * xabs[8]   # undo density normalization (height = output 1; neped at idx 8)
    end
    μ = sum(preds) / length(preds)
    σ = sqrt(sum((preds .- μ) .^ 2) / length(preds))   # population std (matches numpy default)
    return (mean=μ, std=σ)
end

"""
    extrapolation_distance(ens::EPEDNNEnsemble, x12)

Per-axis normalized distance outside the (transformed) training bounds; 0 = inside the box.
"""
function extrapolation_distance(ens::EPEDNNEnsemble, x12::AbstractVector{<:Real})
    xabs = _ensemble_xabs(x12)
    per = Dict{String,Float64}()
    max_dist = 0.0
    worst = ""
    for ix in eachindex(xabs)
        xmin, xmax = ens.xbounds[ix, 1], ens.xbounds[ix, 2]
        rng = xmax - xmin
        rng <= 0.0 && continue
        d = max(0.0, (xmin - xabs[ix]) / rng, (xabs[ix] - xmax) / rng)
        per[ENSEMBLE_INPUT_COLS[ix]] = d
        if d > max_dist
            max_dist = d
            worst = ENSEMBLE_INPUT_COLS[ix]
        end
    end
    return (per_input=per, max_distance=max_dist, worst_input=worst)
end

"""
    ensemble_uncertainty(ens, x12; sigma_threshold=0.05)

Combined out-of-distribution UQ for one operating point. Returns a NamedTuple:
  - `height`, `sigma`      : ensemble mean & std of p_E1 (MPa)
  - `sigma_frac`           : sigma/height (ensemble fractional uncertainty)
  - `extrapolation`        : max per-axis normalized distance outside training bounds (0 = in-box)
  - `sigma_frac_combined`  : max(sigma_frac, extrapolation) — ensemble σ% in-box, geometric out-of-box
  - `in_distribution`      : sigma_frac < sigma_threshold && extrapolation == 0

σ% catches in-box-but-off-manifold points where the ensemble disagrees; the geometric distance
catches deep extrapolation where the ensemble collapses onto the power law (false confidence). The
two are complementary — see iter_claude/eped/CLAUDE.md "UQ as an out-of-distribution filter".
"""
function ensemble_uncertainty(ens::EPEDNNEnsemble, x12::AbstractVector{<:Real}; sigma_threshold::Real=0.05)
    p = ensemble_predict(ens, x12)
    extr = extrapolation_distance(ens, x12).max_distance
    sfrac = p.std / max(p.mean, 1e-30)
    return (height=p.mean, sigma=p.std, sigma_frac=sfrac, extrapolation=extr,
        sigma_frac_combined=max(sfrac, extr),
        in_distribution=(sfrac < sigma_threshold && extr == 0.0))
end

"""
    ensemble_uncertainty(ens, input::InputEPED; nesep_ratio=0.25, tesep=75.0, sigma_threshold=0.05)

Convenience wrapper from the legacy 10-input `InputEPED` (supplies the two new inputs).
"""
function ensemble_uncertainty(ens::EPEDNNEnsemble, input::InputEPED; nesep_ratio::Real=0.25, tesep::Real=75.0, sigma_threshold::Real=0.05)
    x12 = [input.a, input.betan, input.bt, input.delta, input.ip, input.kappa, input.m, input.neped, nesep_ratio, input.r, tesep, input.zeffped]
    return ensemble_uncertainty(ens, x12; sigma_threshold)
end

export loadensemble, ensemble_predict, ensemble_uncertainty

const document = Dict()
document[Symbol(@__MODULE__)] = [name for name in Base.names(@__MODULE__; all=false, imported=false) if name != Symbol(@__MODULE__)]

end # module
