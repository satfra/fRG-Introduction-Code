using OrdinaryDiffEq, LinearAlgebra
using SparseArrays, FastBroadcast, PreallocationTools
using BenchmarkTools
using Plots

include("include/FiniteDifference.jl")
include("include/runner.jl")

# This is at fixed d = 4, with d-1-dimensional regulators. Feel free to generalize this to arbitrary dimensions

##################################################
## Global definitions
##################################################

# parameter struct holding all model parameters
struct ParametersQM
  grid::Array{Float64} # the grid
  n_fields::Int # number of fields - for this model, it's just mπ²
  Nf::Int # number of flavors
  Nc::Int # number of colors
  Λ::Float64 # UV cutoff scale
  λ::Float64 # quartic coupling of the mesons at UV scale
  mϕ2::Float64 # mass of the mesons at UV scale
  hϕ::Float64 # Yukawa coupling
  μ::Float64 # quark chemical potential
  T::Float64 # temperature
end

# for verbosity, the field directions. Here, there's only one, ρϕ = σ² / 2. Note, that this is just the Index!
const ρϕ = 1

# buffers for intermediate variables, to avoid allocations in the kernel
buffer1 = 0
buffer2 = 0

##################################################
## Flow equations
##################################################

# the quark contributions to the flow of V_k(ρϕ)
@fastmath function quarkFlow(ρϕ, μq, hϕ, k, T, Nc, Nf)
  Eq = sqrt(k^2 + hϕ^2 * ρϕ / Nf)
  return Nc * Nf * (k^5 * (-tanh((Eq - μq) / (2 * T)) - tanh((Eq + μq) / (2 * T)))) / (6 * Eq * π^2)
end

# all meson contributions to the flow of V_k(ρϕ)
@fastmath function mesonFlow(mπ2, mσ2, k, T, Nf)
  flowπ = (k^5 * coth(sqrt(k^2 + mπ2) / (2 * T))) / (12 * sqrt(k^2 + mπ2) * π^2)
  flowσ = (k^5 * coth(sqrt(k^2 + mσ2) / (2 * T))) / (12 * sqrt(k^2 + mσ2) * π^2)
  return flowσ + (Nf^2 - 1) * flowπ
end

##################################################
## Kernel and Initialization function
##################################################


@fastmath @inbounds function kernel!(du, u, p::ParametersQM, t)
  println("time: ", t)

  # In the above, we have the identification
  # u = mπ²
  # This method computes the flow of mπ²
  # du = = ∂_t(mπ²) = -∂_t ∂_ρϕ V_k
  # In other words, we solve a hyperbolic PDE for the flow of mπ²

  ##################################################
  # initialization

  # make sure du is zero at start
  du .= 0

  # access the part of the grid that corresponds to ρϕ
  ρϕ_grid = @view p.grid[:, ρϕ]

  # compute the current RG scale from the RG-time t. Note that t = -log(k/Λ), in contrast to the usual definition t = log(k/Λ)
  k = exp(-t) * p.Λ

  ##################################################
  # compute the sigma mass squared (Ms2), given by mπ² + 2 ρϕ ∂_ρϕ mπ²
  # we take a left-derivative, so that diffusive effects will in the end factor in as a central derivative

  Ms2 = get_tmp(buffer1, zero(u[1]))
  D_l!(Ms2, u, ρϕ_grid, ρϕ)
  @.. thread = true Ms2 .= u .+ 2 .* ρϕ_grid .* Ms2

  ##################################################
  # Evaluate the flow and take derivatives

  # Obtain a temporary array for the flux storage
  flux = get_tmp(buffer2, zero(du[1]))

  # evaluate the flows
  try
    @.. thread = true flux .= -(quarkFlow.(ρϕ_grid, p.μ, p.hϕ, k, p.T, p.Nc, p.Nf)
                                .+
                                mesonFlow.(u, Ms2, k, p.T, p.Nf)
    )
  catch e
    # In case of a failure, invalidate the result, so that the solver can try again
    du .= NaN
    return
  end

  # take a right-derivative of the flux for the change of u
  D_r!(du, flux, ρϕ_grid, ρϕ)

  # prevent this function from returning anything.
  nothing
end

function init(::typeof(kernel!), p::ParametersQM)
  # first, create the buffers for the kernel
  # a buffer holds one field component, therefore its size is size(p.grid)[1:end-1]
  barr = zeros(Float64, (size(p.grid)[1:end-1]...))
  global buffer1 = DiffCache(barr)
  global buffer2 = DiffCache(barr)

  # then, initialize the field
  arr = zeros(Float64, (size(p.grid)[1:end-1]...))
  # the initial condition is V_UV = mϕ² ρϕ + λ/4 ρϕ², i.e. we have an inital mass and a quartic coupling
  arr[:, 1] .= (p.λ ./ 2 .* p.grid[:, ρϕ]
                .+
                p.mϕ2)
  return collect(Iterators.flatten(arr))
end

##################################################
## Grid Setup
##################################################

# The grid can be chosen freely. Here, we choose a grid that is fine enough to resolve the flow of the fields
# Ωϕ is the grid for the field ρϕ
Ωϕ = unique(vcat(0:2e-4:6e-4, 6e-4:4e-4:3e-3, 3e-3:2e-4:5e-3, 5e-3:1e-3:1e-2))
# grid holds the grid in all directions. As we only have one direction, we have a 1D grid and populate only the first and only column
grid = zeros(Float64, (length(Ωϕ), 1))
grid[:, 1] = Ωϕ[:]

##################################################
## Run the program
##################################################

# set up the parameters
parameters = ParametersQM(grid, 1, 2, 3, 0.65, 71.6, 0.0, 7.2, 0.32, 1e-2);
# show the parameters
dump(parameters)

# The following gets the sparsity pattern of the Jacobian, which is needed for an implicit solver
jac_sparsity = sparsityNN(parameters)

# set up the ODE entry; this gives the ODE solver all the information it needs to solve the ODE
odeargs = Dict(:abstol => 1e-12, :reltol => 1e-10, :dt => 1e-4, :dtmin => 1e-14, :saveat => 0.1)
entry = ODEentry(:(QNDF()), kernel!, odeargs, parameters, 5.0)

# run the ODE solver
result = run(entry, jac_sparsity)

##################################################
## Visualization
##################################################

# show the result at three times, t = 0, 1, 5
result_data_start = reshape(result.u[begin], (size(parameters.grid)[1:end-1]..., parameters.n_fields))
result_data_mid = reshape(result.u[10], (size(parameters.grid)[1:end-1]..., parameters.n_fields))
result_data_end = reshape(result.u[end], (size(parameters.grid)[1:end-1]..., parameters.n_fields))
plot(Ωϕ, result_data_start[:, 1], label="t = 0")
plot!(Ωϕ, result_data_mid[:, 1], label="t = 1")
plot!(Ωϕ, result_data_end[:, 1], label="t = 5")

##################################################
# Phase diagram
##################################################

#...