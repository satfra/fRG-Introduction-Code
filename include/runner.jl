mutable struct ODEentry{A,P}
  stepper::Expr
  kernel!::Function
  ODEargs::A
  params::P
  maxT::Float64
end

function run(e::ODEentry)
  u0 = init(e.kernel!, e.params)

  f = ODEFunction(e.kernel!)
  problem = ODEProblem(f, u0, (0.0, e.maxT), e.params)
  solve(problem, eval(e.stepper); e.ODEargs...)
end

function run(e::ODEentry, jac_sparsity)
  u0 = init(e.kernel!, e.params)

  f = ODEFunction(e.kernel!; jac_prototype=jac_sparsity)
  problem = ODEProblem(f, u0, (0.0, e.maxT), e.params)
  solve(problem, eval(e.stepper); e.ODEargs...)
end

function run(e1::ODEentry, e2::ODEentry, jac_sparsity)
  u0 = init(e1.kernel!, e1.params)

  f1 = ODEFunction(e1.kernel!; jac_prototype=jac_sparsity)
  problem = ODEProblem(f1, u0, (0.0, e1.maxT), e1.params)
  res1 = solve(problem, eval(e1.stepper); e1.ODEargs...)

  f2 = ODEFunction(e2.kernel!; jac_prototype=jac_sparsity)
  problem = ODEProblem(f2, res1.u[end], (e1.maxT, e2.maxT), e2.params)
  solve(problem, eval(e2.stepper); e2.ODEargs...)
end