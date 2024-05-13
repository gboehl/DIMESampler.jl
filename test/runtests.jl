using Test, DIMESampler, Distributions, Random, LinearAlgebra

Random.seed!(1)

# define distribution
m = 2
cov_scale = 0.05
weight = (0.33, .1)
ndim = 35

LogProb = CreateDIMETestFunc(ndim, weight, m, cov_scale)
LogProbParallel(x) = pmap(LogProb, eachslice(x, dims=2))

# for chain
niter = 3000
nchain = ndim*5

initmean = zeros(ndim)
initcov = I(ndim)*2
initchain = rand(MvNormal(initmean, initcov), nchain)

# check 4 real
chains, lprobs, pdist = RunDIME(LogProb, initchain, niter, progress=false)
sample = chains[end-Int(niter/4):end,:,1][:]

tval = 1.772542206271537
@test isapprox(median(sample), tval)

# check if also runs with progress and DE-MCMC only
chains, lprobs, pdist = RunDIME(LogProb, initchain, 10, progress=true, aimh_prob=0.)
