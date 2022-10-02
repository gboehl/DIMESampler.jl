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

chains, lprobs, pdist = RunDIME(LogProb, initchain, niter, progress=true)

sample = chains[end-Int(niter/4):end,:,1][:]

tval = 1.6811348536497772
@test isapprox(median(sample), tval)
