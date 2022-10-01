# -*- coding: utf-8 -*-

"""A sampler using adaptive differential evolution proposals.

This is a standalone julia version of the `Adaptive Differential Ensemble MCMC sampler` as prosed in `Ensemble MCMC Sampling for DSGE Models <https://gregorboehl.com/live/ademc_boehl.pdf>`_.
"""
module DIMESampler
    
using Distributions, ProgressBars, Printf, LinearAlgebra, StatsFuns

export RunDIME, CreateDIMETestFunc, DIMETestFuncMarginalPDF

@doc raw"""
    DIMESampler(lprobFunc::Function, init::Array, niter::Int; sigma::Float64=1e-5, gamma=nothing, aimh_prob::Float64=0.05, nsamples_proposal_dist=nothing, df_proposal_dist::Int=10, progress::Bool=true)

# Arguments
- `lprobFunc::Function`: the likelihood function to be sampled. Expected to be vectorized.
- `init::Array`: the initial ensemble. Used to infer the number of chains and the dimensionality of `lprobFunc`. A rule of thumb for the number of chains is :math:`nchain = 5*ndim`.
- `niter::Int`: the number of iterations to be run.
- `sigma::Float=1e-5`: the standard deviation of the Gaussian used to stretch the proposal vector.
- `gamma::Float=nothing`: the mean stretch factor for the proposal vector. By default, it is ``2.38 / \sqrt{2\,\mathrm{ndim}}`` as recommended by `ter Braak (2006) <http://www.stat.columbia.edu/~gelman/stuff_for_blog/cajo.pdf>`_.
- `aimh_prob::Float=0.1`: the probability to draw a AIMH proposal. 
- `neff_proposal_dist::Int=nothing`: the window size used to calculate the rolling-window covariance estimate. By default this is the number of unique elements in the proposal mean and covariance divided by mean ACF times mean accetance ratio ``4 d(d+3)``.
- `df_proposal_dist::Float=10`: the degrees of freedom of the multivariate t distribution used for AIMH proposals.
"""
function RunDIME(lprobFunc::Function, init::Array, niter::Int; sigma::Float64=1e-5, gamma=nothing, aimh_prob::Float64=0.1, neff_prop_dist=nothing, df_proposal_dist::Int=10, progress::Bool=true)

    ndim, nchain = size(init)

    # get some default values
    dft = df_proposal_dist

    if gamma == nothing 
        g0 = 2.38 / sqrt(2 * ndim)
    else
        g0 = gamma
    end

    if neff_prop_dist == nothing
        npdist = 4*ndim*(ndim + 3)
    else
        npdist = neff_prop_dist
    end

    # calculate intial values
    x = copy(init)
    lprob = lprobFunc(x)
    ccov = cov(transpose(x)) # initialization does not matter
    cmean = mean(x, dims=2) # initialization does not matter
    accepted = ones(nchain)
    cumlweight = -Inf

    # preallocate
    chains = zeros((niter, nchain, ndim))
    lprobs = zeros((niter, nchain))
    dist = MvTDist(dft, cmean[:], ccov*(dft - 2)/dft)

    # optional progress bar
    if progress
        iter = ProgressBar(1:niter)
    else
        iter = 1:niter
    end

    for i in iter

        # get differential evolution proposal
        # draw the indices of the complementary chains
        i1 = collect(0:nchain-1) .+ rand(1:nchain-1, nchain)
        i2 = collect(0:nchain-1) .+ rand(1:nchain-2, nchain)
        i2[i2 .>= i1] .+= 1
        # add small noise and calculate proposal
        f = sigma * rand(Normal(0,1), (1,nchain))
        q = x + g0 * (x[:,(i1 .% nchain) .+ 1] - x[:,(i2 .% nchain) .+ 1]) .+ f
        factors = zeros(nchain)

        # log weight of current ensemble
        lweight = logsumexp(lprobs) + log(sum(accepted)) - log(nchain)

        # calculate stats for current ensemble
        ncov = cov(transpose(x))
        nmean = mean(x, dims=2)

        # update AIMH proposal distribution
        newcumlweight = logaddexp(cumlweight, lweight)
        ccov = exp(cumlweight - newcumlweight) * ccov + exp(lweight - newcumlweight) * ncov
        cmean = exp(cumlweight - newcumlweight) * cmean + exp(lweight - newcumlweight) * nmean

        # get AIMH proposal
        xchnge = rand(Uniform(0,1), nchain) .<= aimh_prob

        # draw alternative candidates and calculate their proposal density
        dist = MvTDist(dft, cmean[:], ccov*(dft - 2)/dft)

        xcand = rand(dist, sum(xchnge))
        lprop_old = logpdf(dist, x[:, xchnge])
        lprop_new = logpdf(dist, xcand)

        # update proposals and factors
        q[:,xchnge] = xcand
        factors[xchnge] = lprop_old - lprop_new

        # Metropolis-Hasings 
        newlprob = lprobFunc(q)
        lnpdiff = factors + newlprob - lprob
        accepted = lnpdiff .> log.(rand(Uniform(0,1), nchain))
        naccepted = sum(accepted)
        # update chains
        x[:,accepted] = q[:,accepted]
        lprob[accepted] = newlprob[accepted]

        # store
        chains[i,:,:] = transpose(x)
        lprobs[i,:] = lprob

        if progress
            set_description(iter, string(@sprintf("[ll/MAF: %.3f(%1.0e)/%d%%]", maximum(lprob), std(lprob), 100*naccepted/nchain)))
        end
    end

    return chains, lprobs, dist
end

@doc raw"""
    CreateDIMETestFunc(ndim::Int, weight::Float, distance::Float, scale::Float)

Create a bimodal Gaussian mixture for testing.
"""
function CreateDIMETestFunc(ndim, weight, distance, scale)

    covm = I(ndim)*scale
    mean1 = zeros(ndim)
    mean2 = copy(mean1)
    mean1[1] = -distance/2
    mean2[1] = +distance/2

    lw1 = log(weight)
    lw2 = log(1-weight)

    dist = MvNormal(zero(mean1), covm)

    function TestLogProb(p)

        return logaddexp.(lw1 .+ logpdf(dist, p .- mean1), 
                          lw2 .+ logpdf(dist, p .- mean2))

    end
end

@doc raw"""
    MarginalPDF(x::Array, cov_scale::Float, distance::Float, weight::Float)

Get the marginal PDF over the first dimension of the test distribution.
"""
function DIMETestFuncMarginalPDF(x, cov_scale, distance, weight)
    normd = Normal(0, sqrt(cov_scale))
    return (1-weight)*pdf.(normd, x .- distance/2) + weight*pdf.(normd, x .+ distance/2)
end

end
