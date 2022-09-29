# -*- coding: utf-8 -*-

"""A sampler using adaptive differential evolution proposals.

This is a standalone julia version of the `Adaptive Differential Ensemble MCMC sampler` as prosed in `Ensemble MCMC Sampling for DSGE Models <https://gregorboehl.com/live/ademc_boehl.pdf>`_.
"""
module DIMESampler
    
using Distributions, ProgressBars, Printf, LinearAlgebra, StatsFuns

export RunDIME, CreateDIMETestFunc, DIMETestFuncMarginalPDF

@doc raw"""
    DIMESampler(lprobFunc::Function, init::Array, niter::Int; sigma::Float64=1e-5, gamma=nothing, aimh_prob::Float64=0.05, df_proposal_dist::Int=10, progress::Bool=true)

# Arguments
- `lprobFunc::Function`: the likelihood function to be sampled. Expected to be vectorized.
- `init::Array`: the initial ensemble. Used to infer the number of chains and the dimensionality of `lprobFunc`. A rule of thumb for the number of chains is :math:`nchain = 5*ndim`.
- `niter::Int`: the number of iterations to be run.
- `sigma::Float=1e-5`: the standard deviation of the Gaussian used to stretch the proposal vector.
- `gamma::Float=nothing`: the mean stretch factor for the proposal vector. By default, it is ``2.38 / \sqrt{2\,\mathrm{ndim}}`` as recommended by `ter Braak (2006) <http://www.stat.columbia.edu/~gelman/stuff_for_blog/cajo.pdf>`_.
- `aimh_prob::Float=0.05`: the probability to draw a AIMH proposal. 
- `df_proposal_dist::Float=10`: the degrees of freedom of the multivariate t distribution used for AIMH proposals.
"""
function RunDIME(lprobFunc::Function, init::Array, niter::Int; sigma::Float64=1e-5, gamma=nothing, aimh_prob::Float64=0.1, df_proposal_dist::Int=10, progress::Bool=true)

    ndim, nchain = size(init)

    # get some default values
    dft = df_proposal_dist

    if gamma == nothing 
        g0 = 2.38 / sqrt(2 * ndim)
    else
        g0 = gamma
    end

    # calculate intial values
    x = copy(init)
    lprob = lprobFunc(x)
    ccov = cov(transpose(x))
    cmean = mean(x, dims=2)
    accepted = zeros(nchain)
    naccepted = sum(accepted)
    npdist = nchain

    # preallocate
    chains = zeros((niter, nchain, ndim))
    lprobs = zeros((niter, nchain))
    dist = nothing

    # optional progress bar
    if progress
        iter = ProgressBar(1:niter)
    else
        iter = 1:niter
    end

    for i in iter

        # update AIMH proposal distribution
        if naccepted > 1

            xaccepted = x[:, accepted]

            # only use newly accepted to update AIMH proposal distribution
            ncov = cov(transpose(xaccepted))
            nmean = mean(xaccepted, dims=2)

            ccov = (npdist - 1) / (naccepted + npdist - 1) * ccov + (naccepted - 1) / (naccepted + npdist - 1) * ncov
            cmean = npdist / (naccepted + npdist) * cmean + naccepted / (naccepted + npdist) * nmean
            npdist += naccepted
        end

        # get differential evolution proposal
        # draw the indices of the complementary chains
        i1 = collect(0:nchain-1) .+ rand(1:nchain-1, nchain)
        i2 = collect(0:nchain-1) .+ rand(1:nchain-2, nchain)
        i2[i2 .>= i1] .+= 1
        # add small noise and calculate proposal
        f = sigma * rand(Normal(0,1), (1,nchain))
        q = x + g0 * (x[:,(i1 .% nchain) .+ 1] - x[:,(i2 .% nchain) .+ 1]) .+ f
        factors = zeros(nchain)

        # get AIMH proposal
        xchnge = rand(Uniform(0,1), nchain) .<= aimh_prob

        if i > 1 & sum(xchnge) > 0

            # draw alternative candidates and calculate their proposal density
            dist = MvTDist(dft, cmean[:], ccov*(dft - 2)/dft)

            xcand = rand(dist, sum(xchnge))
            lprop_old = logpdf(dist, x[:, xchnge])
            lprop_new = logpdf(dist, xcand)

            # update proposals and factors
            q[:,xchnge] = xcand
            factors[xchnge] = lprop_old - lprop_new
        end

        # Metropolis-Hasings 
        newlprob = lprobFunc(q)
        lnpdiff = factors + newlprob - lprob
        accepted = lnpdiff .> log.(rand(Uniform(0,1), nchain))
        naccepted = sum(accepted)
        # update chains
        x[:,accepted] = q[:,accepted]
        lprob[accepted] = newlprob[accepted]
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
