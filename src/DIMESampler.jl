# -*- coding: utf-8 -*-

"""A sampler using adaptive differential evolution proposals.

This is a standalone julia version of the `Adaptive Differential Ensemble MCMC sampler` as prosed in `Ensemble MCMC Sampling for Robust Bayesian Inference <https://gregorboehl.com/live/ademc_boehl.pdf>`_.
"""
module DIMESampler
    
using Distributions, ProgressBars, Printf, LinearAlgebra, StatsFuns

export RunDIME, CreateDIMETestFunc, DIMETestFuncMarginalPDF

@doc raw"""
    DIMESampler(lprobFunc::Function, init::Array, niter::Int; sigma::Float64=1e-5, gamma=nothing, aimh_prob::Float64=0.05, nsamples_proposal_dist=nothing, df_proposal_dist::Int=10, delta::Float64=.999, progress::Bool=true)

# Arguments
- `lprobFunc::Function`: the likelihood function to be sampled. Expected to be vectorized.
- `init::Array`: the initial ensemble. Used to infer the number of chains and the dimensionality of `lprobFunc`. A rule of thumb for the number of chains is :math:`nchain = 5*ndim`.
- `niter::Int`: the number of iterations to be run.
- `sigma::Float=1e-5`: the standard deviation of the Gaussian used to stretch the proposal vector.
- `gamma::Float=nothing`: the mean stretch factor for the proposal vector. By default, it is ``2.38 / \sqrt{2\,\mathrm{ndim}}`` as recommended by `ter Braak (2006) <http://www.stat.columbia.edu/~gelman/stuff_for_blog/cajo.pdf>`_.
- `aimh_prob::Float=0.1`: the probability to draw a AIMH proposal. 
- `delta::Float=0.999`: the decay parameter for mean and covariance of the AIMH proposals.
- `df_proposal_dist::Float=10`: the degrees of freedom of the multivariate t distribution used for AIMH proposals.
"""
function RunDIME(lprobFunc::Function, init::Array, niter::Int; sigma::Float64=1e-5, gamma=nothing, aimh_prob::Float64=0.1, df_proposal_dist::Int=10, delta::Float64=.999, progress::Bool=true)

    ndim, nchain = size(init)
    isplit = nchain ÷ 2

    # get some default values
    dft = df_proposal_dist

    if gamma == nothing 
        g0 = 2.38 / sqrt(2 * ndim)
    else
        g0 = gamma
    end

    # fix that MvTDist does not accept positive demi-definite covariance matrices
    fixPSD = Matrix(1e-16I, ndim, ndim)

    # initialize
    ccov = Matrix(1.0I, ndim, ndim)
    cmean = zeros(ndim)
    dist = MvTDist(dft, cmean, ccov + fixPSD)
    accepted = ones(nchain)
    cumlweight = -Inf

    # calculate intial values
    x = copy(init)
    lprob = lprobFunc(x)
    # split ensemble
    xref, xcur = (@view x[:, 1:isplit+1]), (@view x[:, isplit+1:end])

    # preallocate
    lprobs = Array{Float64,2}(undef, niter, nchain)
    lprobs = fill!(lprobs, 0.0)

    chains = Array{Float64,3}(undef, niter, nchain, ndim)
    chains = fill!(chains, 0.0)

    # optional progress bar
    if progress
        iter = ProgressBar(1:niter)
    else
        iter = 1:niter
    end

    @inbounds for i in iter

        # calculate stats for current ensemble
        # log weight of current ensemble
        lweight = logsumexp(lprobs) + log(sum(accepted)) - log(nchain)

        ncov = cov(transpose(x))
        nmean = mean(x, dims=2)

        # update AIMH proposal distribution
        newcumlweight = logaddexp(cumlweight, lweight)
        statelweight = cumlweight - newcumlweight
        ccov = exp(statelweight) * ccov + exp(lweight - newcumlweight) * ncov
        cmean = exp(statelweight) * cmean + exp(lweight - newcumlweight) * nmean
        cumlweight = newcumlweight + log(delta)
        naccepted = 0

        # must iterate over current and reference ensemble
        @inbounds for complementary_ensemble in (false,true)

            # define current ensemble
            if complementary_ensemble
                xcur, xref = (@view x[:, 1:isplit+1]), (@view x[:, isplit+1:end])
                lprobcur = @view lprob[1:isplit+1]
            else
                xref, xcur = (@view x[:, 1:isplit+1]), (@view x[:, isplit+1:end])
                lprobcur = @view lprob[isplit+1:end]
            end
            cursize = size(xcur)[2]
            refsize = nchain - cursize + 1

            # get differential evolution proposal
            # draw the indices of the complementary chains
            i1 = collect(0:cursize-1) .+ rand(1:cursize-1, cursize)
            i2 = collect(0:cursize-1) .+ rand(1:cursize-2, cursize)
            i2[i2 .>= i1] .+= 1
            # add small noise and calculate proposal
            f = sigma * rand(Normal(0,1), (1,cursize))
            q = xcur + g0 * (xref[:,(i1 .% refsize) .+ 1] - xref[:,(i2 .% refsize) .+ 1]) .+ f
            factors = zeros(cursize)

            # get AIMH proposals if any chain is drawn
            xchnge = rand(Uniform(0,1), cursize) .<= aimh_prob

            if sum(xchnge) > 0
                # draw alternative candidates and calculate their proposal density
                dist = MvTDist(dft, cmean[:], ccov*(dft - 2)/dft + fixPSD)

                xcand = rand(dist, sum(xchnge))
                lprop_old = logpdf(dist, xcur[:, xchnge])
                lprop_new = logpdf(dist, xcand)

                # update proposals and factors
                q[:,xchnge] = xcand
                factors[xchnge] = lprop_old - lprop_new
            end

            # Metropolis-Hasings 
            newlprob = lprobFunc(q)
            lnpdiff = factors + newlprob - lprobcur
            accepted = lnpdiff .> log.(rand(Uniform(0,1), cursize))
            naccepted += sum(accepted)
            # update chains
            xcur[:,accepted] = q[:,accepted]
            lprobcur[accepted] = newlprob[accepted]
        end

        # store
        chains[i,:,:] = transpose(x)
        lprobs[i,:] = lprob

        if progress
            set_description(iter, string(@sprintf("[ll/MAF: %7.3f(%1.0e)/%2.0d%% | %1.0e]", maximum(lprob), std(lprob), 100*naccepted/nchain, statelweight)))
        end
    end

    return chains, lprobs, dist
end

@doc raw"""
    CreateDIMETestFunc(ndim::Int, weight::Float, distance::Float, scale::Float)

Create a trimodal Gaussian mixture for testing.
"""
function CreateDIMETestFunc(ndim, weight, distance, scale)

    covm = I(ndim)*scale
    meanm = zeros(ndim)
    meanm[1] = distance

    lw1 = log(weight[1])
    lw2 = log(weight[2])
    lw3 = log(1-weight[1]-weight[2])

    dist = MvNormal(zeros(ndim), covm)

    function TestLogProb(p)

        stack = cat(lw1 .+ logpdf(dist, p .+ meanm), 
                    lw2 .+ logpdf(dist, p), 
                    lw3 .+ logpdf(dist, p .- meanm), 
                    dims=2)
        return logsumexp(stack, dims=2)[:]

    end
end

@doc raw"""
    DIMETestFuncMarginalPDF(x::Array, cov_scale::Float, distance::Float, weight::Float)

Get the marginal PDF over the first dimension of the test distribution.
"""
function DIMETestFuncMarginalPDF(x, cov_scale, distance, weight)

    normd = Normal(0, sqrt(cov_scale))

    return weight[1]*pdf.(normd, x .+ distance) + weight[2]*pdf.(normd, x) + (1-weight[1]-weight[2])*pdf.(normd, x .- distance)
end

end

