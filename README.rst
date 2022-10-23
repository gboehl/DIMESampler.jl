DIMESampler.jl
========

**Differential-Independence Mixture Ensemble ("DIME") MCMC sampling for Julia**

This is a standalone Julia implementation of the DIME sampler proposed in `Ensemble MCMC Sampling for DSGE Models <https://gregorboehl.com/live/dime_mcmc_boehl.pdf>`_. *(Gregor Boehl, 2022, CRC 224 discussion paper series)*.

The sampler has a series of advantages over conventional samplers:

#. DIME MCMC is a (very fast) gradient-free **global multi-start optimizer** and, at the same time, a **MCMC sampler** that converges to the posterior distribution. This makes any posterior mode density maximization prior to MCMC sampling superfluous.
#. The DIME sampler is pretty robust for odd shaped, **multimodal distributions**.
#. DIME MCMC is **parallelizable**: many chains can run in parallel, and the necessary number of draws decreases almost one-to-one with the number of chains.
#. DIME proposals are generated from an **endogenous and adaptive proposal distribution**, thereby providing close-to-optimal proposal distributions for black box target distributions without the need for manual fine-tuning.

.. image:: https://github.com/gboehl/DIMESampler.jl/blob/main/docs/dist.png?raw=true
  :width: 800
  :alt: Sample and target distribution

Installation
------------

Just get the package from the official Julia registry:

.. code-block:: julia

   using Pkg; Pkg.add("DIMESampler")


There exist complementary implementations `for Python <https://github.com/gboehl/emcwrap>`_ and `for matlab <https://github.com/gboehl/dime-mcmc-matlab>`_.

Usage
-----

The core functionality is included in the function ``RunDIME``:

.. code-block:: julia

    # import package
    using DIMESampler

    # define your density function
    function LogProb(x):
        ...
        return lprob
    end

    # define the initial ensemble
    initchain = ...

    # define the number of iterations to run
    niter = ...

    # off you go sampling
    chains, lprobs, propdist = RunDIME(LogProb, initchain, niter)
    ...

The function returning the log-density must be vectorized, i.e. able to evaluate inputs with shape ``[ndim, :]``. 

Tutorial
--------

Define a challenging example distribution **with three separate modes** (the distribution from the figure above):

.. code-block:: julia

    # some imports
    using DIMESampler, Distributions, Random, LinearAlgebra, Plots

    # make it reproducible
    Random.seed!(1)

    # define distribution
    m = 2
    cov_scale = 0.05
    weight = (0.33, 0.1)
    ndim = 35

    LogProb = CreateDIMETestFunc(ndim, weight, m, cov_scale)

``LogProb`` will now return the log-PDF of a 35-dimensional Gaussian mixture.

**Important:** the function returning the log-density must be vectorized, i.e. able to evaluate inputs with shape ``[ndim, :]``. If you want to make use of parallelization (which is one of the central advantages of ensemble MCMC), you may want to ensure that this function evaluates its vectorized input in parallel, i.e. using ``pmap`` from `Distributed <https://docs.julialang.org/en/v1/stdlib/Distributed/>`_:

.. code-block:: julia

    LogProbParallel(x) = pmap(LogProb, eachslice(x, dims=2))

For this example this is overkill since the overhead from parallelization is huge. Just using the vectorized ``LogProb`` is perfect.

Next, define the initial ensemble. In a Bayesian setup, a good initial ensemble would be a sample from the prior distribution. Here, we will go for a sample from a rather flat Gaussian distribution.

.. code-block:: julia

    initvar = 2
    nchain = ndim*5 # a sane default
    initcov = I(ndim)*initvar
    initmean = zeros(ndim)
    initchain = rand(MvNormal(initmean, initcov), nchain)

Setting the number of parallel chains to ``5*ndim`` is a sane default. For highly irregular distributions with several modes you should use more chains. Very simple distributions can go with less. 

Now let the sampler run for 5000 iterations.

.. code-block:: julia

    niter = 5000
    chains, lprobs, propdist = RunDIME(LogProb, initchain, niter, progress=true, aimh_prob=0.1)

.. code-block::

    [ll/MAF:  12.187(4e+00)/19% | -5e-04] 100.0%┣███████████████████████████████┫ 5.0k/5.0k [00:15<00:00, 198it/s]

The setting of ``aimh_prob`` is the actual default value. For less complex distributions (e.g. distributions closer to Gaussian) a higher value can be chosen, which accelerates burn-in. The information in the progress bar has the structure ``[ll/MAF: <maximum log-prob>(<standard deviation of log-prob>)/<mean acceptance fraction> | <log state weight>]...``, where ``<log state weight>`` is the current log-weight on the history of the proposal distribution. The closer this value is to zero (i.e. the actual weight to one), the less relevant are current ensembles for the estimated proposal distribution. It can hence be seen as a measure of convergence.

The following code creates the figure above, which is a plot of the marginal distribution along the first dimension (remember that this actually is a 35-dimensional distribution).

.. code-block:: julia

   # analytical marginal distribution in first dimension
    x = range(-4,4,1000)
    mpdf = DIMETestFuncMarginalPDF(x, cov_scale, m, weight)

    plot(x, mpdf, label="Target", lw=2, legend_position=:topleft)
    plot!(x, pdf.(Normal(0, sqrt(initvar)), x), label="Initialization")
    plot!(x, pdf.(TDist(10), (x .- propdist.μ[1])./sqrt(propdist.Σ[1,1]*10/8)), label="Final proposal")
    # histogram of the actual sample
    histogram!(chains[end-niter÷2:end,:,1][:], normalize=true, alpha=.5, label="Sample", color="black", bins=100)

To ensure proper mixing, let us also have a look at the MCMC traces, again focussing on the first dimension:

.. code-block:: julia

   plot(chains[:,:,1], color="cyan4", alpha=.1, legend=false, size=(900,600))

.. image:: https://github.com/gboehl/DIMESampler.jl/blob/main/docs/traces.png?raw=true
  :width: 800
  :alt: MCMC traces
  
Note how chains are also switching between the three modes because of the global proposal kernel.

While DIME is a MCMC sampler, it can straightforwardly be used as a global optimization routine. To this end, specify some broad starting region (in a non-Bayesian setup there is no prior) and let the sampler run for an extended number of iterations. Finally, assess whether the maximum value per ensemble did not change much in the last few hundred iterations. In a normal Bayesian setup, plotting the associated log-likelihood over time also helps to assess convergence to the posterior distribution.

.. code-block:: julia

   plot(lprobs[:,:], color="orange4", alpha=.05, legend=false, size=(900,300))
   plot!(maximum(lprobs)*ones(niter), color="blue3")

.. image:: https://github.com/gboehl/DIMESampler.jl/blob/main/docs/lprobs.png?raw=true
  :width: 800
  :alt: Log-likelihoods

References
----------

If you are using this software in your research, please cite

.. code-block::

    @techreport{boehl2022mcmc,
    title         = {Ensemble MCMC Sampling for DSGE Models},
    author        = {Boehl, Gregor},
    year          = 2022,
    institution   = {CRC224 discussion paper series}
    }
