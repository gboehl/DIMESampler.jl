DIMESampler.jl
========

**Differential-Independence Mixture Ensemble MCMC sampling for Julia**

This is a standalone Julia implementation of the DIME sampler (previously ADEMC sampler) proposed in `Ensemble MCMC Sampling for DSGE Models <https://gregorboehl.com/live/ademc_boehl.pdf>`_. *(Gregor Boehl, 2022, CRC 224 discussion paper series)*.

The sampler has a series of advantages over conventional samplers:

#. At core, DIME MCMC is a (very fast) **global multi-start optimizer** that converges to the posterior distribution. This makes any posterior mode density maximization prior to MCMC sampling superfluous.
#. The DIME sampler is pretty robust for odd shaped, **bimodal distributions**.
#. DIME MCMC is **parallelizable**: many chains can run in parallel, and the necessary number of draws decreases almost one-to-one with the number of chains.
#. DIME proposals are generated from an **endogenous and adaptive proposal distribution**, thereby reducing the number of necessary meta-parameters and providing close-to-optimal proposal distributions.

Installation
------------

As long as this is not in the official repositories, download the file `DIMESampler.jl <https://github.com/gboehl/DIMESampler.jl/blob/main/src/DIMESampler.jl>`_ (from ``src``) and go for:

.. code-block:: julia

   # already load multithreading interface here and initialize
   using Distributed
   addprocs(8) # or whatever your number of cores is

   # use @everywhere to ensure that the module is known to each thread
   @everywhere push!(LOAD_PATH,<insert_path_to_DIMESampler.jl>) # insert path to DIMESampler.jl here!
   @everywhere using DIMESampler

There exists a complementary Python implementation `here <https://github.com/gboehl/emcwrap>`_.

Usage
-----

Define an example distribution:

.. code-block:: julia

    # some imports
    using Distributions, Random, LinearAlgebra, Plots

    # make it reproducible
    Random.seed!(1)

    # define distribution
    m = 2
    cov_scale = 0.05
    weight = 0.33
    ndim = 35

    LogProb = CreateDIMETestFunc(ndim, weight, m, cov_scale)

``LogProb`` will now return the log-PDF of a 35-dimensional bimodal Gaussian mixture. 
**Important:** the function returning the log-density must be vectorized, i.e. able to evaluate inputs with shape ``[ndim, :]``. If you want to make use of parallelization (which is one of the central advantages of ensemble MCMC), you may want to ensure that this function evaluates its vectorized input in parallel, i.e.:

.. code-block:: julia

    LogProbParallel(x) = pmap(LogProb, eachslice(x, dims=2))

For this example this is overkill since the overhead from parallelization is huge. Just using the vectorized ``LogProb`` is perfect.

Next, define the initial ensemble. In a Bayesian setup, a good initial ensemble would be a sample from the prior distribution. Here, we will go for a sample from a rather flat Gaussian distribution.

.. code-block:: julia

    nchain = ndim*5 # a sane default
    initcov = I(ndim)*sqrt(2)
    initchain = rand(MvNormal(zeros(ndim), initcov), nchain)

Setting the number of parallel chains to ``5*ndim`` is a sane default. For highly irregular distributions with several modes you should use more chains. Very simple distributions can go with less. 

Now let the sampler run for 2000 iterations.

.. code-block:: julia

    chains, lprobs = RunDIME(LogProb, initchain, 2000, progress=true, aimh_prob=0.05)

.. code-block::

    [ll/MAF: 12.440(4e+00)/0.21] 100.0%┣█████████████████████████████┫ 2.0k/2.0k [00:01<00:00, 1.4kit/s]

The setting of ``aimh_prob`` is actually the default. For less complex distributions a higher value (e.g. ``aimh_prob=0.1`` for medium-scale DSGE models) can be chosen, which accelerates burn-in.

Finally, plot the results.

.. code-block:: julia

   # analytical marginal distribution in first dimension
   x = range(-4,4,1000)
   mpdf = DIMETestFuncMarginalPDF(x, cov_scale, m, weight)
   plot(x, mpdf, label="Target", lw=2)

   # a larger sample from the initial distribution
   init = rand(MvNormal(initmean, initcov), Int(nchain*niter/4))
   histogram!(init[1,:], normalize=true, alpha=.5, label="Initialization")
   # histogram of the actual sample
   histogram!(chains[end-Int(niter/4):end,:,1][:], normalize=true, alpha=.5, label="Sample", color="black")

.. image:: https://github.com/gboehl/DIMESampler.jl/blob/main/docs/figure.png?raw=true
  :width: 800
  :alt: Sample and target distribution

To ensure propper mixing, let us also have a look at the MCMC traces. Note how chains are also switching between the two modes because of the global proposal kernel.

.. code-block:: julia
    plot(chains[:,:,1], color="cyan4", alpha=.1, legend=false, size=(900,600))

.. image:: https://github.com/gboehl/DIMESampler.jl/blob/main/docs/traces.png?raw=true
  :width: 800
  :alt: MCMC traces

While DIME is an MCMC sampler, it can straightforwardy be used as a global optimization routine. To this end, specify some broad starting region (in a non-Bayesian setup there is no prior) and let the sampler run for an extended number of iterations. Finally, assess whether the maximum value per ensemble did not change much in the last few hundered iterations. In a normal Bayesian setup, plotting the associated log-likelhood over time also helps to assess convergence to the posterior distribution.

.. code-block:: julia
    plot(lprob[:,:], color="black", alpha=.05, legend=false, size=(900,300))

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
