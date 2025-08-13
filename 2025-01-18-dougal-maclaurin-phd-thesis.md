# Dougal MacLaurin's PhD Thesis

**Start:** 2024-01-18
**End:** TODO

[Modeling, Inference and Optimization with Composable Differentiable Procedures](https://dougalmaclaurin.com/phd-thesis.pdf)

**Paperpile:** <https://app.paperpile.com/view/?id=30a08235-6959-4715-9117-4de7cf17f6db>

## Chapter 1: Intro

Five somewhat disparate topics:

1. Firefly Monte Carlo - MCMC for large datasets by querying a subset of data per iteration
2. Autograd for Python - Precursor to Jax
3. Convolutional Networks on Graphs
4. Hyperparam optimization via autograd
5. Early stopping framed as variational inference

## Chapter 2: Background

**Bayesian Inference:** Data $D$, params $\theta$. Estimating $\theta$ as $p(\theta|D) = \frac{p(D, \theta)}{p(D)} = \frac{p(D | \theta)p(\theta)}{p(D)}$. Inferring the model parameters is just a matter of conditioning on the data to obtain the posterior distribution $p(\theta | D)$.

But integrations such as $P(D) = \int_{}{} p(D,\theta) d\theta$ are intractable. Therefore, need to resort to **posterior approximation**.

**Three approaches to posterior approximation:**

1. Point estimation - approximation by a single parameter value.
2. Variational inference - Approximate the intractable distribution with a tractable one.
3. Markov Chain Monte Carlo (MCMC) - Approximation by samples generated from a Markov chain.

**Notations and Terminology:**

- $\tilde{p}(x)$: Unnormalized density / distribution
- $p(x) = \tilde{p}(x) / Z$: Normalized density / distribution
- $Z$: Normalization constant. Called marginal likelyhood in bayesian modeling, and partition function in physics. Typically requires an intractable integration.
- $log(D, \theta)$: Unnormalized log posterior or unnormalized log joint density. In physics, called (negative) energy.

**Backpropagation:** Differentiation with reverse mode accumulation.

Given a function $R^D \to R$ composed of a set of primitive functions $R^M \to R^N$ with known Jacobians, the gradient of the composition is given by the product of the Jacobians of the primitive functions per the chain rule. But the chain rule doesn't precribe the order of multiplications, which makes all the difference from a computational complexity perspective.

Assume function $f: R^M \to R^N$. Jacobian of f is $R^{N \times M}$. JVP and VJP are higher-order functions where Jacobian of f doesn't actually get materialized:

- JVP(f, x) - Jacobian Vector Product. Here, $x \in R^M$. Forward mode differentiation.
- VJP(f, x) - Vector Jacobian Product. Here, $x \in R^N$. Reverse mode differentiation.
