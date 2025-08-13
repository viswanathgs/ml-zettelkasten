# SGD Optimization Algorithms

## Fundamentals

- [ ] Blog: [Overview of various SGD optimization algorithms](https://ruder.io/optimizing-gradient-descent/)
- [X] [Hinton lecture slides](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) covering momentum, Nesterov, adaptive optimizers, RMSProp with intuition and general bag of tricks.
  - Large gradients doesn't necessarily mean it's the right direction towards global optimum.
  - Goal:
    - Move quickly in directions with small but consistent gradients
    - move slowly in directions with big bug inconsistent gradients
  - Global LR (SGD, SGD with momentum, Nesterov) vs per-param adaptive LR (AdaGrad, RMSProp, Adam)
  - Momentum: Instead of using the gradient to change the position, use it to change the velocity
    - Builds up speed in directions with gentle by consistent gradients
    - Damps oscillations in directions of high curvature (large but inconsistent gradients)
    - Start with a small momentum factor to reduce noise, and smoothly increase as things start to converge to accelerate training
  - Adaptive methods: Adapt the LR for each param based on the consistency of the gradient for that param
    - Intuition: the magnitude of the gradients are often very different for different params and keeps changing, choosing a single global LR that works well is hard
  - RMSProp: Divide the LR for each param by a running mean of magnitudes of the recent gradients for that param.
    - The idea is just to use the sign of the gradient to make the update. But directly using the sign in a stochastic mini-batch setting is problematic because the magnitude of the gradients is important too (eg., 9 consecutive updates of +0.1 and then one update of -0.9 should cancel out; this won't happen if we just take the sign).
    - RMSProp solves this problem by keeping a running mean estimate (with decay) of the magnitude of the gradients for each param, and then normalizing the incoming gradient with that estimate. This essentially gives us the sign of the gradient when averaged over several mini-batches while also avoiding the above-mentioned pathology.
    - But what does this solve? By essentially using only the sign of the gradient, we are punting the problem of having to choose a global LR that works well for a variety of gradient magnitudes (across params and across timesteps).
- [ ] Original Adam paper: [Adam: A method for stochastic optimization](https://arxiv.org/abs/1412.6980)

## Weight Decay

- [ ] Paper on why weight decay != L2 reg for Adam: [Decoupled weight decay regularization](https://arxiv.org/abs/1711.05101)
  - SGD with L2 reg == SGD with weight decay (following a simple scaling factor)
  - Adam with L2 reg != Adam with weight decay
  - In the Adam with L2 reg case, For params with large gradients, Adam's scaling factor also downweights the regularization component. Thus not all params get regularized equally. But with decoupled weight decay, this is not a problem.
  - This makes Adam with (decoupled) weight decay perform better than Adam with L2 reg.
  - Coupled together with cosine annealing LR scheduler, things become even better empirically.
  - Section 3 of the paper has a Bayesian filtering based theoritical justification for why decoupled weight decay performs better for adaptive gradient methods.
- [ ] Paper: [Three mechanisms of weight decay regularization](https://arxiv.org/abs/1810.12281)
