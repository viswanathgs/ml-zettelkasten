# Diffusion Models

- **Created**: 2025-08-19
- **Last Updated**: 2025-08-19
- **Status**: `Not Started`

---

- [ ] [2015] Deep Unsupervised Learning using Nonequilibrium Thermodynamics. <https://arxiv.org/abs/1503.03585>
- [ ] [2020] Denoising Diffusion Probabilistic Models. <https://arxiv.org/abs/2006.11239>
- [ ] [2022] The Annotated Diffusion Model. <https://huggingface.co/blog/annotated-diffusion>
- [ ] TODO

## [2022] The Annotated Diffusion Model

- **Date**: 2025-03-05
- **Blog**: <https://huggingface.co/blog/annotated-diffusion>

---

- Two processes
  - Forward diffusion process: sample an image from the true distribution and gradually add gausian noise for $T$ steps until it's eventually pure noise / isotropic gaussian.
  - Reverse denoising diffusion process: neural net trained to gradually denoise an image starting from pure noise to an eventual image in the distribution.
- Forward diffusion process: $q(x_t | x_{t - 1})$. $x_0$ is the actual image and $x_T$ is pure noise.
  - At each step $t$, sample from a conditional gaussian distrubution with mean $\sqrt{1 - \beta_t}x_{t-1}$ and variance $\beta_tI$.
  - This can be done by sampling $\epsilon$ noise from the standard gaussian (0 mean, unit variance) and setting $x_t = \sqrt{1 - \beta_t}x_{t - 1} + \beta_t\epsilon$.
  - $\beta_t$ values change aross time steps following a "variance schedule" (can be linear, quadratic, cosine, etc), kinda like learning rate schedule.
- Backward denoising diffusion process:
  - In the forward diffusion process, starting with an actual sample $x_0$, if we set the schedule appropriately, we end up with pure gaussian noise at $x_T$.
- TODO
