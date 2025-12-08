# Reinforcement Learning

- **Created**: 2019-04
- **Last Updated**: 2023-11-28
- **Status**: `Paused`

---

- [ ] [2018] Investigating Human Priors for Playing Video Games - [paper](https://arxiv.org/abs/1802.10217)
- [ ] [2020] Atari 100K: Model-Based Reinforcement Learning for Atari - [paper](https://arxiv.org/abs/1903.00374)
- [ ] [2020] Revisiting Fundamentals of Experience Replay - [paper](https://arxiv.org/abs/2007.06700)
- [ ] [2023] Bigger, Better, Faster (BBF): Human-level Atari with human-level efficiency - [paper](https://arxiv.org/abs/2305.19452)
- [ ] [2023] [deepmind] A Definition of Continual Reinforcement Learning - [paper](https://arxiv.org/abs/2307.11046)

---

## [2023] Bigger, Better, Faster (BBF): Human-level Atari with human-level efficiency

- **Date**: 2025-11-28
- **Arxiv**: <https://arxiv.org/abs/2305.19452>
- **Paperpile**: <https://app.paperpile.com/view/?id=9cd1b87a-f170-4c80-bdc3-7b135a501947>
- **Code**: <https://github.com/google-research/google-research/tree/master/bigger_better_faster>

---

- **Abstract**:
  - > We introduce a value-based RL agent, which we call BBF, that achieves super-human performance in the Atari 100K benchmark. BBF relies on scaling the neural networks used for value estimation, as well as a number of other design choices that enable this scaling in a sample-efficient manner. We conduct extensive analyses of these design choices and provide insights for future work. We end with a discussion about updating the goal- posts for sample-efficient RL research on the ALE. We make our code and data publicly available.
- **Human-level sample efficiency on Atari**:
  - > The success of these RL methods has relied on large neural networks and an enormous number of environment samples to learn from – **a human player would require tens of thousands of years of game play to gather the same amount of experience as OpenAI Five or AlphaGo**.
  - > It is plausible that such large networks are necessary for the agent’s value estimation and/or policy to be expressive enough for the environment’s complexity, while large number of samples might be needed to gather enough experience so as to deter- mine the long-term effect of different action choices as well as train such large networks effectively. As such, **obtaining human-level sample efficiency with deep RL remains an outstanding goal**.
  - > **as RL continues to be used in increasingly challenging and sample-scarce scenarios, the need for scalable yet sample-efficient online RL methods becomes more pressing**. Despite the variability in problem characteristics making a one-size-fits-all solution unrealistic, there are many insights that may transfer across problem domains. As such, methods that achieve “state-of-the-art” performance on established benchmarks can provide guidance and insights for others wishing to integrate their techniques.
  - **BBF**:
    - Atari 100K benchmark: agents are constrained to 2 hours of gameplay to evaluate human-level efficiency. 100k steps (400k frames) at 60 FPS is 111 minutes.
    - EfficientZero: achieves human-level sample efficiency via model-based RL.
    - BBF: achieves this via model-free RL while being much more computationally efficient than EfficientZero.
- **Background - RL Axes**:
  - (1) Value-Based vs Policy-Based vs Actor-Critic (hybrid) - the "what do we learn?" axis.
  - (2) Model-Based vs Model-Free - the "do we understand the world?" axis.
  - (3) On-Policy vs Off-Policy
  - (4) Online RL (Environment interaction) vs Offline RL (Batch RL) - Offline RL is inherently off-policy, online RL can be either on-policy or off-policy.
- **Method**:
  - > The question driving this work is: **How does one scale networks for deep RL when samples are scarce?**

## 2019-04 Reading List

1. Beginner's introduction to RL and Deep Q-Learning (DQN): <https://www.intel.ai/demystifying-deep-reinforcement-learning/#gs.ac37fu>
2. Fundamentals of Policy Gradients (forms the basis of IMPALA): <http://karpathy.github.io/2016/05/31/rl>
3. John Schulman's 4-part lecture series on Policy Gradients (lectures 2 and 3 are particularly relevant): <https://www.youtube.com/@mlsscadiz4148/search?query=john%20schulman>
4. A deeper explanation of the theory and the equations underneath Policy Gradients (follows lectures 2 and 3, a very useful read-along): <https://danieltakeshi.github.io/2017/03/28/going-deeper-into-reinforcement-learning-fundamentals-of-policy-gradients/>
5. Actor-Critic Methods:
    <https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f>
    <http://rail.eecs.berkeley.edu/deeprlcourse-fa17/f17docs/lecture_5_actor_critic_pdf.pdf>
6. A high-level overview/refresher of everything above (Markov Decision Processes, Temporal-Difference Learning, DQN, various Policy Gradient algorithms, Actor-Critic Methods, etc.): <https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html>
7. IMPALA Paper: <https://arxiv.org/abs/1802.01561>
8. TorchBeast code: <https://github.com/fairinternal/torchbeast>
9. TRPO
10. PPO
11. Multi-armed bandits: <https://lilianweng.github.io/lil-log/2018/01/23/the-multi-armed-bandit-problem-and-its-solutions.html>
12. Kris Jensen's [An introduction to reinforcement learning for neuroscience, 2023](https://arxiv.org/abs/2311.07315)
