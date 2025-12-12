# Reinforcement Learning

- **Created**: 2019-04
- **Last Updated**: 2023-12-12
- **Status**: `In Progress`

---

- [ ] [2000] [Andrew Ng] Algorithms for Inverse Reinforcement Learning - [paper](https://ai.stanford.edu/~ang/papers/icml00-irl.pdf)
- [ ] [2013] Guided Policy Search - [paper](https://graphics.stanford.edu/projects/gpspaper/gps_full.pdf)
- [ ] [2014] [David Silver] DPG: Deterministic Policy Gradient Algorithms - [paper](https://proceedings.mlr.press/v32/silver14.pdf)
- [ ] [2015] [Tim Lillicrap, David Silver] Continuous control with deep reinforcement learning - [paper](https://arxiv.org/abs/1509.02971)
- [ ] [2018] [Ben Recht] A Tour of Reinforcement Learning: The View from Continuous Control - [paper](https://arxiv.org/abs/1806.09460)
- [ ] [2018] Investigating Human Priors for Playing Video Games - [paper](https://arxiv.org/abs/1802.10217)
- [ ] [2020] Atari 100K: Model-Based Reinforcement Learning for Atari - [paper](https://arxiv.org/abs/1903.00374)
- [ ] [2020] Revisiting Fundamentals of Experience Replay - [paper](https://arxiv.org/abs/2007.06700)
- [ ] [2023] Bigger, Better, Faster (BBF): Human-level Atari with human-level efficiency - [paper](https://arxiv.org/abs/2305.19452)
- [ ] [2023] [deepmind] A Definition of Continual Reinforcement Learning - [paper](https://arxiv.org/abs/2307.11046)
- [ ] [2025] Kevin Murphy RL book: <https://arxiv.org/abs/2412.05265>

---

## [2018] [Ben Recht] A Tour of Reinforcement Learning: The View from Continuous Control - [paper](https://arxiv.org/abs/1806.09460)

- **Date**: 2025-12-10
- **Arxiv**: <https://arxiv.org/abs/1806.09460>
- **Paperpile**: <https://app.paperpile.com/view/?id=e49dc43d-6eb4-4832-8751-443e6964d352>

---

- **Intro**:
  - > This survey aims to provide a language for the control and reinforcement learning communities to begin communicating, highlighting what each can learn from the other.  Controls is the theory of designing complex actions from well-specified models, while reinforcement learning often makes intricate, model-free predictions from data alone.  Yet both RL and control aim to design systems that  use  richly  structured  perception,  perform  planning  and  control  that  adequately  adapt  to environmental changes, and exploit safeguards when surprised by a new scenario.
  - > I try  to  put  RL  and  control  techniques  on  the  same  footing  through  a  case  study  of  the linear quadratic regulator (LQR) with unknown dynamics.  This baseline will illuminate the var- ious  trade-offs  associated  with  techniques  from  RL  and  control.
  - > “model-free”  methods  popular  in  deep  reinforcement  learning  are  considerably  less effective in both theory and practice than simple model-based schemes when applied to LQR. Per- haps surprisingly, I also show cases where these observations continue to hold on more challenging nonlinear applications.  I then argue that model-free and model-based perspectives can be unified, combining their relative merits.
- **RL - optimal control when the dynamics are unknown**:
  - > find a sequence of inputs that drives a dynamical system to maximize some objective beginning with minimal knowledge of how the system responds to inputs.
  - > Since the dynamics are stochastic, the optimal control problem typically allows a controller to observe the state before deciding upon the next action [12].  This allows a controller to continually mitigate uncertainty through feedback.  Hence, rather than optimizing over deterministic sequences of actions $a_t$, we instead optimize over policies. A control policy (or simply “a policy”) is a function, $\pi$, that takes a trajectory from a dynamical system and outputs a new control action.  Note that $\pi$ gets access only to previous states and control actions.
  - > we can’t solve this optimization problem using standard optimization methods unless we know the state transidion dynamics.  We must learn something about the dynamical system and subsequently choose the best policy based on our knowledge.
  - > The main paradigm in contemporary RL is to play the following game.  We decide on a policy $\pi$ and horizon length $L$. Then we pass this policy either to a simulation engine or to a real physical system and are returned a trajectory $\tau_L$ and a sequence of rewards. We want to find a policy that maximizes the reward with the fewest total number of samples computed by the oracle, and we are allowed to do whatever we’d like with the previously observed trajectories and reward information when computing a new policy.  If we were to run $m$ queries with horizon length $L$, we would pay a total cost of $mL$.  However, we are free to vary our horizon length for each experiment. This is our oracle model and is called **episodic reinforcement learning** (See, for example Chapter 3 of Sutton and Barto [76], Chapter 2 of Puterman [58], or Dann and Brunskill [24]).  We want the expected reward to be high for our derived policy, but we also need the number of oracle queries to be small.
  - > Do we decide an algorithm is best if it crosses some reward threshold in the fewest number of samples?  Or is it best if it achieves the highest reward given a fixed budget of samples?  Or maybe there’s a middle ground?
- **RL vs supervised learning**:
  - > A key distinguishing aspect of RL is the control action $a$.  Unlike in prediction,  the practitioner can vary $a$, which has implications both for learning (e.g., designing experiments to learn about a given system) and for control (e.g., choosing inputs to maximize reward).
  - > There is a precarious trade-off that must be carefully considered:  reinforcement learning demands interventions with the promise that these actions will directly lead to valuable returns, but the resulting complicated feedback loops are hard to study in theory, and failures can have catastrophic consequences.
- **RL Strategies**:
  - **(1) Model-based RL**:
    - fits a model of the state transitions to best match observed trajectories, and uses this to approximate the solution to the RL problem.
  - **(2) Model-free RL**:
    - eschews the need for the system's model, directly seeking a map from observations to actions.
    - > The term “model-free” almost always means “no model of the state transition function” when casually claimed in reinforcement learning research.  However, this does not mean that modeling is not heavily built into the assumptions of model-free RL algorithms.
    - **(a) Approximate Dynamic Programming / Value Based**:
      - uses Bellman’s principle of optimality to approximate the RL problem using previously observed data.
      - > Also troubling is the fact that we had to introduce the discount factor in order to get a simple Bellman equation.  One can avoid discount factors,  but this requires considerably more sophisticated analysis.  Large discount factors do in practice lead to brittle methods, and the discount becomes a hyperparameter that must be tuned to stabilize performance.
    - **(b) Policy Search / Policy Based**:
      - directly searches for policies by using data from previous episodes in order to improve the reward.
      - `REINFORCE` algorithm and log-likelihood trick.
  - > The main question is which of these approaches makes the best use of samples and how quickly do the derived policies converge to optimality.
- > **This survey has focused on “episodic” reinforcement learning and has steered clear of a much harder problem:  adaptive control.  In the adaptive setting, we want to learn the policy online.  We only get one trajectory.  The goal is, after a few steps, to have a model whose reward from here to eternity will be large.  This is very different, and much harder that what people are doing in RL. In episodic RL, you get endless access to a simulator.  In adaptive control, you get one go.**
- > as soon as a  machine  learning  system  is  unleashed  in  feedback  with  humans,  that  system  is  a  reinforcement learning system.

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
