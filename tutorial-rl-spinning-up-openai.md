# [Spinning Up in RL, OpenAI](https://spinningup.openai.com/)

- **Created**: 2025-12-13
- **Last Updated**: 2025-12-13
- **Status**: `In Progress`

---

- [[papers-rl.md]]

---

- [X] Overview: <https://spinningup.openai.com/en/latest/user/algorithms.html>
- [X] Key Concepts in RL: <https://spinningup.openai.com/en/latest/spinningup/rl_intro.html>
- [ ] Kinds of RL Algorithms: <https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html>
- [ ] Intro to Policy Optimization: <https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html>

---

## [RL Algorithms Covered](https://spinningup.openai.com/en/latest/user/algorithms.html)

1. Vanilla Policy Gradients (VPG)
2. Trust Region Policy Optimization (TRPO)
3. Proximal Policy Optimization (PPO)
4. Deep Deterministic Policy Gradient (DDPG)
5. Twin Delayed DDPG (TD3)
6. Soft Actor-Critic (SAC)

## [Key Concepts in RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)

- **State $s$**: Complete description of the world/environment. There is no information about the world which is hidden from the state.
- **Observation $o$**: What the agent observes about the state.
  - **(a) Fully Observed**: The agent can observe the state completely.
  - **(b) Partially Observed**: The agent can only see a parial observation of the state.
- **Action Space**: Set of all valid actions in a given environment.
  - **(a) Discrete action space**
  - **(b) Continuous action space**
- **Policy $\pi$**: Rule used by an agent to decide what actions to take.
  - **(a) Deterministic Policy** $a_t = \mu_\theta(s_t)$
  - **(b) Stochastic Policy**: $a_t \sim \pi_\theta( \cdot | s_t)$
    - **Categorical Policy**: For discrete action spaces. Outputs a categorical distribution over actions to sample from.
    - **Diagonal Gaussian Policy**: For continuous actions spaces. "Diagonal" in the sense that covariance matrix of the multivariate gaussian distribution is a diagonal matrix. Mean actions, $\mu_\theta(s)$, is output by a neural net. Log std of the actions, $\log \sigma_\theta(s)$, is either output by the neural net or are standalone params. Action is sampled from the gaussian distribution with this mean and std $a \sim \mathcal{N}(\mu_\theta(s), \sigma_\theta(s))$, implemented as $a = \mu_{\theta}(s) + \sigma_{\theta}(s) \odot z$ where $z \sim \mathcal{N}(0, I)$.
- **Trajectory/Rollout/Episode $\tau$**: Sequence of states and actions in the world: $\tau = (s_0, a_0, s_1, a_1, ...)$.
  - Initial state $s_0 \sim \rho_0(.)$.
  - State transitions depend only the previous state and action.
  - State transitions can either be deterministic ($s_{t+1} = f(s_t, a_t)$) or stochastic ($s_{t+1} \sim P(\cdot|s_t, a_t)$) depending on the environment.
- **Reward Function $R$**: $r_t = R(s_t, a_t, s_{t+1})$, although frequently simplified to $r_t = R(s_t, a_t)$.
- **Return $R(\tau)$**: The goal of the agent is to maximize some notion of cumulative reward $R(\tau)$ over a trajectory $\tau$.
  - **Finite-horizon undiscounted return**: $R(\tau) = \sum_{t=0}^T r_t$
  - **Infinite-horizon discounted return**: $R(\tau) = \sum_{t=0}^{\infty} \gamma^t r_t$, where $\gamma \in (0,1)$ is the discount factor.
- **The RL Problem**: Select a policy which **maximizes expected return** when the agent acts according to it.
  - **Probability of a $T$-step trajectory** $P(\tau|\pi) = \rho_0 (s_0) \prod_{t=0}^{T-1} \pi(a_t | s_t) P(s_{t+1} | s_t, a_t)$.
  - **Expected return** $J(\pi) = \underset{\tau\sim \pi}{\mathbb{E}}[R(\tau)] = \int_{\tau} P(\tau|\pi) R(\tau)$.
  - **Optimal policy** $\pi^* = \arg \max_{\pi} J(\pi)$. This is the central optimization problem in RL.
- **On-Policy**: Old data not used -> weaker sample efficiency, but more stability.
  - Algorithms: VPG, TRPO, PPO.
- **Off-Policy**: Reuses old data by exploiting Bellman’s equations for optimality -> sample efficient, but can be unstable.
  - Algorithms: Q-learning, DDPG, TD3, SAC.
  - > But problematically, there are no guarantees that doing a good job of satisfying Bellman’s equations leads to having great policy performance. Empirically one can get great performance—and when it happens, the sample efficiency is wonderful—but the absence of guarantees makes algorithms in this class potentially brittle and unstable. TD3 and SAC are descendants of DDPG which make use of a variety of insights to mitigate these issues.
- **Value Function**: The expected return if you start in that state or state-action pair, and then act according to a particular policy forever after.
  - **(a) On-Policy Action-Value Function**: $Q^{\pi}(s,a) = \underset{\tau \sim \pi}{\mathbb{E}}[R(\tau) | s_0 = s, a_0 = a]$. The expected return if you start in state $s$, take an arbitrary action $a$ (which may not have come from the policy), and then always act according to policy $\pi$.
  - **(b) On-Policy Value Function**: $V^{\pi}(s) = \underset{\tau \sim \pi}{\mathbb{E}}[R(\tau) | s_0 = s] = \underset{a \sim \pi}{\mathbb{E}}[Q^{\pi}(s,a)]$. The expected return if you start in state $s$ and always act according to policy $\pi$.
  - **(c) Optimal Action-Value Function**: $Q^*(s,a) = \max_\pi Q^{\pi}(s,a)$. The expected return if you start in state $s$, take an arbitrary action $a$ (which may not have come from the policy), and then always act according to the optimal policy.
  - **(d) Optimal Value Function**: $V^*(s) = \max_\pi V^{\pi}(s) = \max_a Q^*(s,a)$. The expected return if you start in state $s$ and always act according to the optimal policy.
- **Advantage Function**: $A^{\pi}(s,a) = Q^{\pi}(s,a) - V^{\pi}(s)$. Captures the **relative advantage of an action**.
  - Describes how much better it is to take a specific action $a$ in state $s$, over randomly sampling an action according to the distribution $\pi(\cdot|s)$, assuming you act according to $\pi$ forever after.
  - We don't always need to know how good an action is in an absolute sense, but only how much better it is than others on average.
- **Optimal Action**: $a^*(s) = \arg \max_a Q^* (s,a)$.
- **Bellman Equations**: Basic idea - the value of your starting point is the reward you expect to get from being there, plus the value of wherever you land next.
  - **Bellman Equations for On-Policy Value Functions**:
    - $Q^{\pi}(s,a) = \underset{s' \sim P}{\mathbb{E}}[{r(s,a) + \gamma \underset{a'\sim \pi}{\mathbb{E}}[{Q^{\pi}(s',a')}]}]$
    - $V^{\pi}(s) = \underset{a \sim \pi; s'\sim P}{\mathbb{E}}[{r(s,a) + \gamma V^{\pi}(s')}]$
  - **Bellman Equations for Optimal Value Functions**:
    - $Q^*(s,a) = \underset{s'\sim P}{\mathbb{E}}[{r(s,a) + \gamma \max_{a'} Q^*(s',a')}]$
    - $V^*(s) = \max_a \underset{s'\sim P}{\mathbb{E}}[{r(s,a) + \gamma V^*(s')}]$
- **Markov Decision Processes (MDP)**:
  - System obeys the Markov property - transitions only depend on the most recent state and action, and no prior history.
  - $\langle S, A, R, P, \rho_0 \rangle$, where
    - $S$ is the set of all valid states,
    - $A$ is the set of all valid actions,
    - $R : S \times A \times S \to \mathbb{R}$ is the reward function, with $r_t = R(s_t, a_t, s_{t+1})$,
    - $P : S \times A \to \mathcal{P}(S)$ is the transition probability function, with $P(s_{t+1}|s_t,a_t)$ being the probability of transitioning into state $s_{t+1} if you start in state $s_t$ and take action $a_t$,
    - $\rho_0$ is the starting state distribution, that is, $s_0 \sim \rho_0(\cdot)$.
