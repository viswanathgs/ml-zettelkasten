# Prospective Learning

- **Created**: 2025-10-27
- **Last Updated**: 2025-10-27
- **Status**: `In Progress`

---

- [ ] [2016] [deepmind] Progressive Neural Networks - [paper](https://arxiv.org/abs/1606.04671)
- [ ] [2020] [jovo] Simple Lifelong Learning Machines - [paper](https://arxiv.org/abs/2004.12908)
- [ ] [2022] [jovo] Prospective Learning: Principled Extrapolation to the Future - [paper](https://arxiv.org/abs/2201.07372)
- [ ] [2024] [jovo] Prospective Learning: Learning for a Dynamic Future - [paper](https://arxiv.org/abs/2411.00109)
- [ ] [2025] [jovo] Prospective Learning in Retrospect - [paper](https://arxiv.org/abs/2507.07965)
- [ ] [2025] [jovo] Optimal control of the future via prospective foraging - [preprint](https://drive.google.com/file/d/1JDGTq-N8JJtMDWVmbN_FGU0HKyc4Rt8y/view?usp=sharing)
- [ ] [2024] [sutton] Loss of plasticity in deep continual learning - [paper](https://www.nature.com/articles/s41586-024-07711-7)

---

## Misc Questions on Prospective Learning

- Prospective Learning
  - what makes prospective ERM a strong prospective leraning?
  - just adding time?
  - predicting the future how - CPC, JEPA relation?
  - biological grounding?
  - but what if the non-stationarity isn't predictable / there's no structure and just continually drifting?
- Why does RL not fit definition 2? it adapts to changes in environment as the policy evolves?
- Why does encoding time itself not solve the problem entirely?
- A concrete practical example of non-stationarity in a home robotics task. isaac sim, RL environments?
- RL non-stationarity
  - in the context of LLMs (RLHF, RLAIF, RLVR, etc) - engineering heuristics (eg., reward model drift), KL div loss to stay close to the original distribution: <https://chatgpt.com/share/6900e6e7-6a04-8005-ab2d-d99ffc6a81d9>
  - in the context of robotics - fauna navigation stuff, skild.ai's adaptation? <https://www.skild.ai/blogs/omni-bodied>
  
---

## [2020] [jovo] Simple Lifelong Learning Machines

- **Date**: 2025-11-24
- **Arxiv**: <https://arxiv.org/abs/2004.12908>
- **Paperpile**: <https://app.paperpile.com/view/?id=e9a7adf4-4ac3-46de-98aa-0a09593bddfd>

---

- **Abstract**:
  - > In  lifelong  learning,  data  are  used  to  improve  performance not only on the present task, but also on past and future (unencountered) tasks. While typical transfer learning algorithms can improve performance on future tasks, their performance on prior tasks degrades upon learning new tasks (called forgetting). Many  recent  approaches  for  continual  or  lifelong  learning  have attempted   to maintain performance   on   old   tasks   given   new tasks. **But striving to avoid forgetting sets the goal unnecessarily low.  The  goal  of  lifelong  learning  should  be  to  use  data  to improve  performance  on  both  future  tasks  (forward  transfer) and past tasks (backward transfer)**. In this paper, we show that a   simple   approach—representation   ensembling—demonstrates both  forward  and  backward  transfer  in  a  variety  of  simulated and benchmark data scenarios, including tabular, vision (CIFAR- 100, 5-dataset, Split Mini-Imagenet, Food1k, and CORe50), and speech (spoken digit), in contrast to various reference algorithms, which  typically  failed  to  transfer  either  forward  or  backward, or  both.  Moreover,  our  proposed  approach  can  flexibly  operate with  or  without  a  computational  budget.
- **Intro**:
  - > While  it  is  relatively easy to simultaneously optimize for multiple tasks (multi-task learning) [4], it has proven much more difficult to sequentially optimize  for  multiple  tasks.

## [2022] [jovo] Prospective Learning: Principled Extrapolation to the Future

- **Date**: 2025-10-27
- **Arxiv**: <https://arxiv.org/abs/2201.07372>
- **Paperpile**: <https://app.paperpile.com/view/?id=757687a3-71b3-4b28-9a6e-8e1376258296>

---

- **Abstract**:
  - > Learning is a process which can update decision rules, based on past experience, such that future performance improves. Traditionally, machine learning is often evaluated under the assumption that the future will be identical to the past in distribution or change adversarially. But these assumptions can be either too optimistic or pessimistic for many problems in the real world. Real world scenarios evolve over multiple spatiotemporal scales with partially predictable dynamics. Here we reformulate the learning problem to one that centers around this idea of dynamic futures that are partially learn- able.  We conjecture that certain sequences of tasks are not retrospectively learnable (in which the data distribution is fixed), but are prospectively learnable (in which distributions may be dynamic), suggesting that prospective learning is more difficult in kind than retrospective learning.  We argue that prospective learning more accurately characterizes many real world problems that (1) currently stymie existing artificial intelligence solutions and/or (2) lack adequate explanations for how nat- ural intelligences solve them.  Thus, studying prospective learning will lead to deeper insights and solutions to currently vexing challenges in both natural and artificial intelligences.
- TODO

## [2024] [jovo] Prospective Learning: Learning for a Dynamic Future

- **Date**: 2025-10-27
- **Arxiv**: <https://arxiv.org/abs/2411.00109>
- **Paperpile**: <https://app.paperpile.com/view/?id=191169a0-5c20-4ad0-ba27-355f9c2ffb97>
- **NeurIPS Poster**: <https://neurips.cc/virtual/2024/poster/94786>
- **OpenReview Discussion**: <https://openreview.net/forum?id=XEbPJUQzs3&noteId=74fm5Z4Lk6>
- **Code**: <https://github.com/neurodata/prolearn>

---

- **Abstract**:
  - > In  real-world  applications,  the  distribution  of  the  data,  and  our  goals,  evolve over time.  The prevailing theoretical framework for studying machine learning, namely probably approximately correct (PAC) learning, largely ignores time. As a consequence, existing strategies to address the dynamic nature of data and goals exhibit poor real-world performance. This paper develops a theoretical framework called “Prospective Learning” that is tailored for situations when the optimal hypothesis changes over time. In PAC learning, empirical risk minimization (ERM) is known to be consistent. We develop a learner called Prospective ERM, which returns a sequence of predictors that make predictions on future data. We prove that the risk of prospective ERM converges to the Bayes risk under certain assumptions on the stochastic process generating the data. Prospective ERM, roughly speaking, incorporates time as an input in addition to the data. We show that standard ERM as done in PAC learning, without incorporating time, can result in failure to learn when distributions are dynamic. Numerical experiments illustrate that prospective ERM can learn synthetic and visual recognition problems constructed from MNIST and CIFAR-10. Code at <https://github.com/neurodata/prolearn>.
- TODO

## [2025] [jovo] Prospective Learning in Retrospect - [paper](https://arxiv.org/abs/2507.07965)

- **Date**: 2025-10-28
- **Arxiv**: <https://arxiv.org/abs/2507.07965>
- **Paperpile**: <https://app.paperpile.com/view/?id=60c02eab-3fa7-458d-acf9-cc4b174770ec>
- **Code**: <https://github.com/neurodata/prolearn2>

---

- **Abstract**:
  - > In most real-world applications of artificial intelligence, the distributions of the data and the goals of the learners tend to change over time. The Probably Approximately Correct (PAC) learning framework, which underpins most machine learning algorithms, fails to account for dynamic data distributions and evolving objectives, often resulting in suboptimal performance. Prospective learning is a recently introduced mathematical framework that overcomes some of these limitations. We build on this framework to present preliminary results that improve the algorithm and numerical results, and extend prospective learning to sequential decision-making scenarios, specifically foraging. Code is available at: <https://github.com/neurodata/prolearn2>.
- TODO
