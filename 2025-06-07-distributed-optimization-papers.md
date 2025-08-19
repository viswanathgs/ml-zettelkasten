# Distributed Optimization Papers

**Created**: 2025-07-06

- [ ] Historical papers: google async, baidu, hogwild
- [ ] <https://arxiv.org/abs/2110.08133>
- [ ] RL papers
  - [ ] IMPALA
  - [ ] GALA paper (Mido Assran)
  - [ ] TODO
- [ ] Marc'Aurelio Ranzato and Arthur Szlam's recent papers
  - [X] [2023] Diloco: Distributed low-communication training of language models
  - [ ] [2024] Asynchronous Local-SGD Training for Language Modeling
  - [ ] [2024] DiPaCo: Distributed Path Composition
  - [ ] [2024] Streaming DiLoCo with overlapping communication: Towards a Distributed Free Lunch
  - [ ] [2025] Communication-Efficient Language Model Training Scales Reliably and Robustly: Scaling Laws for DiLoCo
- [ ] Blog/notebook writeup

## Blog Sketch

- <https://chatgpt.com/share/686a9f87-ceac-8005-a2f8-48b165b2d77b>
- History
  - Async SGD google
  - Baidu sync SGD
  - Hogwild!
- RL related
  - A3C, IMPALA, or SEED RL
  - torchbeast
- Federated Learning related
  - Read the "4.1. Local SGD and Federated Learning" section of DiLoCo paper <https://app.paperpile.com/view/?id=40f86e2b-3728-415b-8d66-a35bc35a60dc>
- Newer methods
  - DiLoCo, Gshard

## [2023] Diloco: Distributed low-communication training of language models

**Date:** 2025-07-06
**Arxiv:** <https://arxiv.org/abs/2311.08105>
**Paperpile:** <https://app.paperpile.com/view/?id=40f86e2b-3728-415b-8d66-a35bc35a60dc>

- Abstract/Intro
  - Fully sync SGD on ultra-large clusters is insanely challenging infrastructurally: a single rank failing can halt training or lead to numerical issues, poorly leverages heterogenous devices. Difficult to colocate and tightly synchronize a large number of accelerators.
  - As opposed to a single very large cluster for training LLMs, train on islands of poorly connected clusters.
  - Variant of federated averaging, where the number of inner steps is large, the inner optimizer is AdamW, and the outer optimizer is Nesterov momentum.
  - Inspired from Federated Learning
    - k workers, each operating on their own island of devices, consuming a certain data partition, and updating a model replica. Exchange gradients once in a while (every H steps) to get their replica back in sync globally.
    - Reduces colocation need by a factor of k, reduces communication needs across islands, each island can be of a different device type.
    - Relies on FedOpt and FedAvg algorithms from Federated Learning: <https://chatgpt.com/share/686abdbb-e8fc-8005-aaab-bcb5553441b9>
- DiLoCo
  - The algorithm is basically FedOpt from <https://arxiv.org/pdf/2003.00295>, but applied to LLM training instead of Federated Learning.
    - FedAvg simply averages the weights of each worker models. FedOpt instead computes gradients as the delta between the current global param and the averaged worker params, and updates params using an optimizer given this gradient.
    - Inner optimizer is AdamW (standard in LLM training) and outer optimizer is Nesterov momentum (empirically works better).
    - If the number of outer steps (T) is 1, then this reduces to model souping.
    - If the number of inner steps (H) is 1, then this is just fully synchronous SGD.
  - 8 workers are used.
- Limitations
  - Diminishing returns beyond 8 workers.
  - Still synchronous - all workers need to communicate. This means heterogenity of hardware, imbalanced data splits, etc., will lead to suboptimality due to comms.

## [2024] Asynchronous Local-SGD Training for Language Modeling

**Date:** 2025-07-06
**Arxiv:** <https://arxiv.org/abs/2401.09135>
**Paperpile:** <https://app.paperpile.com/view/?id=5809d175-43b1-4788-8bed-6666abaa7688>
