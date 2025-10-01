# Neuro AI

- **Created**: 2025-10-01
- **Last Updated**: 2025-10-01
- **Status**: `In Progress`

---

- [ ] <https://www.biorxiv.org/content/10.1101/055624v1>
- [ ] <https://arxiv.org/abs/1911.09451>
- [ ] <https://www.nature.com/articles/s41593-019-0520-2>
- [ ] <https://www.thetransmitter.org/neuroai/accepting-the-bitter-lesson-and-embracing-the-brains-complexity/>
- [ ] [2020] A natural lottery ticket winner: Reinforcement learning with ordinary neural circuits
- [ ] [2021] Can a fruit fly learn word embeddings? <https://arxiv.org/abs/2101.06887>
- [ ] [2023] Incorporating neuro-inspired adaptability for continual learning in artificial intelligence.
- [ ] [2023] Blake Richards: The study of plasticity has always been about gradients. <https://physoc.onlinelibrary.wiley.com/doi/full/10.1113/JP282747>
- [ ] [2023] The connectome of an insect brain - [paper](https://www.biorxiv.org/content/10.1101/2022.11.28.516756v1)
- [ ] [2025] [Joshua Vogelstein, JHU] Biological Processing Units (BPU): Leveraging an Insect Connectome to Pioneer Biofidelic Neural Architectures - [paper](https://arxiv.org/abs/2507.10951)

---

## Terminology

- Axons: specialized for sending signals (presynaptic terminals)
- Dendrites: specialized for receiving signals (postsynaptic sites)
- Edges aren't symmetric:
  - A -> B means neuron A’s axon terminal synapses onto neuron B’s dendrite.
  - The opposite is a different connection and may or may not exist.

## [2025] [Joshua Vogelstein, JHU] Biological Processing Units: Leveraging an Insect Connectome to Pioneer Biofidelic Neural Architectures

- **Date**: 2025-10-01
- **Arxiv**: <https://arxiv.org/abs/2507.10951>
- **Paperpile**: <https://app.paperpile.com/view/?id=5b87463f-3b9d-4ac5-9e37-a62466dd7243>

---

- **Abstract**:
  - > The complete connectome of the Drosophila larva brain of- fers a unique opportunity to investigate whether biologically evolved circuits can support artificial intelligence. We convert this wiring dia- gram into a Biological Processing Unit (BPU)—a fixed recurrent net- work derived directly from synaptic connectivity. Despite its modest size (3,000 neurons and 65,000 weights between them), the unmodified BPU achieves 98% accuracy on MNIST and 58% on CIFAR-10, surpassing size-matched MLPs. Scaling the BPU via structured connectome expan- sions further improves CIFAR-10 performance, while modality-specific ablations reveal the uneven contributions of different sensory subsystems. On the ChessBench dataset, a lightweight GNN-BPU model trained on only 10,000 games achieves 60% move accuracy, nearly 10x better than any size transformer. Moreover, CNN-BPU models with ∼ 2M param- eters outperform parameter-matched Transformers, and with a depth-6 minimax search at inference, reach 91.7% accuracy, exceeding even a 9M- parameter Transformer baseline. These results demonstrate the potential of biofidelic neural architectures to support complex cognitive tasks and motivate scaling to larger and more intelligent connectomes in future work.
- **Intro**
  - <https://www.biorxiv.org/content/10.1101/2022.11.28.516756v1> mapped **an entire Drosophila larval connectome of 3k neurons and 65k weights** between them.
    - Opportunity to examine a fully nature-optimized neural net.
    - Drosophila can achieve complex behaviors with minimal resources in contrast to large-scale AI models.
    - > This suggests that **a complete biological connectome may serve as a biological lottery ticket** [2, 3]: a compact, evolutionarily selected circuit capable of supporting a broad range of cognitive functions.
    - > With the full larval connectome now available, we hypothesize that a fully intact biological neural circuit can inform the design of efficient and generalizable artificial systems, as it embodies solutions to many of the same computational challenges neural networks aim to address.
    - > To test this, **we directly employ the complete connectome without altering its structure or synaptic weights, assessing whether it can support diverse cognitive tasks without task-specific adaptation**.
  - Leverage the complete Drosophila larval connectome to develop Biological Processing Units (BPU).
    - Evalauted on two categories of tasks:
      - (1) sensory processing (MNIST, CIFAR-10)
      - (2) decision making (chess)
    - > By including peripheral sensors alongside the central BPU circuit, we test whether the BPU can support generalized cognition under realistic biological constraints.
    - > Finally, to understand how far this advantage can scale, we introduce a directed, signed degree–corrected Stochastic Block Model (DCSBM) that lets us expand the larval connectome up to 5x while faithfully preserving its block-level wiring statistics and synaptic polarity.
