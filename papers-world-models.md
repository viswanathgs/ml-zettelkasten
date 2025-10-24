# World Models

- **Created**: 2025-10-22
- **Last Updated**: 2025-10-22
- **Status**: `Not Started`

---

- [ ] [2018] [hardmaru] World Models - [paper](https://arxiv.org/abs/1803.10122)
- [ ] [2022] [rockt] General Intelligence Requires Rethinking Exploration - [paper](https://arxiv.org/abs/2211.07819)
- [ ] [2024] [rockt] Genie: Generative Interactive Environments - [paper](https://arxiv.org/abs/2402.15391)
- [ ] [2024] [rockt] Genie 2: A large-scale foundation world model - [blog](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/)
- [ ] [2025] [rockt] Genie 3: A new frontier for world models - [blog](https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/)

---

## [2024] [rockt] Genie: Generative Interactive Environments

- **Date**: 2025-10-22
- **Arxiv**: <https://arxiv.org/abs/2402.15391>
- **Paperpile**: <https://app.paperpile.com/view/?id=d369b9a2-4115-4a93-b839-1617a20d31ec>
- **Blog**: <https://sites.google.com/view/genie-2024/home>

---

- **Abstract**:
  - > We introduce Genie, the first generative interactive environment trained in an unsupervised manner from unlabelled Internet videos. The model can be prompted to generate an endless variety of action- controllable virtual worlds described through text, synthetic images, photographs, and even sketches. At 11B parameters, Genie can be considered a foundation world model. It is comprised of a spatiotemporal video tokenizer, an autoregressive dynamics model, and a simple and scalable latent action model. **Genie enables users to act in the generated environments on a frame-by-frame basis despite training without any ground-truth action labels** or other domain-specific requirements typically found in the world model literature. Further the resulting learned latent action space facilitates training agents to imitate behaviors from unseen videos, opening the path for training generalist agents of the future.
  - > **Learning to control without action labels**: What makes Genie unique is its ability to learn fine-grained controls exclusively from Internet videos. This is a challenge because Internet videos do not typically have labels regarding which action is being performed, or even which part of the image should be controlled. Remarkably, Genie learns not only which parts of an observation are generally controllable, but also infers diverse latent actions that are consistent across the generated environments.
- TODO
