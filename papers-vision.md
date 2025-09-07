# Vision

- **Created**: 2025-07-21
- **Last Updated**: 2025-09-06
- **Status**: `In Progress`

---

- [X] [2021] CLIP: Learning Transferable Visual Models From Natural Language Supervision — [paper](https://arxiv.org/abs/2103.00020)
- [ ] [2022] OpenCLIP: Reproducible scaling laws for contrastive language-image learning — [paper](https://arxiv.org/abs/2212.07143)
  - [ ] OpenCLIP code: <https://github.com/mlfoundations/open_clip>
- [X] [2023] MetaCLIP: Demystifying CLIP Data — [paper](https://arxiv.org/abs/2309.16671)
- [ ] [2020] VQGAN: Taming Transformers for High-Resolution Image Synthesis - [paper](https://arxiv.org/abs/2012.09841)
- [ ] [2022] RQ-Transformer: Autoregressive Image Generation using Residual Quantization — [paper](https://arxiv.org/abs/2203.01941)
- [X] [2022] MaskGIT: Masked Generative Image Transformer — [paper](https://arxiv.org/abs/2202.04200)
- [ ] [paper](https://arxiv.org/abs/2506.22355)
- [ ] Cambrian (Saining): <https://cambrian-mllm.github.io/>, [paper](https://arxiv.org/abs/2406.16860)
- [ ] TODO vision transformer (ViT)
- [ ] SAM papers
- [ ] Flamingo: a Visual Language Model for Few-Shot Learning — [paper](https://arxiv.org/abs/2204.14198)

## [2021] CLIP: Learning Transferable Visual Models From Natural Language Supervision

- **Date**: 2025-07-21
- **Arxiv**: <https://arxiv.org/abs/2103.00020>
- **Paperpile**: <https://app.paperpile.com/view/?id=9ca94ee0-8932-434a-8c28-5d67ba36741d>
- **Assistant**: <https://chatgpt.com/share/687ea62a-ad50-8005-8e98-9defc0dc34bc>
- **CLIP Loss Implementation**: <https://colab.research.google.com/drive/1YqGGDPXh6fyAmdJXyb3163c698DOVze2?usp=sharing>

---

- (1) Intro
  - Motivation
    - Fully supervised object classification methods so far have been on fixed set of classes, not open domain.
    - Weakly supervised methods like URU (trained on instagram hashtags) showed such methods at scale to be effective pre-trained models, and improved ImageNet performance by 5% when finetuned. But still a fixed set of classes.
    - Everything so far use fixed set of classes, static softmax classifiers to perform prediction and lack a mechanism for dynamic outputs. This severely limits flexibility and limits zero-shot capabilities.
    - Fully open domain classification trained with natural language supervision has been tried in the past, but not at scale like URU and other methods focused on weak supervision but with fixed vocab. CLIP closes this gap.
  - CLIP
    - Dataset of 400 million (image, text) pairs, series of 8 models trained at spanning 2 orders of magnitude of compute. Observed that transfer performance is a smoothly predictable function of compute (scaling laws).
    - Learns to perform a wide set of tasks including OCR, geo-localization, action recognition, etc.
    - Outperforms the best available ImageNet model while also being more computationally efficient.
- (2) Approach
  - > Learning  from  natural  language  has  several  potential strengths  over  other  training  methods.   It’s  much  easier to scale natural language supervision compared to standard crowd-sourced labeling for image classification since it does not require annotations to be in a classic “machine learning compatible format” such as the canonical 1-of-N majority vote “gold label”. Instead, methods which work on natural language can learn passively from the supervision contained in the vast amount of text on the internet.  Learning from natural language also has an important advantage over most unsupervised or self-supervised learning approaches in that it doesn’t “just” learn a representation but also connects that representation to language which enables flexible zero-shot transfer. In the following subsections, we detail the specific approach we settled on.
  - > Given a batch of N (image, text) pairs, CLIP is trained to predict which of the N × N possible (image, text) pairings across a batch actually occurred. To do this, CLIP learns a multi-modal embedding space by jointly training an image encoder and text encoder to maximize the cosine similarity of the image and text embeddings of the N real pairs in the batch while minimizing the cosine similarity of the embeddings of the $N^2 - N$ incorrect pairings. We optimize a symmetric cross entropy loss over these similarity scores.
  - At test time for zero-shot classification given a dataset with target classes: image is fed through the image encoder, and the target classes through the text encoder, and the target class with the max cosine similarity between the given image and each of the target classes is chosen.
  - Psuedocode in Fig 3. Notable things:
    - L2 normalization of image and text embeddings before dot product for cosine similarity.
    - Learned temperature parameter rather than being a hyperparam. `logits = np.dot(I_e, T_e^T) * np.exp(t)`, where `I_e` is the image embedding (L2 normalized) of shape `(N, E)`, `T_e` is the text embedding (L2 normalized) of shape `(N, E)`, and `t` is the learned temperature parameter. `logits` is the final scaled pairwise cosine similarities of shape `(N, N)`.
    - Symmetric cross-entropy loss: ((cross-entropy loss of correct text for each image in batch) + (cross-entropy loss of correct image for each text in batch)) / 2
      - Basically, logits is an `(N, N)` matrix. Do softmax reduction over rows first and compute negative log-likelihood for image-to-text matching, and then softmax reduction over cols and compute negative log-likelihood for text-to-image matching, and finally take the average of the two.
  - See my CLIP loss impl in <https://colab.research.google.com/drive/1YqGGDPXh6fyAmdJXyb3163c698DOVze2?usp=sharing>
  - (3) Experiments / Analyses
    - > In computer vision, zero-shot learning usually refers to the study of generalizing to unseen object categories in image classification (Lampert et al., 2009).  We instead use the term in a broader sense and study generalization to unseen datasets.  We motivate this as a proxy for performing un- seen tasks, as aspired to in the zero-data learning paper of Larochelle et al. (2008). While much research in the field of unsupervised learning focuses on the representation learn- ing capabilities of machine learning systems, we motivate studying zero-shot transfer as a way of measuring the task- learning capabilities of machine learning systems. In this view, a dataset evaluates performance on a task on a spe- cific distribution.
      - Inspirational, considering how I felt about CV in 2018.
      - Some organizational thoughts: <https://chatgpt.com/c/687fe733-2c4c-8005-afb5-6555db88dd39>
    - > Our focus on studying zero-shot transfer as an evaluation of task learning is inspired by work demonstrating task learn- ing in the field of NLP. To our knowledge Liu et al. (2018) first identified task learning as an “unexpected side-effect” when a language model trained to generate Wikipedia ar- ticles learned to reliably transliterate names between lan- guages. While GPT-1 (Radford et al., 2018) focused on pre-training as a transfer learning method to improve supervised fine-tuning, it also included an ablation study demonstrat- ing that the performance of four heuristic zero-shot transfer methods improved steadily over the course of pre-training, without any supervised adaption. This analysis served as the basis for GPT-2 (Radford et al., 2019) which focused exclu- sively on studying the task-learning capabilities of language models via zero-shot transfer.
    - > CLIP is pre-trained to predict if an image and a text snippet are paired together in its dataset. To perform zero-shot clas- sification, we reuse this capability. For each dataset, we use the names of all the classes in the dataset as the set of poten- tial text pairings and predict the most probable (image, text) pair according to CLIP. In a bit more detail, we first compute the feature embedding of the image and the feature embed- ding of the set of possible texts by their respective encoders. The cosine similarity of these embeddings is then calculated, scaled by a temperature parameter τ, and normalized into a probability distribution via a softmax. Note that this predic- tion layer is a multinomial logistic regression classifier with L2-normalized inputs, L2-normalized weights, no bias, and temperature scaling. When interpreted this way, the image encoder is the computer vision backbone which computes a feature representation for the image and the text encoder is a hypernetwork (Ha et al., 2016) which generates the weights of a linear classifier based on the text specifying the visual concepts that the classes represent.
      - Alternative interpreation of zero-shot classification: logistic regression where the inputs the image embeddings and the weights are the text embeddings, and the text encoder becomes a hypernetwork.
    - Section 3.1.4 on prompt engineering is interesting.
    - > While zero-shot CLIP generalizes well to many natural im- age distributions as investigated in Section 3.3, we’ve ob- served that zero-shot CLIP still generalizes poorly to data that is truly out-of-distribution for it. An illustrative exam- ple occurs for the task of OCR as reported in Appendix E. CLIP learns a high quality semantic OCR representation that performs well on digitally rendered text, which is common in its pre-training dataset, as evidenced by performance on Rendered SST2. However, CLIP only achieves 88% accu- racy on the handwritten digits of MNIST. An embarrassingly simple baseline of logistic regression on raw pixels outper- forms zero-shot CLIP. Both semantic and near-duplicate nearest-neighbor retrieval verify that there are almost no im- ages that resemble MNIST digits in our pre-training dataset. This suggests CLIP does little to address the underlying problem of brittle generalization of deep learning models. Instead CLIP tries to circumvent the problem and hopes that by training on such a large and varied dataset that all data will be effectively in-distribution. This is a naive assumption that, as MNIST demonstrates, is easy to violate.
      - "CLIP tries to circumvent the problem and hopes that by training on such a large and varied dataset that all data will be effectively in-distribution".

## [2023] MetaCLIP: Demystifying CLIP Data

- **Date**: 2025-07-23
- **Arxiv**: <https://arxiv.org/abs/2309.16671>
- **Paperpile**: <https://app.paperpile.com/view/?id=5df5730f-947e-4a5b-b3f1-dad04a0687ec>

---

- "We believe that the main ingredient to the success of CLIP is its data and not the model architecture or pre-training objective."
- MetaCLIP (Metadata-Curated Language-Image Pretraining) aims to reveal CLIP's data curation process.

## [2022] MaskGIT: Masked Generative Image Transformer

- **Date**: 2025-05-09
- **Arxiv**: <https://arxiv.org/abs/2202.04200>
- **Paperpile**: <https://app.paperpile.com/view/?id=48910966-4885-4350-a09d-52e3e767c136>

---

- **Abstract**:
  - > Generative transformers have experienced rapid popularity growth in the computer vision community in synthesizing high-fidelity and high-resolution images. The best generative transformer models so far, however, still treat an image naively as a sequence of tokens, and decode an image sequentially following the raster scan ordering (i.e., line-by-line). We find this strategy neither optimal nor efficient. This paper proposes a novel image synthesis paradigm using a bidirectional transformer decoder, which we term MaskGIT. During training, MaskGIT learns to predict randomly masked tokens by attending to tokens in all directions. At inference time, the model begins with generating all tokens of an image simultaneously, and then refines the image iteratively conditioned on the previous generation. Our experiments demonstrate that MaskGIT significantly outperforms the state-of-the-art transformer model on the ImageNet dataset, and accelerates autoregressive decoding by up to 64x. Besides, we illustrate that MaskGIT can be easily extended to various image editing tasks, such as inpainting, extrapolation, and image manipulation.
- **Intro**:
  - Inspired by the successs of autoregressive models (transformer, GPT) in NLP, **generative transformer models have received growing interests in image synthesis**.
    - Generally, **autoregressive modeling for image generation is done in two stages**:
      - **Stage 1**: Vector quantize an image into a sequence of discrete tokens.
      - **Stage 2**: Train a transformer to generate the discrete tokens sequentially and autoregressively based on the previously generated tokens.
    - Stage 1 gets most of the focus while Stage 2 is a drop in replacement from NLP.
    - For Stage 2 (autoregressive modeling of discrete image tokens), **even SoTA methods treat an image naively as a flattened 1D sequence of tokens from left to right line-by-line (raster scan order)**.
      - Neither optimal nor efficient. Unlike text, images are not sequential.
  - **Masked Generative Image Transformer (MaskGIT)**:
    - **Bidirectional transformer for image synthesis**.
    - **Training (Fig 3)**: Similar to mask prediction in BERT.
    - **Inference (Fig 2)**: A novel non-autoregressive decoding method to synthesize an image in constant number of steps. At each step, all tokens are predicted in parallel but only the most confident ones are kept for the next autoregressive step, with the remaining token masked out. The mask ratio is decreased until all tokens are generated with a few steps of refinement.
      - Predicitons within each step are parallelizable.
      - Order of magnitude faster decoding.
      - For 32x32 image tokens, 8 steps with MaskGIT instead of 256 steps with raster scan order autoregressive decoding.
      - Mask ratio scheduling (i.e., fraction of tokens masked at each step) significantly affects generation quality. Propose to use cosine schedule.
    - > MaskGIT’s multidirectional nature makes it readily extendable to image manipulation tasks that are otherwise difficult for autoregressive models. Fig 1 shows a new application of class-conditional image  editing in which MaskGIT regenerates content inside the bounding box based on the given class while keeping the context (outside of the box) unchanged. This task, which is either infeasible for autoregressive model or difficult for GAN models, is trivial for our model.
- **Method**:
  - **Training (Fig 3)**:
    - Stage 1 (image tokenization) uses the same setup as in VQGAN.
    - Stage 2 (autoregressive modeling) learns a **bidirectional transformer with Masked Visual Token Modeling (MVTM)**.
      - **(1) Tokenize**: Obtain discrete tokens by feeding the image to a VQ-encoder such as in VQGAN.
      - **(2) Mask**: Sample a mask ratio $\gamma$ from 0 to 1, and uniformly select as many tokens to replace with `[MASK]`.
      - **(3) Model**: Feed through a bi-directional tranformer to optimize with negative log-likelihood loss corresponding to the masked tokens.
  - **Iterative Decoding (Fig 2)**: Start with a blank canvas with all the tokens masked out. Loop for $T$ steps:
    - **(1) Predict**: Model inference to predict output token probabilities corresponding to masked positions.
    - **(2) Sample**: At each masked position in the current step, sample an output token based on the predicted probabilities. The prediciton probability corresponding to the sampled output token is used as a confidence score, with the unmasked tokens in the current step receiving a confidence score of 1.0.
    - **(3) Mask Schedule**: Using a mask scheduling function $\gamma(r)$ with $r \in [0,1)$, compute the number of tokens to mask at the current step: $n = \lceil \gamma(\frac{t}{T})N \rceil$, where $t$ is the current decoding step count, $T$ is the total number of decoding steps, and $N$ is the total number of tokens.
    - **(4) Mask**: Mask $n$ of the least confident tokens according to the confidence score computed in (2).
  - **Masking Design**: Significantly affects the quality of image generation.
    - **Mask scheduling function** $\gamma(r)$ that computes the token mask ratio given an input $r \in [0,1]$.
      - Inference: $r = t/T$, where $T$ is the total number of decoding steps and $t \in \{0, 1, 2, ..., T-1\}$ is the current decoding step.
      - Training: $r$ is randomly sampled from $[0,1)$ to simulate various decoding scenarios.
    - **Properties of mask scheduling function**:
      - $\gamma(r)$ must be a continuous monotonically descreasing function wrt $r \in [0,1]$.
      - $\gamma(0) \to 1$ (all tokens masked out at decoding step $t=0$).
      - $\gamma(1) \to 0$ (all tokens unmasked at decoding step $t=T$).
    - **Choices of mask scheduling function (Fig 8)**:
      - Linear function
      - Concave function (less to more tokens unmasked per step) - cosine, square, cubic, exponential
      - Convex function (more to less tokens unmasked per step) - square root, logarithmic
  - **Experiments**:
    - **Metrics to measure image generation quality**: <https://chatgpt.com/share/68bce46c-3050-8005-906e-d4374a78d582>
      - **Frechet Inception Distance (FID)**
      - **Inception Score (IS)**
    - Outperforms VQGAN in quality (owing to bidirectional nature), speed (parallelism of decoding), and versatility (extends to image inpainting/outpaintaing/editing beyond image synthesis).
