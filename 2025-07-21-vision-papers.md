# Vision Papers

**Created:** 2025-07-21

- [X] [2021] CLIP: Learning Transferable Visual Models From Natural Language Supervision. <https://arxiv.org/abs/2103.00020>
- [ ] [2022] OpenCLIP: Reproducible scaling laws for contrastive language-image learning. <https://arxiv.org/abs/2212.07143>
  - [ ] OpenCLIP code: <https://github.com/mlfoundations/open_clip>
- [X] [2023] MetaCLIP: Demystifying CLIP Data. <https://arxiv.org/abs/2309.16671>
- [ ] [2022] RQ-Transformer: Autoregressive Image Generation using Residual Quantization. <https://arxiv.org/abs/2203.01941>
- [ ] <https://arxiv.org/abs/2506.22355>
- [ ] Cambrian (Saining): <https://cambrian-mllm.github.io/>, <https://arxiv.org/abs/2406.16860>
- [ ] TODO vision transformer (ViT)
- [ ] SAM papers
- [ ] Flamingo: a Visual Language Model for Few-Shot Learning. <https://arxiv.org/abs/2204.14198>

## [2021] CLIP: Learning Transferable Visual Models From Natural Language Supervision

**Date:** 2025-07-21
**Arxiv:** <https://arxiv.org/abs/2103.00020>
**Paperpile:** <https://app.paperpile.com/view/?id=9ca94ee0-8932-434a-8c28-5d67ba36741d>
**Assistant:** <https://chatgpt.com/share/687ea62a-ad50-8005-8e98-9defc0dc34bc>
**CLIP Loss Implementation:** <https://colab.research.google.com/drive/1YqGGDPXh6fyAmdJXyb3163c698DOVze2?usp=sharing>

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

**Date:** 2025-07-23
**Arxiv:** <https://arxiv.org/abs/2309.16671>
**Paperpile:** <https://app.paperpile.com/view/?id=5df5730f-947e-4a5b-b3f1-dad04a0687ec>

- "We believe that the main ingredient to the success of CLIP is its data and not the model architecture or pre-training objective."
- MetaCLIP (Metadata-Curated Language-Image Pretraining) aims to reveal CLIP's data curation process.
