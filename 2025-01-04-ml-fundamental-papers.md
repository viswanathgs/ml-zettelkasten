# ML Fundamental Papers

**Start:** 2025-01-04
**End:** TODO

- [ ] [2014] Bahdanau, Neural machine translation by jointly learning to align and translate
- [ ] [2015] Attention-Based Models for Speech Recognition. <https://arxiv.org/abs/1506.07503>
- [X] [2016] Neural Machine Translation of Rare Words with Subword Units. <https://arxiv.org/abs/1508.07909>
- [ ] [2017] Faiss: Billion-scale similarity search with GPUs. <https://arxiv.org/abs/1702.08734>
- [ ] [2018] Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates <https://arxiv.org/abs/1804.10959>
- [ ] Tokenizer: Byte-pair encoding (BPE, <https://arxiv.org/abs/1508.07909>) and SentencePiece (<https://github.com/google/sentencepiece>)
- [ ] titoken
- [ ] Attention is all you need
- [ ] [2018] The Annotated Transformer. <https://nlp.seas.harvard.edu/annotated-transformer/>
- [ ] [2020] Efficient Transformers: A Survey. <https://arxiv.org/abs/2009.06732>
- [X] [2021] RASP: Thinking Like Transformers. <https://arxiv.org/abs/2106.06981>
- [ ] [2020] Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning. <https://arxiv.org/abs/2012.13255>
- [ ] [2020] Nucleus Sampling: The curious case of neural text degeneration. <https://arxiv.org/abs/1904.09751>
- [X] [2020] [Patrick Lewis et al] RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. <https://arxiv.org/abs/2005.11401>
- [X] [2020] [Kelvin Guu et al] REALM: Retrieval-Augmented Language Model Pre-Training. <https://arxiv.org/abs/2002.08909>
- [X] [2021] LoRA: Low-Rank Adaptation of Large Language Models. <https://arxiv.org/abs/2106.09685>
- [ ] TODO QLoRA
- [ ] [2024] LoRA Learns Less and Forgets Less. <https://arxiv.org/abs/2405.09673>
- [ ] [2024] The Faiss library. <https://arxiv.org/abs/2401.08281>
- [ ] BERT
- [ ] RLHF
- [ ] FSDP
- [ ] MoE
- [ ] TODO gemini long context blog/report
- [ ] TODO llama post-training internal primer <https://docs.google.com/document/d/1JajJ_5gwOFND7-cHh9pJnpuQ6BrDclh5dvFSB7N1UdE/edit?usp=sharing>
- [ ] TODO RoPE
- [ ] TODO GQA <https://arxiv.org/abs/2305.13245v1>
- [ ] TODO Gumbel-Softmax trick: <https://sassafras13.github.io/GumbelSoftmax/>

## [2016] Neural Machine Translation of Rare Words with Subword Units

**Date:** 2025-03-26
**Arxiv:** <https://arxiv.org/abs/1508.07909>
**Paperpile:** <https://app.paperpile.com/view/?id=c465c277-f3b0-4c79-b05c-a66dcc4dea35>
**Code:** <https://github.com/rsennrich/subword-nmt>

- Neural Machine Translation (NMT) is an open vocabulary problem. Previously, fixed word-level vocabulary was used with back-off to dictionary lookup for out-of-vocabulary (OOV) words. This doesn't work well, often not scalable (eg., compound words in Deutsch), isn't fully learned/e2e due to dictionary fallback, etc.
- Goal: Open-vocabulary NMT without dictionary back-off. Solution: Subword units.
- Contributions:
  - (1) Demonstrating subword-level NMT can work, doesn't require any dictionary fallback, is simpler and effective.
  - (2) Adapting Byte Pair Encoding (BPE) for subword tokenization.
- Competing tradeoff between vocabulary size and text size (aka token length). Want to minimize both, but minimizing one increases the other:
  - One extreme: characater-level tokenization. Small vocab, but very long token length (text size) reduces efficiency and increases the distances over which neural models need to pass info.
  - Other extreme: word-level tokenization. Small token length / text size, but very large vocab and still can't account for all words (rare words) and will need to fallback to dictionary translation for OOV words.
  - A balanced approach: Short lists of unsegmented words with subword-units for rare words.
  - BPE (Byte Pair Encoding): The above intermediate approach is manual. Alternatively, BPE allows learning a vocab that strikes this balance and provides a good compression rate.
- Byte Pair Encoding (BPE):
  - Iteratively replaces the most frequent pair of bytes in a sequence with a single, unused byte.
  - Applying this to word segmentation by merging characters or character sequences rather than bytes.
  - Ref: Algorithm 1
  - Start with symbol vocabulary as the character vocabularly, iteratively replace each occurrence of the most frequent symbol pair (A, B) with a new symbol AB. Pairs crossing word-boundaries not counted.
  - Each merge op produces a new symbol which represents a characater n-gram. Frequent n-grams (or whole words) are eventually merged into a single symbol.
  - Final symbol vocab size = initial symbol vocab size + number of merge ops.
  - The number of merge ops is the only hyperparam to control.
- Two methods of applying BPE:
  - Independent BPE: Separate tokenization for source and target vocabularies. Compact text and vocab size.
  - Joint BPE: A single tokenization combining the source and target vocabularies. More consistent source and target segmentations (eg., avoids segmenting the same name differently in the source and target languages, which would make it a harder translation problem to learn)

## [2021] RASP: Thinking Like Transformers

**Date:** 2025-01-28
**Arxiv:** <https://arxiv.org/abs/2106.06981>
**Paperpile:** <https://app.paperpile.com/view/?id=6e497a3e-fba3-4796-8f08-56786390797b>

- RNNs have a parallel in finite state automata, what is the equivalent for transformer?
- RASP Language: Restricted Access Sequence Processing Language
- RASP Computation Model:
  - Inputs: $tokens$ of length $n$, $indices$ of length $n$
  - Output: another sequence
  - Operations over sequences:
    - element-wise ops
    - $select$ op: returns a boolean matrix $S$ of shape $n \times n$. Simlar to attention matrix, but unweighted.
    - $aggregate$ op: takes a selection matrix $S$ of shape $n \times n$, an input sequence of length $n$, and outputs a sequence of length $n$ in which each position in the output is a combination of the positions in the input according $S$. Similar to attention operation.
      - $aggregate$ is the only non-elementwise / cross-positional operation. That is, the only way to combine values from different positions or move values from one position to another.
  - RASP programs are functional. Lazy, currying, etc.
  - RASP language is defined in terms of s-ops (sequence ops, aka functions) and selectors, not sequences and matrices.
- Overall helps understand theoretically if a certain problem can be solved via a transformer.

## [2021] LoRA: Low-Rank Adaptation of Large Language Models

**Date:** 2025-03-03
**Arxiv:** <https://arxiv.org/abs/2106.09685>
**Paperpile:** <https://app.paperpile.com/view/?id=a1e6be67-62b8-41df-a1c3-10e62df070af>
**Code:** <https://github.com/microsoft/LoRA>

- Abstract
  - Naive finetuning of a pretrained transformer is expensive.
  - LoRA: freeze pretrained model weights but inject trainable rank decomposition matrices into each transformer layer.
    - Less params to finetune (10000x compared to GPT-3) while performing on par as full-finetuning.
    - No additional inference latency.
- Intro
  - Premise: Inspired by works such as <https://arxiv.org/abs/2012.13255> which show that learned overparametrized models in fact reside in intrinsic low dimension.
  - Hypothesis: The change in weights during model adaptation also has a low intrinsic rank.
  - Approach: Pretrained dense weights of a layer kept frozen and insteaad indirectly adapted by optimizing its rank decomposition matrices.
    - Pretrained dense weights $W \in R^{d \times d}$ for large $d$. Instead of finetuning $W$ (expensive), learn low-rank matrices $A \in R^{r \times d}$ and $B \in R^{d \times r}$ such that $d >> r$. Final adapted weight is $W + BA$, and thus no additional inference cost.
  - Several additional benefits:
    - Reduced storage cost means many task-specific LoRA modules can be built and composed easily.
    - Lower training cost, don't need to calculate grads or maintain optimizer states for most params.
    - No additional inference latency compared to full-finetuning.
    - Natural generalization to full finetuning as $r$ approaches $d$.
- Training
  - Transformer has 4 weight matrices in the self-attention module (query, key, value and output projections) and two in the MLP module.
  - MLP weights are frozen, just the attention weights are adapted by LoRA for simplicity and efficiency.
  - Bias: Can either finetunine biases for just the LoRA modules or finetune all biases in the model.
  - Initialization: Random Gaussian for $A$, and zero for $B$, so $BA$ is zero at the beginning of training.
    - That said, LoRA github repo uses Kaiming Uniform init for $A$ for `nn.Linear`: <https://github.com/microsoft/LoRA/blob/main/loralib/layers.py#L119-L125>
  - Update: Low-rank $r$, a fixed hyperparameter $\alpha$ that doesn't vary during training. Updated calculated as $W + \frac{\alpha}{r}BA$.

## [2020] RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

**Date:** 2025-07-07
**Arxiv:** <https://arxiv.org/abs/2005.11401>
**Paperpile:** <https://app.paperpile.com/view/?id=f89e65a4-6bb1-4d08-a3db-dc84384d5cd6>

- End-to-end differentiable retriever and generator. Retriever encodes query $x$ and uses maximum inner product search (MIPS) to select top-K documents $z_i$. The generator uses the original query $x$, retrieved document latent variable $z$, and previously generated tokens to generate the next token.
- Two methods:
  - (1) RAG-Sequence Model: A single document $z$ is used to generated all tokens in $y$. Marginalized over the top-K retrieved documents. For document $z_i$, product over probs of generated tokens $y_i$. Marginalized sum over each such $P(y | x, z_i)$.
  - (2) RAG-Token Model: Essentially product and sum are flipped. Different document $z$ for each token and marginalize accordingly.
  - For a target sequence length of 1, both methods reduce to the same formulation.
- Training: document retriever is pretrained and fixed, only the query encoder and the generator are updated.
- Test-time decoding:
  - (2) RAG-Token Model: Standard beam decoding
  - (1) RAG-Sequence Model: Needs one beam search per document $z_i$. The details can be tricky, refer to the paper.

## [2020] REALM: Retrieval-Augmented Language Model Pre-Training

**Date:** 2025-07-31
**Arxiv:** <https://arxiv.org/abs/2002.08909>
**Paperpile:** <https://app.paperpile.com/view/?id=438b99d9-7110-4170-8ab8-911a99e9b9ee>

- Same vein as RAG, but see differences here: <https://chatgpt.com/share/688ce137-6e80-8005-9945-9d3d5df68202>. The main thing with REALM is that retrieval is part of the pretraining objective, not just during downstream fine-tuning or inference.
- > The key intuition of REALM is to train the retriever us- ing a performance-based signal from unsupervised text: a retrieval that improves the language model’s perplex- ity is helpful and should be rewarded, while an un- informative retrieval should be penalized.   For exam- ple, in Figure 1, if the model needs to fill the blank in “the at the top of the pyramid”, the re- triever should be rewarded for selecting a document con- taining “The pyramidion on top allows for less material higher up the pyramid”. We achieve this behavior by modeling our retrieve-then-predict approach as a latent variable language model and optimizing the marginal likelihood.
- Mostly straightforward, the main thing is the complexity of doing this during pre-training where the document embedding index needs to be updated as the whole thing is trained e2e including the document embedding model $p(z | x)$. But because things like FAISS are only used to select the top-k documents, even though the document embedding model params are updated after each step, the embedding index is only updated every few steps since the top-k documents more or less still will be the same. But once the top k documents are retrieved, the marginalization is still done with the latest document embedding model to compute $p(z | x)$.
  - See "Implementing asynchronous MIPS refreshes" in the paper.
- Very well written and very accessible paper.

## [2017] Faiss: Billion-scale similarity search with GPUs

**Date:** 2025-07-08
**Arxiv:** <https://arxiv.org/abs/1702.08734>
**Paperpile:** <https://app.paperpile.com/view/?id=88cdd2ad-c73e-4072-93b5-7478cb2f3076>

- k-NN embedding search on large databases.
  - Insight: "accepting a minimal accuracy loss results in orders of magnitude of compression"
  - Vector Compression methods can be categorized into: (1) binary codes, (2) quantization methods. For both, searching neighbors doesn't require reconstructing the vectors.
  - Faiss: Product Quantization (PQ) codes
- IVFADC - Inverted File with Asymmetric Distance Computation
  - Two levels of quantization. Given a vector $y$, it's approximated as $y ~= q(y) = q_1(y) + q_2(y - q_1(y))$
  - $q_1$ is a first-level coarse quantizer, $q_2$ is a second-level fine quantizer for the residual. Values are encoded using their indices.
  - Distance is computed between query $x$ and $q(y)$.
  - Non-exhaustive search given only the top scores (least L2 distance) first-level / coarsely quantized centroids are retained.
- First-level / coarse quantizer
  - Typically sqrt(total database size) first level centroids.
  - Trained using k-means.
- Second-level / fine quantizer
  - Product Quantization
  - TODO
