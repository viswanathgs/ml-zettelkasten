# ML Fundamentals

- **Created**: 2025-01-04
- **Last Updated**: 2025-07-31
- **Status**: `In Progress`

---

- [ ] [2014] Bahdanau, Neural machine translation by jointly learning to align and translate
- [ ] [2014] Neural Turing Machines - [paper](https://arxiv.org/abs/1410.5401)
- [ ] [2015] Attention-Based Models for Speech Recognition — [paper](https://arxiv.org/abs/1506.07503)
- [X] [2015] Neural Machine Translation of Rare Words with Subword Units — [paper](https://arxiv.org/abs/1508.07909)
- [ ] SentencePiece ([code](https://github.com/google/sentencepiece))
- [ ] [2017] Faiss: Billion-scale similarity search with GPUs — [paper](https://arxiv.org/abs/1702.08734)
- [ ] [2018] Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates — [paper](https://arxiv.org/abs/1804.10959)
- [ ] titoken
- [ ] Attention is all you need
- [ ] [2018] The Annotated Transformer — [blog](https://nlp.seas.harvard.edu/annotated-transformer/)
- [X] [2018] The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks - [paper](https://arxiv.org/abs/1803.03635)
- [ ] [2020] Efficient Transformers: A Survey — [paper](https://arxiv.org/abs/2009.06732)
- [X] [2021] RASP: Thinking Like Transformers — [paper](https://arxiv.org/abs/2106.06981)
- [ ] [2020] Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning — [paper](https://arxiv.org/abs/2012.13255)
- [ ] [2020] Nucleus Sampling: The curious case of neural text degeneration — [paper](https://arxiv.org/abs/1904.09751)
- [X] [2020] [Patrick Lewis et al] RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks — [paper](https://arxiv.org/abs/2005.11401)
- [X] [2020] [Kelvin Guu et al] REALM: Retrieval-Augmented Language Model Pre-Training — [paper](https://arxiv.org/abs/2002.08909)
- [X] [2021] LoRA: Low-Rank Adaptation of Large Language Models — [paper](https://arxiv.org/abs/2106.09685)
- [ ] QLoRA
- [ ] [2024] LoRA Learns Less and Forgets Less — [paper](https://arxiv.org/abs/2405.09673)
- [ ] [2024] The Faiss library — [paper](https://arxiv.org/abs/2401.08281)
- [ ] BERT
- [ ] RLHF
- [ ] FSDP
- [ ] MoE
- [ ] TODO gemini long context blog/report
- [ ] TODO RoPE
- [ ] TODO GQA — [paper](https://arxiv.org/abs/2305.13245v1)
- [ ] TODO Gumbel-Softmax

---

## [2015] Neural Machine Translation of Rare Words with Subword Units

- **Date**: 2025-03-26
- **Arxiv**: <https://arxiv.org/abs/1508.07909>
- **Paperpile**: <https://app.paperpile.com/view/?id=c465c277-f3b0-4c79-b05c-a66dcc4dea35>
- **Code**: <https://github.com/rsennrich/subword-nmt>

---

- Neural Machine Translation (NMT) is an open vocabulary problem. Previously, fixed word-level vocabulary was used with back-off to dictionary lookup for out-of-vocabulary (OOV) words. This doesn't work well, often not scalable (eg., compound words in Deutsch), isn't fully learned/e2e due to dictionary fallback, etc.
- **Goal**: Open-vocabulary NMT without dictionary back-off. Solution: Subword units.
- **Contributions**:
  - (1) Demonstrating subword-level NMT can work, doesn't require any dictionary fallback, is simpler and effective.
  - (2) Adapting Byte Pair Encoding (BPE) for subword tokenization.
- Competing tradeoff between vocabulary size and text size (aka token length). Want to minimize both, but minimizing one increases the other:
  - One extreme: characater-level tokenization. Small vocab, but very long token length (text size) reduces efficiency and increases the distances over which neural models need to pass info.
  - Other extreme: word-level tokenization. Small token length / text size, but very large vocab and still can't account for all words (rare words) and will need to fallback to dictionary translation for OOV words.
  - A balanced approach: Short lists of unsegmented words with subword-units for rare words.
  - BPE (Byte Pair Encoding): The above intermediate approach is manual. Alternatively, BPE allows learning a vocab that strikes this balance and provides a good compression rate.
- **Byte Pair Encoding (BPE)**: Also see [minbpe](https://github.com/karpathy/minbpe) in [[karpathy-curriculum.md]]
  - Iteratively replaces the most frequent pair of bytes in a sequence with a single, unused byte.
  - Applying this to word segmentation by merging characters or character sequences rather than bytes.
  - Ref: Algorithm 1
  - Start with symbol vocabulary as the character vocabularly, iteratively replace each occurrence of the most frequent symbol pair (A, B) with a new symbol AB. Pairs crossing word-boundaries not counted.
  - Each merge op produces a new symbol which represents a characater n-gram. Frequent n-grams (or whole words) are eventually merged into a single symbol.
  - Final symbol vocab size = initial symbol vocab size + number of merge ops.
  - The number of merge ops is the only hyperparam to control.
- **Two methods of applying BPE for Machine Translation**:
  - **Independent BPE**: Separate tokenization for source and target vocabularies. Compact text and vocab size.
  - **Joint BPE**: A single tokenization combining the source and target vocabularies. More consistent source and target segmentations (eg., avoids segmenting the same name differently in the source and target languages, which would make it a harder translation problem to learn).

## [2021] RASP: Thinking Like Transformers

- **Date**: 2025-01-28
- **Arxiv**: <https://arxiv.org/abs/2106.06981>
- **Paperpile**: <https://app.paperpile.com/view/?id=6e497a3e-fba3-4796-8f08-56786390797b>

---

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

- **Date**: 2025-03-03
- **Arxiv**: <https://arxiv.org/abs/2106.09685>
- **Paperpile**: <https://app.paperpile.com/view/?id=a1e6be67-62b8-41df-a1c3-10e62df070af>
- **Code**: <https://github.com/microsoft/LoRA>

---

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

- **Date**: 2025-07-07
- **Arxiv**: <https://arxiv.org/abs/2005.11401>
- **Paperpile**: <https://app.paperpile.com/view/?id=f89e65a4-6bb1-4d08-a3db-dc84384d5cd6>

---

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

- **Date**: 2025-07-31
- **Arxiv**: <https://arxiv.org/abs/2002.08909>
- **Paperpile**: <https://app.paperpile.com/view/?id=438b99d9-7110-4170-8ab8-911a99e9b9ee>

---

- Same vein as RAG, but see differences here: <https://chatgpt.com/share/688ce137-6e80-8005-9945-9d3d5df68202>. The main thing with REALM is that retrieval is part of the pretraining objective, not just during downstream fine-tuning or inference.
- > The key intuition of REALM is to train the retriever us- ing a performance-based signal from unsupervised text: a retrieval that improves the language model’s perplex- ity is helpful and should be rewarded, while an un- informative retrieval should be penalized.   For exam- ple, in Figure 1, if the model needs to fill the blank in “the at the top of the pyramid”, the re- triever should be rewarded for selecting a document con- taining “The pyramidion on top allows for less material higher up the pyramid”. We achieve this behavior by modeling our retrieve-then-predict approach as a latent variable language model and optimizing the marginal likelihood.
- Mostly straightforward, the main thing is the complexity of doing this during pre-training where the document embedding index needs to be updated as the whole thing is trained e2e including the document embedding model $p(z | x)$. But because things like FAISS are only used to select the top-k documents, even though the document embedding model params are updated after each step, the embedding index is only updated every few steps since the top-k documents more or less still will be the same. But once the top k documents are retrieved, the marginalization is still done with the latest document embedding model to compute $p(z | x)$.
  - See "Implementing asynchronous MIPS refreshes" in the paper.
- Very well written and very accessible paper.

## [2017] Faiss: Billion-scale similarity search with GPUs

- **Date**: 2025-07-08
- **Arxiv**: <https://arxiv.org/abs/1702.08734>
- **Paperpile**: <https://app.paperpile.com/view/?id=88cdd2ad-c73e-4072-93b5-7478cb2f3076>

---

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

## [2018] The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks

- **Date**: 2025-10-24
- **Arxiv**: <https://arxiv.org/abs/1803.03635>
- **Paperpile**: <https://app.paperpile.com/view/?id=54b808a1-e559-4046-8c8b-f24a26e58144>
- **Assistant**: <https://chatgpt.com/share/68fcea86-f42c-8005-a232-37da8d46b74a>

---

- **Abstract**:
  - > Neural network pruning techniques can reduce the parameter counts of trained networks by over 90%, decreasing storage requirements and improving computational performance of inference without compromising accuracy. However, contemporary experience is that the sparse architectures produced by pruning are difficult to train from the start, which would similarly improve training performance.
  - > We find that a standard pruning technique naturally uncovers subnetworks whose initializations made them capable of training effectively. Based on these results, we articulate the **lottery ticket hypothesis: dense, randomly-initialized, feed-forward networks contain subnetworks (winning tickets) that—when trained in isolation— reach test accuracy comparable to the original network in a similar number of iterations.  The winning tickets we find have won the initialization lottery: their connections have initial weights that make training particularly effective**.
  - We present an algorithm to identify winning tickets and a series of experiments that support the lottery ticket hypothesis and the importance of these fortuitous initializations. We consistently find winning tickets that are less than 10-20% of the size of several fully-connected and convolutional feed-forward architectures for MNIST and CIFAR10. Above this size, the winning tickets that we find learn faster than the original network and reach higher test accuracy.
- **Motivation**:
  - Typically, pruning works by training a larger network, then prune, then fine-tune the unpruned weights. This works and creates a smaller network that performs just as well as the larger one.
  - But, **if the smaller network can perform just as well after pruning, why can’t we train it directly from scratch?**
  - May be it's not the smaller networks can't train, but that we don't know which ones to train.
- **Lottery Ticket Hypothesis**:
  - > **A randomly-initialized, dense neural network contains a subnetwork that is initialized such that—when trained in isolation—it can match the test accuracy of the original network after training for at most the same number of iterations.**
  - Formally,
    - Dense feed-forward neural network $f(x; \theta_0)$ with initial parameters $\theta_0$.
    - Train a sparser network $f(x; m \odot \theta_0)$ with binary mask $m \in \{0, 1\}^{\lVert \theta_0 \rVert}$.
    - There's a very sparse $m$ such that the same accuracy can be obtained with at most the same number of training iterations.
  - These trainable subnetworks, winning tickets, can be found by standard unstructured pruning techniques.
  - They have **won the initialization lottery with a combination of weights and connections capable of learning**.
  - When the parameters of the winning tickets are randomly re-initialized ($f(x; m \odot \theta_0')$) as opposed to retaining the original random initialization prior to training and pruning ($f(x; m \odot \theta_0)$), they don't train effectively. Therefore, initialization matters.
    - Up to moderate pruning (~80% unstructured sparsity), random reinit still works fine. But at extreme pruning (>98% sparsity), only the original init works.
    - > One possible explanation for this behavior is these initial weights are close to their final values after training—that in the most extreme case, they are already trained. However, experiments in Appendix F show the opposite—that **the winning ticket weights move further than other weights**. This suggests that the **benefit of the initialization is connected to the optimization algorithm, dataset, and model**. For example, the winning ticket initialization might land in a region of the loss landscape that is particularly amenable to optimization by the chosen optimization algorithm.
- **Identifying winning tickets**:
  - Train a network normally as prune its smallest-magnitude weights. The remaning unpruned connections constitute the architecture of the winning ticket, and their weights are reset to its initialization prior to the original training.
  - **One-shot pruning**:
    - (1) Randomly init a neural net $f(x; \theta_0)$.
    - (2) Train for $j$ iters, arriving at params $\theta_j$.
    - (3) Prune $p\%$ of the smallest-magnitude params in $\theta_j$, creating a mask $m$.
    - (4) Reset the remaining params to their values in $\theta_0$, creating the winning ticket $f(x; m \odot \theta_0)$.
  - **Iterative pruning**:
    - Instead of pruning $p\%$ of weights at once, repeat the above steps over $n$ rounds, pruning $p^{\frac{1}{n}}\%$ of the weights that survive the previous round.
    - Leads to smaller networks compared to one-shot pruning while matching the accuracy of the original network.
    - > On deeper networks (Resnet-18 and VGG-19), iterative pruning is unable to find winning tickets unless we train the networks with learning rate warmup.
- **Lottery Ticket Conjecture**:
  - **Untested conjecture: SGD seeks out and trains a subset of well-initialized weights**.
  - > Dense, randomly-initialized networks are easier to train than the sparse networks that result from pruning because there are more possible subnetworks from which training might recover a winning ticket.
- **Implications**:
  - (1) **Improve training performance**: Training schemes that search for winning tickets and prune as early as possible?
  - (2) **Design better networks**: New/sparse architectures and initialization schemes conducive to learning?
  - (3) **Improve theoretical understanding of neural networks**: Why do randomly-initialized feed-forward networks seem to contain winning tickets?
