# Speech/Audio

- **Created**: 2025-07-16
- **Last Updated**: 2025-09-05
- **Status**: `In Progress`

---

- [X] [2019] wav2vec: Unsupervised Pre-training for Speech Recognition - [paper](https://arxiv.org/abs/1904.05862)
- [X] [2019] vq-wav2vec: Self-Supervised Learning of Discrete Speech Representations - [paper](https://arxiv.org/abs/1910.05453)
- [X] [2020] wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations - [paper](https://arxiv.org/abs/2006.11477)
- [X] [2021] HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units - [paper](https://arxiv.org/abs/2106.07447)
- [X] [2022] Whisper: Robust Speech Recognition via Large-Scale Weak Supervision - [paper](https://arxiv.org/abs/2212.04356)
- [X] [2021] SoundStream: An End-to-End Neural Audio Codec - [paper](https://arxiv.org/abs/2107.03312)
- [X] [2022] EnCodec: High Fidelity Neural Audio Compression - [paper](https://arxiv.org/abs/2210.13438)
- [X] [2022] AudioLM: a Language Modeling Approach to Audio Generation - [paper](https://arxiv.org/abs/2209.03143)
- [ ] [2023] SoundStorm: Efficient Parallel Audio Generation - [paper](https://arxiv.org/abs/2305.09636)
- [X] [2023] AudioPaLM: A Large Language Model That Can Speak and Listen - [paper](https://arxiv.org/abs/2306.12925)
- [X] [2023] MusicGen: Simple and Controllable Music Generation - [paper](https://arxiv.org/abs/2306.05284)
- [X] [2023] Speech-Llama: Prompting Large Language Models with Speech Recognition Abilities - [paper](https://arxiv.org/abs/2307.11795)
- [ ] [2024] Moshi: a speech-text foundation model for real-time dialogue - [paper](https://arxiv.org/abs/2410.00037)
- [X] [2025] Sesame AI Conversational Speech Model - [blog](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice)

## [2019] wav2vec: Unsupervised Pre-training for Speech Recognition

- **Date**: 2025-08-13
- **Arxiv**: <https://arxiv.org/abs/1904.05862>
- **Paperpile**: <https://app.paperpile.com/view/?id=6d1013f6-b029-4f6d-aaa3-4c264e08d651>

---

- **Intro**:
  - Unsupervised pretraining of raw audio to improve supervised speech recognition.
  - Pretrain a simple multilayer convnet optimized via a **noise contrastive binary classification task on large amounts of unlabeled raw audio**.
  - The pretrained representations can then be used as inputs to train an ASR model, as opposed to using log-mel filterbank features.
- **Model**:
  - **Encoder network (causal conv)**: Takes raw audio samples $x$ and produces latent representations $z$
  - **Context network (causal conv)**: For a receptive field size $v$, takes multiple latent representations $(z_t, ..., z_{t-v})$ and outputs a single contextualized vector $c_t$.
- **Self-supervised contrastive pretraining objective**:
  - Train the model to **have the context vector $c_t$ distinguish a latent representation $k$ steps in the future $z_{t+k}$ from distractor/negative samples**.
  - For a step size $k$,
    - **Loss for positive sample** $L_k^{pos} = \log \sigma(z_{t+k}h_k(c_t))$, where $h_k(c_t)$ is an affine transformation.
    - **Loss for negative sample** $L_k^{neg} = \log \sigma(-z'h_k(c_t))$, where $z'$ is the latent representation of the negative sample.
    - **Total contrastive loss** for step size $k$ is $L_k = \sum_{t=1}^{T-k} L_k^{pos} + \frac{\lambda}{T} L_k^{neg}$, where $\lambda$ is the number of negative samples and $T$ is the total sequence length.
  - $L_k$ for various step sizes $k$ are computed and summed.
- **Downstream use - ASR using wav2vec pretrained representations**:
  - How are the pretrained representations used to improve supervised ASR performance?
    - **ASR models are trained with wav2vec's context representations as inputs instead of using log-mel filterbank features**.
    - Notably, **the pretrained wav2vec model is not used as a checkpoint to finetune from**.
    - wav2vec + ASR aren't trained end-to-end. wav2vec is first trained and frozen, and then its representations are used as inputs to train a CTC-style ASR model.
  - **ASR Decoding**: Maximize $\log p_{\text{CTC}}(Y \mid C) + \alpha \log P_{\text{LM}}(Y) + \beta |Y|$, where $p_{\text{LM}}$ is a language model and the last term is the insertion bonus.
  - Outperforms the best character-based ASR model at that time (DeepSpeech 2) using two orders of magnitude less labeled training data.
- **Lineage**: <https://chatgpt.com/share/68a3908b-d03c-8005-b91e-5f2571b2acc0>
  - **(1) wav2vec**: Raw audio $X$ to latent representation $Z$ (encoder network $f \colon X \to Z$) and then to context embeddings $C$ (context network $g \colon Z \to C$). The model is trained to have $c_t$ distinguish latent representations $k$ steps in the future $z_{t+k}$ from negative latent representations $z'$ using a contrastive loss.

## [2019] vq-wav2vec: Self-Supervised Learning of Discrete Speech Representations

- **Date**: 2025-08-14
- **Arxiv**: <https://arxiv.org/abs/1910.05453>
- **Paperpile**: <https://app.paperpile.com/view/?id=c209ea0e-0adf-4d98-9821-fdb14c3c39bf>

---

- **Intro**:
  - vq-wav2vec (Vector Quantized wav2vec): wav2vec, but with discrete audio tokens rather than continuous embeddings.
  - Uses either Gumbel-softmax or online k-means clustering (similar to VQ-VAE) to quantize dense audio representations.
  - Discretization enables direct application of methods from NLP.
- **Discretized speech training pipeline**:
  - **(1) vq-wav2vec model** (Fig 1):
    - In original wav2vec, we went from raw audio $X$ to latent representation $Z$ (encoder network $f \colon X \to Z$) and then to context embeddings $C$ (context network $g \colon Z \to C$). The model is trained to have $c_t$ distinguish latent representations $k$ steps in the future $z_{t+k}$ from negative latent representations $z'$ using a contrastive loss.
    - In vq-wav2vec, we insert a vector quantization module $q$ between $f$ and $g$ such that $f \colon X \to Z$, $q \colon Z \to \tilde{Z}$, and $g \colon \tilde{Z} \to C$. $q$ builds discrete representations and $\tilde{Z}$ is the quantized representation reconstruction.
    - The quantization module replaces the original representation $z$ by $\tilde{z} = e_i$ from a fixed-size codebook $e \in \mathbb{R}^{V \times d}$, where $V$ is the codebook size and $d$ is the embedding dim (as in `nn.Embedding`).
    - Loss: Same as wav2vec, except that the context network output $c_t$ predicts the future quantized latent representations $\tilde{z}_{t+k}$ rather than continuous encoder output $z_{t+k}$.
  - **(2) BERT on discrete audio tokens**: The discretization step allows a direct drop-in of algorithms from NLP which are built around discrete inputs. BERT encoder is trained on the discretized unlabeled audio tokens.
  - **(3) ASR using BERT representations**: Acoustic models are trained on labeled speech data using BERT representations as inputs instead of log-mel spectrogram features.
- **Two approaches to Vector Quantization**:
  - **(a) Gumbel-Softmax**: <https://chatgpt.com/share/68a5e923-6d40-8005-adf1-a1ab20a7ad4e>
    - Gumbel-Softmax to sample discrete codebook entries in a fully-differentiable manner.
    - **Training**:
      - Linear projection added to the dense latent representation $z$ to produce logits $l \in \mathbb{R}^V$ for Gumbel-Softmax to sample from the $V$ codebook entries.
      - Probability for choosing the $j$-th codebook entry is $p_j = \frac{\exp(l_j + g_j) / \tau}{\sum_{v=1}^V \exp(l_v + g_v) / \tau}$, where $g = -\log(-\log(u))$ is the Gumbel noise, $u \sim U(0,1)$ are uniform samples, and $\tau$ is the temperature parameter to approximate argmax as in Gumbel-max.
      - The added Gumbel noise helps sample discrete values from the distribution $l$ while making the process differentiable thanks to the reparametrization trick. The temperature parameter $\tau$ helps approximate the non-differentiable argmax op in Gumbel-max.
      - During forward pass, pick the codebook entry $\tilde{z}$ corresponding to the largest $p$. This non-differentiable step is still needed as we don't want a weighted sum of codebook entries, but strictly want to select a single codebook entry.
      - During backward pass, the usage of straight-through estimator (STE) enables gradient flow from the chosen $\tilde{z}$ to the distribution $p$. The actual implementation can be formulated as going from $p$ to argmax one-hot, and then applying STE to bypass the non-differentiable argmax op$:
        - $p_{\text{onehot}} = \text{onehot}(\text{argmax}_j p_j)$
        - $p_{\text{onehot}} = (p_{\text{onehot}} - p).\text{detach}() + p$
        - $\tilde{z} = \text{codebook}[p_{\text{onehot}}]$
    - **Inference**:
      - We simply pick the codebook entry corresponding to the largest index in $l$.
      - No Gumbel-noise is added during inference as we don't have to sample and we don't need gradients.
  - **(b) K-means**: <https://chatgpt.com/share/689f46dc-7978-8005-94d7-285099e82814>
    - Quantization module $q$ is simply replacing $z$ by a $\tilde{z} = e_i$ from the codebook that's closest in terms of L2 distance.
    - Unlike in Gumbel-Softmax, the above codebook lookup step is not differentiable anymore.
    - **Loss $L_{vq-wav2vec}^{kmeans} = L_{wav2vec} + L_{codebook} + L_{commitment}$**
      - **wav2vec loss**: $L_{wav2vec}$ is the same as wav2vec loss except that the context network predicts the quantized latent $\tilde{Z}$, and straight-through estimator (STE) is used to backprop all the way.
      - **Codebook and commitment losses**: $L_{codebook} + L_{commitment} = \|z.\text{detach}() - \tilde{z}\|^2 + \|z - \tilde{z}.\text{detach}()\|^2$.
        - The first term (codebook loss) updates the chosen codebook entry $\tilde{z}$ to be closer to the frozen encoder representation $z$ (frozen due to stop-gradient).
        - The second term (commitment loss) updates the encoder output $z$ towards the chosen (and frozen) codebook vector $\tilde{z}$, forcing the encoder to "commit" to a codebook entry.
        - This separation is more stable and avoids collapse rather than having a single $\|z - \tilde{z}\|^2$ term: <https://chatgpt.com/c/689e8361-ac38-8329-91b9-3d958b282100>
    - **Note**: The above loss updates codebook using gradient descent, but subsequent works like SoundStream (below) switch to EMA (exponential moving average) updates for improved stability and don’t rely on gradient flow through discrete assignments which can be noisy.
    - **Gumbel-Softmax vs K-means approaches to quantization**: <https://chatgpt.com/share/68a39c2c-db50-8005-b153-150bc2eaa58f>
  - Product Quantization (PQ) can be applied to the codebook to improve performance. That is, instead of replacing the latent representation $z \in \mathbb{R}^d$ by a single codebook entry $e_i \in \mathbb{R}^{V \times d}$ where $V$ is the number of codebook entries, $z$ can be organized into $G$ groups such that $z \in \mathbb{R}^{G \times (d/G)}$, with $G$ codebooks of size $V \times (d/G)$ each.
- **Lineage**: <https://chatgpt.com/share/68a3908b-d03c-8005-b91e-5f2571b2acc0>
  - **(1) wav2vec**: Raw audio $X$ to latent representation $Z$ (encoder network $f \colon X \to Z$) and then to context embeddings $C$ (context network $g \colon Z \to C$). The model is trained to have $c_t$ distinguish latent representations $k$ steps in the future $z_{t+k}$ from negative latent representations $z'$ using a contrastive loss.
  - **(2) wav2vec -> vq-wav2vec**: Add discretization, make speech look like language tokens, apply BERT-style masked LM.

## [2020] wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations

- **Date**: 2025-08-15
- **Arxiv**: <https://arxiv.org/abs/2006.11477>
- **Paperpile**: <https://app.paperpile.com/view/?id=05c0051a-46ed-48fa-9fe6-e6d05ba821ef>

---

- **Intro**:
  - Collapse the two-stage pipeline of vq-wav2vec (wav2vec producing discrete audio tokens, and then using those audio tokens to train a BERT-style masked LM) into one model.
  - Instead of separately training BERT on discrete audio tokens, integrate a transformer over masked latent features and predict quantized units directly (with the whole network being trained e2e).
  - Downstream use on ASR by finetuning rather than using pretrained representations as inputs to train a separate ASR model.
  - On Librispeech, outperforms previous SoTA with 100x less labeled data.
- **Model**: (Fig 1)
  - **(1) Encoder network (temporal conv, $f \colon X \to Z$)**: Takes raw audio $X$ and outputs latent representation $Z$.
  - **(2) Context network (masked transformer, $g \colon Z \to C$)**: Takes latent representation $Z$ with spans masked randomly, and outputs context embedding $C$.
    - The **inputs to the context network are continuous latent embeddings, with the quantized latents only used as targets in the contrastive loss**. This is different from vq-wav2vec where the context network takes quantized latent reconstructions as inputs.
    - Ablations over four combinations (inputs_to_tranformer={continuous,quantized} x contrastive_targets={continuous,quantized}) show continuous inputs and quantized targets to have the lowest WER after finetuning on Librispeech.
    - This makes sense, as continuous latent representations retain more information to enable better context representations, and quantizing the target representations leads to more robust training.
  - **(3) Quantization module ($q \colon Z \to Q$)**: Continuous latent representation $Z$ to quantized targets $Q$ for contrastive loss.
    - Gumbel-Softmax approach from vq-wav2vec.
    - Product Quantization (PQ) with $G$ groups. Product  quantization  amounts  to  choosing  quantized  representations  from multiple codebooks and concatenating them.
- **Objective**: Loss $L = L_{contrastive} + \alpha L_{codebook-diversity}$
  - **Contrastive loss**: $L_{contrastive} = -\log \frac{\exp(\text{sim}(c_t, q_t) / \tau)}{\sum_{k=1}^{K+1} \exp(\text{sim}(c_t, q_k) / \tau)}$, where $\text{sim}(a, b) = \frac{a \cdot b}{\|a\| \|b\|}$ is the cosine similarity between context representations and quantized latent representations.
    - Given context network output $c_t$ centered over masked time step $t$, the model needs to identify the true quantized latent representation $q_t$ amidst $K$ negative samples / distractors.
    - Negatives/distractors are uniformly sampled from other masked time steps of the same utterance.
    - **Notable difference from wav2vec**: wav2vec's contrastive loss takes a sigmoid cross-entropy formulation (multiple binary decisions, one per sample), where as wav2vec 2.0 takes a softmax cross-entropy formulation (categorial classification, single multinomial decision over all samples). <https://chatgpt.com/share/68a24514-7654-8005-ac77-5f6ddd099c57>
  - **Codebook diversity loss**: $L_{codebook-diversity} = -H(\tilde{p}) / V = \frac{1}{V} \sum_{v=1}^{V} \tilde{p}_v \log \tilde{p}_v$
    - $\tilde{p}_v$ is the softmax probability of choosing codebook entry $v$ (used for Gumbel-Softmax formulation, but softmax computed here without the Gumbel noise and temperature terms), averaged over all utterances in a batch.
    - Encourages equal use of entries in the codebook by maximizing entropy of codebook selection in a batch.
    - When there are $G$ codebooks in the Product Quantization scenario, the loss is averaged over them accordingly.
- **Downstream use - ASR finetuning**:
  - Finetuning for ASR by adding a linear projection on top of the context network, and trained with CTC.
  - Outperforms previous SoTA with 100x less labeled data on Librispeech.
- **Lineage**: <https://chatgpt.com/share/68a3908b-d03c-8005-b91e-5f2571b2acc0>
  - **(1) wav2vec**: Raw audio $X$ to latent representation $Z$ (encoder network $f \colon X \to Z$) and then to context embeddings $C$ (context network $g \colon Z \to C$). The model is trained to have $c_t$ distinguish latent representations $k$ steps in the future $z_{t+k}$ from negative latent representations $z'$ using a contrastive loss.
  - **(2) wav2vec -> vq-wav2vec**: Add discretization to the latent representation ($q \colon Z \to \tilde{Z}$), and have the context network output $c_t$ predict the reconstructed latent rather $k$ steps in the future ($\tilde{z}_{t+k}$) rather than the continuous latent embedding ($z_{t+k}$). Make speech look like language tokens, apply BERT-style masked LM.
  - **(3) vq-wav2vec -> wav2vec 2.0**: Collapse the two-stage pipeline of vq-wav2vec (wav2vec producing discrete audio tokens, and then using those audio tokens to train a BERT-style masked LM) into one model; instead of separately training BERT on discrete audio tokens, integrate a transformer over masked latent features and predict quantized units directly; downstream use on ASR by finetuning rather than using pretrained representations as inputs to train a separate ASR model.

## [2021] HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units

- **Date**: 2025-08-17
- **Arxiv**: <https://arxiv.org/abs/2106.07447>
- **Paperpile**: <https://app.paperpile.com/view/?id=b5855f1a-4cd9-41a8-b651-ae0db12fbd14>

---

- **Intro**:
  - HuBERT (Hidden Unit BERT): wav2vec 2.0 style SSL pretraining for speech, but rather than having the targets for contrastive loss come from within the model (in the form of quantized latent representations), HuBERT performs **offline clustering of MFCC features uses the cluster IDs as targets for masked spans**.
  - > Speech  signals  differ  from  text  and  images  in  that  they are continuous-valued sequences. Self-supervised learning for the  speech  recognition  domain  faces  unique  challenges  from those in CV and NLP. Firstly, the presence of multiple sounds in  each  input  utterance  breaks  the  instance  classification  as- sumption used in many CV pre-training approaches. Secondly, during pre-training, there is no prior lexicon of discrete sound units available, as in NLP applications in which words or word pieces are used, hindering the use of predictive losses. Lastly, the  boundaries  between  sound  units  are  not  known,  which complicates masked prediction pre-training.
- **Method**:
  - **Model**: Same as wav2vec 2.0, but there's no quantization module $q$, and instead the targets are clustered offline from MFCC features.
  - **Loss**: Same loss as wav2vec 2.0 - softmax cross-entropy over cosine similarity between the model's output embeddings and the target cluster embedding corresponding to the masked spans. Optionally also has an additional term for the unmasked spans.
  - **Extension - Cluster ensembles**: Can extend this trivially to cluster ensembles. Example, an ensemble of k-means models with different codebook sizes can create targets of different granularity and help learn richer representations. Loss is then summed over the ensembles.
  - **Extension - Iterative refinement of cluster targets**: Another  direction for improved representation is refining the cluster assignments for target prediction throughout the learning process. Example: start the training process with cluster targets as k-means over MFCC features, but progressively change it up to predict clusters computed over learned latent representations of the model being trained.
- **Lineage**: <https://chatgpt.com/share/68a3908b-d03c-8005-b91e-5f2571b2acc0>
  - **(1) wav2vec**: Raw audio $X$ to latent representation $Z$ (encoder network $f \colon X \to Z$) and then to context embeddings $C$ (context network $g \colon Z \to C$). The model is trained to have $c_t$ distinguish latent representations $k$ steps in the future $z_{t+k}$ from negative latent representations $z'$ using a contrastive loss.
  - **(2) wav2vec -> vq-wav2vec**: Add discretization to the latent representation ($q \colon Z \to \tilde{Z}$), and have the context network output $c_t$ predict the reconstructed latent rather $k$ steps in the future ($\tilde{z}_{t+k}$) rather than the continuous latent embedding ($z_{t+k}$). Make speech look like language tokens, apply BERT-style masked LM.
  - **(3) vq-wav2vec -> wav2vec 2.0**: Collapse the two-stage pipeline of vq-wav2vec (wav2vec producing discrete audio tokens, and then using those audio tokens to train a BERT-style masked LM) into one model; instead of separately training BERT on discrete audio tokens, integrate a transformer over masked latent features and predict quantized units directly; downstream use on ASR by finetuning rather than using pretrained representations as inputs to train a separate ASR model.
  - **(4) wav2vec 2.0 -> HuBERT**: Conceptually a "step back" from wav2vec 2.0 in terms of being fully end-to-end. wav2vec 2.0 is farther right on the self-supervision axis as the targets for contrastive loss are also generated by the model itself (via its quantizaton module). On the other hand, HuBERT takes a more "conservative" approach by generating offline targets via k-means clustering of MFCC features to favor more stable learning. HuBERT does update its targets during the course of training by clustering intermediate layer features, but this is done much more infrequently (whereas wav2vec 2.0 can be seen as updating its targets after every gradient update step).

## [2022] Whisper: Robust Speech Recognition via Large-Scale Weak Supervision

- **Date**: 2025-09-03
- **Arxiv**: <https://arxiv.org/abs/2212.04356>
- **Paperpile**: <https://app.paperpile.com/view/?id=170c2958-cbd1-41de-a6cc-8ef0fc1ee507>
- **Code**: <https://github.com/openai/whisper>

---

- **Intro**:
  - Self-supervised methods like wav2vec 2.0 has energized progress in speech recognition because they can rely on much larger datasets of raw audio without the need for human labels.
    - But their usefulness is limited because they require finetuning on a downstream task like ASR to be usable.
    - Finetuning also carries the risk of overfitting to the quirks of the specific dataset without meaningfully improving generalization.
    - > Machine learning methods are exceedingly adept at finding patterns within a training dataset which boost performance on held-out data from the same dataset. However, some of these patterns are brittle and spurious and don’t generalize to other datasets and distributions. In a particularly disturbing example, Rad- ford et al. (2021) documented a 9.2% increase in object classification accuracy when fine-tuning a computer vision model on the ImageNet dataset (Russakovsky et al., 2015) without observing any improvement in average accuracy when classifying the same objects on seven other natural image datasets. A model that achieves “superhuman” per- formance when trained on a dataset can still make many basic errors when evaluated on another, possibly precisely because it is exploiting those dataset-specific quirks that humans are oblivious to (Geirhos et al., 2020).
    - > **while unsupervised pre-training has improved the quality of audio encoders dramatically, the lack of an equivalently high-quality pre-trained decoder, combined with a recommended protocol of dataset-specific fine-tuning, is a crucial weakness which limits their usefulness and robustness. The goal of a speech recognition system should be to work reliably “out of the box” in a broad range of environments without requiring supervised fine-tuning of a decoder for every deployment distribution.**
  - Whisper:
    - > speech recognition systems that are pre-trained in a supervised fashion across many datasets/domains exhibit higher robustness and generalize much more  effectively to held-out datasets than models trained on a single source.
    - But even mixing several pre-existing labeled datasets leads to only ~5k hours of data.
    - This can be increased by relaxing the requirement for gold-standard human-validated transcripts and relying on weak labels. *"This trade-off between quality and quantity is often the right call"*. But even here, prior work have led up to only ~30k hours of weakly supervised data.
    - Whisper takes this much further by scaling to 680k hours of weakly labeled multilingual (97 languages) and multitask (X->en translation) data.
    - > **Our work suggests that simple scaling of weakly supervised pre-training has been underappreciated so far for speech recognition. We achieve these results without the need for the self-supervision or self-training techniques that have been a mainstay of recent large-scale speech recognition work.**
- **Method** (Fig 1):
  - **Data Processing**: Faily minimalist approach.
    - Text transcripts are not standardized/normalized removing the need for a separate inverse text normalization step, automated filtering methods to improve trascript quality, heuristics to detect and remove machine-generated transcripts from the training dataset, etc.
    - Audio files chunked into 30-second segments.
  - **Model**:
    - Focus is on studying the capabilities of large-scale weakly supervised pre-training for ASR, not the model itself.
    - Off-the-shelf transformer encoder-decoder architecture (Fig 1).
    - Input audio resampled to 16 kHz, 80-channel log-mel spectrogram computed with 25ms window size and 10ms stride, followed .
    - For English-only models, label text is processed via the same byte-level BPE tokenizer as in GPT-2 for English-only models.
    - For multilingual models, as GPT-2 BPE tokenizer was fit only on English, using it as is will lead to "excessive fragmentation" (such as being broken down into the smallest possible units). Therefore, the tokenizer was refit on multilingual text, while keeping the vocabulary size the same (50k in the case of GPT-2).
  - **Token Format for Multitask Modeling**:
    - Text token format to facilitate performing multiple tasks such as transcription, translation, voice actitivity detection, language identification, alignment, etc.
    - Allows for a single model to replace many different stages of a traditional speech processing pipeline
    - See Fig 1 for token format spec.
- **Experiments and Analysis**:
  - Zero-shot evaluation on a wide-variety of existing speech processing datasets spanning domains, tasks, and languages.
  - **On human-level performance**:
    - > In 2015, Deep Speech 2 (Amodei et al., 2015) reported a speech recognition system matched human-level perfor- mance when transcribing the LibriSpeech test-clean split. As part of their analysis they concluded: “Given this result, we suspect that there is little room for a generic speech sys- tem to further improve on clean read speech without further domain adaptation.” Yet seven years later the SOTA WER on LibriSpeech test-clean has dropped another 73% from their 5.3% to 1.4% (Zhang et al., 2021), far below their re- ported human-level error rate of 5.8%. Despite this massive and unanticipated further improvement in performance on held-out but in-distribution data, speech recognition mod- els trained on LibriSpeech remain far above human error rates when used in other settings.  What explains this gap between reportedly superhuman performance in-distribution and subhuman performance out-of-distribution?
    - > We suspect a large part of this gap between human and machine behavior is due to conflating different capabilities being measured by human and machine performance on a test set.  This claim may seem confusing at first; if both humans and machines are taking the same test, how can it be that different skills are being tested? The difference arises not in the testing but in how they trained for it. Humans are often asked to perform a task given little to no supervision on the specific data distribution being studied. Thus human performance is a measure of out-of-distribution generalization.  But machine learning models are usually evaluated after training on a large amount of supervision from the evaluation distribution, meaning that machine performance is instead a measure of in-distribution generalization. While both humans and machines are being evaluated on the same test data, two quite different abilities are being measured due to a difference in train data.
    - > Whisper models, which are trained on a broad and diverse distribution of audio and evaluated in a zero-shot setting, could potentially match human behavior much better than existing systems. To study whether this is the case ... we can compare Whisper models with both human performance and standard fine-tuned machine learning models and check which they more closely match.
    - > Although the best zero-shot Whisper model has a relatively unremarkable LibriSpeech clean-test WER of 2.5, ..., zero-shot Whisper models have very different robustness properties than supervised LibriSpeech models and out-perform all benchmarked Lib- riSpeech models by large amounts on other datasets.
    - > **This finding suggests emphasizing zero-shot and out-of-distribution evaluations of models, particularly when attempting to compare to human performance, to avoid overstating the capabilities of machine learning systems due to misleading comparisons.**
  - **On scaling**:
    - > The general trend across tasks of diminishing returns when moving from 54,000 hours to our full dataset size of 680,000 hours could suggest that the current best Whisper models are under-trained relative to dataset size and performance could be further improved by a combination of longer training and larger models. It could also suggest that we are nearing the end of performance improvements from dataset size scaling for speech recognition. Further analysis is needed to characterize “scaling laws” for speech recognition in order to decided between these explanations."*
  - **On general-purpose vs specialized models** (multilingual and multitask vs English-only ASR):
    - > for small models trained with moderate amounts of compute, there is indeed negative transfer between tasks and languages: joint mod- els underperform English-only models trained for the same amount of compute. However, multitask and multilingual models scale better and for our largest experiments outperform their English-only counterparts demonstrating positive transfer from other tasks. For our largest experiments, joint models also slightly outperform English-only models even when not adjusting for compute spent per task.

## [2021] SoundStream: An End-to-End Neural Audio Codec

- **Date**: 2025-07-16
- **Arxiv**: <https://arxiv.org/abs/2107.03312>
- **Paperpile**: <https://app.paperpile.com/view/?id=402f89dc-31cb-4a1b-a930-92b10377ac4c>

---

- **Intro**:
  - Two broad categories of audio codecs: (1) waveform codecs, (2) parametric codecs.
  - (1) waveform codecs: time-domain waveform to time-frequency domain, and then coefficients are quantized and entropy coded. High-quality reconstruction at medium/high bitrates, but coding artifacts at low bitrates.
  - (2) parametric codecs: using a parametric model prior that describes the audio synthesis process. The encoder estimates the parameters of the model which are then quantized, and the decoder reconstructs a time-domain waveform using a synthesis model driven by the quantized parameters. Unlike waveform codes, the goal is not faithful reconstruction, but to generate audio that is perceptionally similar to the original.
- **Soundstream - e2e ML**
  - > SoundStream leverages  state-of-the-art  solutions  in  the  field of  neural  audio  synthesis,  and  introduces  a  new  learnable quantization module, to deliver audio at high perceptual quality, while operating at low-to-medium bitrates.
  - **(1) Encoder (fully convolutional)**: takes raw audio waveform and produces a stream of embeddings at a lower sampling rate.
    - only causal conv: padding applied to the past and not to the future
    - 24kHz original waveform to 75Hz embeddings.
  - **(2) RVQ (Residual Vector Quantizer)**: a variable number of residual vector quantizers that take the embeddings and quantize.
  - **(3) Decoder (fully convolutional)**: takes quantized embeddings and reconstructs an approximation of the original waveform.
    - Transposed conv for upsampling, but otherwise similar to the encoder reversed.
    - 75Hz quantized embeddings to 24kHz reconstructed waveform.
  - Trained with both reconstruction and adversarial losses.
  - Both encoder and decoder are causal, therefore the overall architectural latency is dictated just by the downsampling ratio of the encoder.
  - "Quantizer dropout" helps the model handle different bitrates.
  - Compared to mel-spectrogram features, the learned encoder had much better coding efficiency.
- **Residual Vector Quantizer (RVQ)**:
  - Goal: Compress the output of the encoder $\text{enc}(x) \in \mathbb{R}^{T \times D}$ to a target bitrate of $R$ bits per second (bps).
  - Naive Vector Quantizer (VQ) will lead to a prohibitively large codebook size. For example, if the target bitrate $R$ is 5000 bps, and the encoder output is at 50Hz (50 frames/sec), then this corresponds to 5000/50 = 100 bits allocated to each frame. This would mean a codebook / lookup table size of $2^{100}$ entries.
  - On the other hand, with Residual Vector Quantizer (RVQ), the same 100 bits for each frame can be allocated by using $N_q = 10$ vector quantizers to sequentially quantize the residuals, each with a codebook size of $2^{10}$ entries (much more manageable).
  - The codebook is initialized with k-means of the first batch embeddings and then updated using exponential moving average (rather than backprop): <https://chatgpt.com/share/689cad67-5ff4-8005-b471-7de54dc8f206>
    - > The codebook of each quantizer is trained with exponential moving  average  updates,  following  the  method  proposed  in VQ-VAE-2 [32]. To improve the usage of the codebooks we use two additional methods. First, instead of using a random initialization  for  the  codebook  vectors,  we  run  the  k-means algorithm  on  the  first  training  batch  and  use  the  learned centroids  as  initialization.  This  allows  the  codebook  to  be close to the distribution of its inputs and improves its usage. Second, as proposed in [34], when a codebook vector has not been assigned any input frame for several batches, we replace it with an input frame randomly sampled within the current batch. More precisely, we track the exponential moving average of the assignments to each vector (with a decay factor of 0.99) and replace the vectors of which this statistic falls below 2.
- **Quantizer dropout to build a single model that works for different bitrates**:
  - > Since the vector quantizers are trained jointly with the encoder/decoder, in principle a different SoundStream model should be trained for each target bitrate. Instead, having a single bitrate scalable model that can operate at several target bitrates is much more practical, since this reduces the memory footprint needed to store model parameters both at the encoder and decoder side.
  - This is done by randomly selecting the first $n_q$ of the total $N_q$ vector quantizers. In the example above, we had 10 vector quantizers of $2^{10}$ entries each to achieve a total bitrate per frame of 100 bits. If we choose, say, the first 7 vector quantizers and drop the rest, then the bitrate per frame will be 70 bits.
  - During training, quantizer dropout is applied as above. During inference, for a target bitrate, we choose the required number of the first $n_q$ quantizers.
  - > A key advantage of our residual vector quantizer is that the dimensionality of the embeddings does not change with the bitrate. Indeed, the additive composition of  the  outputs  of  each  VQ  layer  progressively  refines  the quantized embeddings, while keeping the same shape. Hence, no architectural changes are needed in neither the encoder nor the decoder to accommodate different bitrates
- **Training objective**:
  - Definitions:
    - Generator: Soundstream generator $G(x) = dec(Q(enc(x)))$ takes the waveform $x$ through the encoder, the quantizer and the decoder.
    - $\hat{x} = G(x)$ is the reconstructed waveform.
    - Discriminator: $D(x)$ is the discriminator trained to distinguish between the original $x$ and the reconstructed $\hat{x}$ audio.
  - Loss $L = L_{D} + L_{G}^{adversarial} + L_{G}^{reconstruction}$
    - **Reconstruction loss** $L_{G}^{reconstruction}$: L1 or L2 loss between the spectrograms of the original $x$ and decoded $\hat{x}$ audio.
    - **Adversarial loss** $L_{G}^{adversarial}$: Hinge loss forcing the generator to produce outputs that the discriminator classifies as "real" (+1) and not as "fake" (-1). $\max(0, 1 - D(G(x)))$.
      - **Hinge loss**: Used for maximum margin classification (such as in SVM). $L(y') = \max(0, 1 - y \cdot y')$, where $y$ is the ground-truth class (either +1 or -1), and $y'$ is the predicted logit ("raw" output of the classifier's decision function, not the predicted class label). When $y$ and $y'$ have the same sign (meaning y predicts the right class) and $|y'| \geq 1$ (correct class prediction and enough margin), the loss is 0. When they have the same sign, but $|y'| < 1$ (correct class prediction, but not enough margin), the loss decreases linearly with $y'$. When they have opposite signs (incorrect class prediction), the loss increases linearly with $y'$. <https://en.wikipedia.org/wiki/Hinge_loss>.
    - **Discriminator loss** $L_{D}$: Hinge loss over the logits of the discriminator forcing it to classify $x$ as real (+1) and $G(x)$ as fake (-1). $\max(0, 1 - D(x)) + \max(0, 1 + D(G(x)))$.
- **Joint compression and enhancement using FiLM conditioning**:
  - Typically, audio compression and audio enhancement are done by separate modules.
  - In SoundStream codec training, compression and enhancement is merged into a single model as follows:
    - FiLM (Feature-wise Linear  Modulation) conditioning based on a boolean `denoise` parameter.
    - During training, when denoise is False, the input and target audio are the same. When denoise is True, the target is the cleaner/enhanced version of the input audio.
    - When the input itself is clean, the model is trained with input = target and denoise being both True and False. This is done to prevent SoundStream from adversely affecting clean audio when denoising is enabled.
- **Lineage**: <https://chatgpt.com/share/68af5b20-4658-8005-9c83-8ee9afe52c2d>
  - **(1) SoundStream**: Foundational codec for discrete tokenization detokenization of audio.

## [2022] EnCodec: High Fidelity Neural Audio Compression

- **Date**: 2025-09-02
- **Arxiv**: <https://arxiv.org/abs/2210.13438>
- **Paperpile**: <https://app.paperpile.com/view/?id=9d3a89f3-7472-47fc-a42a-a08ee2ca1c54>

---

- **Abstract**:
  - > We introduce a state-of-the-art real-time, high-fidelity, audio codec leveraging neural networks. It consists in a streaming encoder-decoder architecture with quantized latent space trained in an end-to-end fashion. We simplify and speed-up the training by using a single multiscale spectrogram adversary that efficiently reduces artifacts and produce high-quality samples. We introduce a novel loss balancer mechanism to stabilize training: the weight of a loss now defines the fraction of the overall gradient it should represent, thus decoupling the choice of this hyper-parameter from the typical scale of the loss. Finally, we study how lightweight Transformer models can be used to further compress the obtained representation by up to 40%, while staying faster than real time. We provide a detailed description of the key design choices of the proposed model including: training objective, architectural changes and a study of various perceptual loss functions. We present an extensive subjective evaluation (MUSHRA tests) together with an ablation study for a range of bandwidths and audio domains, including speech, noisy-reverberant speech, and music. Our approach is superior to the baselines methods across all evaluated settings, considering both 24 kHz monophonic and 48 kHz stereophonic audio. Code and models are available at github.com/facebookresearch/encodec.
- SoundStream, with some improvements.

## [2022] AudioLM: a Language Modeling Approach to Audio Generation

- **Date**: 2025-08-10
- **Arxiv**: <https://arxiv.org/abs/2209.03143>
- **Paperpile**: <https://app.paperpile.com/view/?id=a0f4dbd9-e3c2-4c78-bdcc-8156542313b2>

---

- **Intro**:
  - > We  introduce  AudioLM,  a  framework  for  high- quality audio generation with long-term consistency. AudioLM maps the input audio to a sequence of discrete tokens and casts au- dio generation as a language modeling task in this representation space. We show how existing audio tokenizers provide different trade-offs between reconstruction quality and long-term structure, and  we  propose  a  hybrid  tokenization  scheme  to  achieve  both objectives. Namely, we leverage the discretized activations of a masked language model pre-trained on audio to capture long-term structure and the discrete codes produced by a neural audio codec to achieve high-quality synthesis. By training on large corpora of raw audio waveforms, AudioLM learns to generate natural and co- herent continuations given short prompts. When trained on speech, and  without  any  transcript  or  annotation,  AudioLM  generates syntactically and semantically plausible speech continuations while also maintaining speaker identity and prosody for unseen speakers. Furthermore, we demonstrate how our approach extends beyond speech by generating coherent piano music continuations, despite being  trained  without  any  symbolic  representation  of  music.
  - Achieves the objective of both high-quality audio generation as well as long-term coherent structure. Combines advances in neural audio compression (SoundStream), self-supervised speech pretraining (wav2vec 2.0, HuBERT, w2v-bert, etc), and language modeling. Essentially, training a language model to generate both semantic and acoustic tokens simultaenously leads to high audio quality and long-term consistency.
    - **Semantic tokens (coarse)**: constructed from a model pretrained with a self-supervised masked language modeling objective.
    - **Acoustic tokens (finer)**: produced by SoundStream neural codec.
- **Three components of the model**:
  - **(1) Tokenizer**: Takes a single-channel audio sequence $x \in \mathbb{R}^T$ and produces a sequence of discrete tokens $h = \text{enc}(x)$ of length $T' \ll T$.
  - **(2) Decoder-only transformer language model** that operates on $h$ to predict the next sequence of tokens $\hat{h}$ autoregressively. Since $T' \ll T$, the language model can now capture long-term dependency more efficiently (as self-attention complexity grows quadratically wrt sequence length).
  - **(3) Detokenizer**: Maps the sequence of predicted tokens $\hat{h}$ to produce the output audio waveform $\hat{x} = \text{dec}(\hat{h})$.
  - In AudioLM, (1) and (3) are pretrained and frozen (such as from SoundStream and w2v-BERT), and only (2) is trained.
- **Details**:
  - **Hybrid tokenization scheme** combining acoustic and semantic tokens:
    - **(i) Acoustic tokens** produced by SoundStream model (encoder + RVQ). See SoundStream notes for details. For a SoundStream model with $Q$ residual vector quantizers with $N$ bits allocated to each (such that the codebook size per quantizer is $2^N$), the raw audio waveform $x \in \mathbb{R}^T$ is transformed to $y \in \{1,...,2^N\}^{T_A \times Q}$ discrete tokens, where $T/T_A$ is the downsampling factor of the SoundStream encoder.
    - **(ii) Semantic tokens** produced by an intermediate layer of w2v-BERT after applying k-means and using the centroid indices. With $K$ clusters, the raw audio waveform $x \in \mathbb{R}^T$ is transformed to $z \in \{1,...,K\}^{T_S}$, where $T/T_S$ is the downsampling factor of the w2v-BERT encoder.
    - The combination of acoustic and semantic tokens helps reconcile conflicting requirements - fine-grained acoustic tokenization (more number of tokens for a fixed time window) helps reconstructing higher quality audio, whereas coarse-grained semantic tokenization (fewer number of tokens for a fixed time window) helps capture long-term dependency efficiently.
  - **Hierarchical modeling of semantic and acoustic tokens**: (Fig 2)
    - In what order should the semantic and acoustic tokens be presented to language model training? Key insights:
      - (a) Semantic tokens (which capture linguistic context) should only be dependent on past semantic tokens and not on past acoustic tokens (which capture context needed for audio synthesis).
      - (b) The acoustic tokens generated with $Q$ residual vector quantizers, the earlier vector quantizers (say, $Q'$) capture coarse acoustic info whereas the later ones (the remaining $Q-Q'$ vector quantizers) capture finer acoustic info (see SoundStream for details). Given this, the finer acoustic tokens should be independent of semantic tokens given coarse acoustic tokens.
    - Given the above insights, modeling is done in three stages:
      - **Stage 1 (Semantic modeling)**: Autoregressive next-token prediction trained on just the semantic tokens $(z_1, z_2, ..., z_{T_S})$. There are no acoustic tokens here per insight (a) above.
      - **Stage 2 (Coarse acoustic modeling)**: Autoregressive next-token prediction trained on the coarse acoustic tokens from the first $Q'$ vector quantizers $(y_1^1, ..., y_1^{Q'}, y_2^1, ..., y_2^{Q'}, y_{T_A}^1, ..., y_{T_A}^{Q'})$ conditioned on the semantic tokens $(z_1, z_2, ..., z_{T_S})$.
      - **Stage 3 (Fine acoustic modeling)**: Autoregressive next-token prediction trained on the fine acoustic tokens from the remaining $Q-Q'$ vector quantizers, conditioned on the coarse acoustic tokens from the first $Q'$ vector quantizers. There are no semantic tokens per per insight (b) above.
      - Stages 2 & 3 can be merged into a single stage (to just model the entire semantic and acoustic token sequence at once), but breaking it down this way improves efficiency by limiting the max sequence length needed to be dealt with.
  - **Inference**:
    - Target application: Given an audio prompt $x$, generate continuations that maintain both semantic and acoustic coherence.
    - Steps:
      - **Step 1 (Semantic token generation)**: First obtain semantic tokens corresponding to the prompt $x$ from w2v-BERT encoder and feed those into the transformer language model to generate semantic token completions.
      - **Step 2 (Coarse acoustic token generation)**: Then, concatenate the entire semantic token sequence (corresponding to the prompt as well as those generated above) along with the coarse acoustic tokens corresponding to the prompt $x$ (obtained from SoundStream). Feed this as conditioning to the coarse acoustic model to generate the coarse acoustic token completions.
      - **Step 3 (Fine acoustic token generation)**: Concatenate the entire coarse acoustic token sequence (corresponding to the prompt as well as those generated above) along with the fine acoustic tokens corresponding to the prompt $x$ (obtained from SoundStream). Feed this as conditioning to the fine acoustic model to generate the fine acoustic token completions.
      - **Step 4 (SoundStream decoder for audio generation)**: Finally, feed all the acoustic tokens to the SoundStream decoder to reconstruct the audio waveform output $\hat{x}$.
- **Lineage**: <https://chatgpt.com/share/68af5b20-4658-8005-9c83-8ee9afe52c2d>
  - **(1) SoundStream**: Foundational codec for discrete tokenization detokenization of audio.
  - **(2) AudioLM**: Train autoregressive generative LM over discrete audio tokens.

## [2023] SoundStorm: Efficient Parallel Audio Generation

- **Date**: 2025-08-27
- **Arxiv**: <https://arxiv.org/abs/2305.09636>
- **Paperpile**: <https://app.paperpile.com/view/?id=b6a32e14-6a86-4bce-bfed-888e152d13c4>

---

- **Intro**:
  - AudioLM's autoregressive decoding process is too slow for audio generation. For generating high-quality audio by modeling the tokens of a neural codec (such as SoundStream), the rate of the discrete tokens should be high, resulting in either an exponential growth in codebook size or in long token sequences. For attention-based models, runtime complexity is quadradic wrt sequence length, leading to tradeoff between perception quality of the generated audio and runtime.
  - Three orthogonal approaches to combat the problem of generating long audio token sequences:
    - (a) efficient attention mechanisms,
    - (b) non-autoregressive parallel decoding schemes,
    - (c) custom architectures adapted to the special structure of tokens produced by neural audio codec (such as relying on RVQ's hierachical structure in SoundStream and AudioLM).
      - > We believe that it is the special structure of the audio token sequence that holds the most promise for future advances in long-sequence audio modeling. Concretely, both Sound- Stream (Zeghidour et al., 2022) and EnCodec (D ́efossez et al., 2022) rely on Residual Vector Quantization (RVQ), where each compressed audio frame is quantized by a series of quantizers, with each quantizer operating on the residual of the previous one, and the number of quantizers control- ling the overall bitrate.  This induces a hierarchical token structure, where tokens from finer RVQ levels contribute less to the perceptual quality, allowing for efficient factor- izations and approximations of the joint distribution of the token sequence. Hence, the models and decoding schemes should take this special structure of the input into account for efficient training and inference.
  - SoundStorm improves efficiency of generating long audio token sequences by relying on:
    - **(i) an architecture adapted to the hierarchical structure of audio tokens**,
    - **(ii) a parallel, non-autoregressive, confidence-based decoding scheme inspired by MaskGIT** for residual vector quantized (RVQ) token sequences.
- TODO
- **Lineage**: <https://chatgpt.com/share/68af5b20-4658-8005-9c83-8ee9afe52c2d>
  - **(1) SoundStream**: Foundational codec for discrete tokenization detokenization of audio.
  - **(2) AudioLM**: Train autoregressive generative LM over discrete audio tokens.
  - **(3) SoundStorm**: Efficiency improvement over the slow autoregressive decoding process of AudioLM.

## [2023] AudioPaLM: A Large Language Model That Can Speak and Listen

- **Date**: 2025-07-16
- **Arxiv**: <https://arxiv.org/abs/2306.12925>
- **Paperpile**: <https://app.paperpile.com/view/?id=39010e7d-62e7-4c64-8166-71d7df2ed434>

---

- **Intro**:
  - AudoPaLM: Fuses text and speech based LMs into a unified multimodal architecture. Combines the best of AudioLM and PaLM-2.
  - Hitherto, although there have been LMs that combine text and audio (speech-to-text or text-to-speech models), the text tokens and audio tokens use different vocabularies. For example, in the case of AudioLM, although an autoregressive LM is a key component of the model, it's only trained on audio tokens (semantic and acoustic tokens obtained from w2v-BERT and SoundStream respectively, see AudioLM notes above). Among other things, audio and text token vocabularies being different means speech-to-text and text-to-speech will have to be two different models.
  - **AudioPaLM combines text and audio vocabularies into a multimodal single vocabulary, allowing for training a single model in both directions and arbitrary interleaving of speech and text**. A single model can then do speech recognition, text-to-speech synthesis, and speech-to-speech translation, unifying tasks that are traditionally solved by heterogeneous models into a single architecture and training run.
  - Initalized from a pretrained text-only LM such as PaLM-2.
- **Method**:
  - Decoder-only transformer to model sequences consisting of text and audio tokens. As far as the model is concerned, text and audio are just sequences of arbitrary integers, as the inputs are tokenized before feeding to the model, and any outputs are detokenized before being returned to a user of the model.
  - **Audio tokenization/detokenization**: Similar to AudioLM but some tweaks.
  - **Modifying text-only decoder to model both text and audio**:
    - In the original text-only autoregressive transformer like PaLM, SentencePiece text tokens are converted to text token embeddings via an `nn.Embedding` layer of shape $V_T \times E$ where $V_T$ is the text token vocabulary size and $E$ the decoder architecture's embedding dim. At the other end, there's an output projection `nn.Linear` layer of shape $E \times V_T$ to produce the logits.
    - Given audio token vocab size $V_A$ (such as from SoundStream or AudioLM tokenizer), the only change to the text-only autoregressive decoder arch is to increase the input `nn.Embedding` to be of shape $(V_T + V_A) \times E$ and the output `nn.Linear` to be of shape $E \times (V_T + V_A)$.
    - The rows in the input token embedding and the output project layer weights that correspond to the text tokens, as well as the rest of the decoder layers, are initalized from a pretrained text-only autoregressive decoder model.
- **Training Tasks**:
  - **Combinations of fields involved**:
    - audio
    - transcript
    - translated audio
    - translated transcript.
  - **Tasks involved**:
    - **ASR (Automatic Speech Recognition)**: audio -> transcript
    - **AST (Automatic Speech Translation)**: audio -> translated transcript
    - **S2ST (Speech-to-Speech Translation)**: audio -> translated audio
    - **TTS (Text-to-Speech)**: transcript -> audio
    - **MT (Machine Translation)**: transcript -> translated transcript
  - Input prefixed with task tag to signal the model which task to perform, such as `[ASR French]`, `[TTS English]`, `[S2ST English French]` etc. The tag is tokenized using the normal text tokenizer of the model.
  - **Task chaining**:
    - Similar in spirit to chain of thought prompting, can prompt the model to output intermediate steps.
    - Example, for `[S2ST English French]` can be alternatively prompted as `[ASR AST S2ST English French]` so that it outputs the intermediate English and French text transcriptions in addition to the final French audio.
    - Note: This is not the same as calling the model three times. The model performs the task of `[ASR AST S2ST English French]` as a single autoregressive decoding process. This means it can attend to the intermediate decoding tokens (English and French text tokens in this example) to improve performance, in addition to attending to just the input tokens (English audio tokens) and the prior decoded output tokens (partial French audio tokens).
- **Lineage**: <https://chatgpt.com/share/68af5b20-4658-8005-9c83-8ee9afe52c2d>
  - **(1) SoundStream**: Foundational codec for discrete tokenization detokenization of audio.
  - **(2) AudioLM**: Train autoregressive generative LM over discrete audio tokens.
  - **(3) SoundStorm**: Efficiency improvement over the slow autoregressive decoding process of AudioLM.
  - **(4) AudioPaLM**: Merges AudioLM (speech-only LM) with PaLM-2 (text-only LM) by unifying the audio and text vocabularies into a single multimodal vocabulary. Allows for training a single model in both directions, arbitrary interleaving of speech and text, and enables a single model to do ASR, TTS, speech-to-speech translation, etc.

## [2023] MusicGen: Simple and Controllable Music Generation

- **Date**: 2025-09-03
- **Arxiv**: <https://arxiv.org/abs/2306.05284>
- **Paperpile**: <https://app.paperpile.com/view/?id=4f7cf33d-a180-44cf-a660-7a822c11ffea>

---

- **Intro**:
  - Tackles the task of **conditional music generation** based on text and/or melody.
  - Additional challenges with music generation compared to speech generation:
    - (1) Requires modeling longer range sequences.
    - (2) Unlike speech, music requires the use of full frequency spectrum.
    - (3) Above leads to higher sampling rate (44.1 kHz or 48 kHz for music recordings vs 16 kHz for speech).
    - (4) Humans are highly sensitive to disharmony, leaves little room for melodic errors.
    - (5) Music creators need to be able to control the generation process in a diverse set of methods such as key, instruments, melody, genre, etc.
  - Unlike prior works that take a cascaded or hierarchical approach, MusicGen relies on a **single-stage transformer LM operating over several streams of discrete tokens via efficient token interleaving patterns**.
    - Introduces a general framework for modeling multiple parallel streams of acoustic tokens (such as the output of RVQ in SoundStream).
    - Reduces the number of autoregresssive timesteps involved in generated acoustic tokens over all streams (corresponding to each codebook/quantizer in RVQ). For example, AudioLM hierarchically models the token steams from coarse to fine quantizers, essentially flattening all streams, and resulting in $T \times Q$ timesteps, where $Q$ is the number of quantizers/codebooks in RVQ.
  - **Contributions**:
    - (i) a simple and efficient model to generate high quality music at 32 kHz through an efficient codebook interleaving strategy.
    - (ii) a single model to perform both text and melody-conditioned generation.
    - (iii) extensive objective and human evaluations.
- **Method**:
  - **(1) Audio Tokenization**: Soundstream/EnCodec with $Q$ codebooks in RVQ, resulting in $Q$ parallel streams of discrete tokens.
  - **(2) Codebook interleaving patterns** (Fig 1):
    - **(a) Flattening pattern ($T \times Q$ timesteps)**: Baseline approach as in AudioLM. Flatten the outputs of all $Q$ streams, resulting in a sequence of $Q$ tokens per timestep predicted autoregressively. Fully satisfies dependencies along both $T$ and $Q$ axes, but wastefully.
    - **(b) Parallel pattern ($T$ timesteps)**: At each time step, the model predicts logits for all $Q$ codebooks in parallel. Fully satisifies dependency only along $T$ axis.
    - **(c) Coarse-first pattern ($T \times 2$ timesteps)**: First predict the tokens corresponding to zeroth/coarsest codebook for all $T$ timesteps autoregresively. Then, predict the remaining $Q-1$ codebook tokens in parallel for each of the $T$ timesteps, conditioned on the zeroth/coarsest codebook tokens. Partially satisifies dependencies along both $T$ and $Q$ axes.
    - **(d) Delay pattern ($T + Q - 1$ timesteps)**: A pipelined approach with conceptual similarlity to pipeline parallelism bubble reduction strategies [[book-huggingface-ultra-scale-llm-training.md]]. Partially satisfies dependency along $T$ axis, fully satisfies dependency along $Q$ axis. See Fig 1.
  - **(3) Model conditioning**:
    - **(a) Text conditioning**: Given a textual description matching the input audio, generate a conditioning tensor $C \in \mathbb{R}^{T_C \times D}$ where $D$ is embedding dim of the autoregresssive model, such as using a pretrained text encoder.
    - **(b) Melody conditioning**: Optionally also condition ussing the chromagram of another track.
      - > In preliminary experiments, we observed that conditioning on the raw chromagram often led to reconstructing the original sample, resulting in overfitting. To reduce it, we introduce an information bottleneck by choosing the dominant time-frequency bin in each time step.
  - **(4) Model architecture**:
    - The core architecture itself is a standard transformer decoder, but input token embedding lookup results sum according to the contribution from each codebook, and the output logic projection is modified accordingly to predict multiple codebook entries in parallel.

## [2023] Speech-Llama: Prompting Large Language Models with Speech Recognition Abilities

- **Date**: 2025-03-04
- **Arxiv**: <https://arxiv.org/abs/2307.11795>
- **Paperpile**: <https://app.paperpile.com/view/?id=d24a1f36-56fd-47cf-b1d9-fbf2ee244e7b>

---

- Simple approach overall. The goal is to leverage llama (trained just on text, not multilingual) to perform ASR.
  - Speech embeddings + text-based LLM to improve upon ASR baselines.
  - Not generative.
  - No discretization of speech tokens anywhere unlike - the input to Llama is audio embeddings from conformer.
- Prepend text prompt in Llama with audio embeddings from a conformer trained with CTC.
  - The initial input would be a sequence of audio embeddings followed by the `<BOS>` token at which point the LLM would start emitting text (hopefully transcribing the audio) auto-regressively.
  - > The ASR-LLM problem can possibly be reinterpreted as a copying/translation task where the LLM needs to regurgitate the information in the audio sequence. If the audio encoder provides a sequence of embeddings aligned with the text embeddings the problem collapses to a repetition task which should not require the full capacity of an LLM.
- Evaluate both using the LLM as is as well as LoRA-finetuned variant.
  - LoRA adaptor is only applied for the attention projection weights (q/k/v/o); feedforward, embedding and final linear layer weights remain frozen.
  - LoRA parameters: $\alpha = 16$ and $r = 8$.
- Section 4: Cosine similarly between the audio embeddings (output of speech conformer followed by linear projection to match dimensionality) and the text embeddings (output of llama's `nn.Embedding` mapping text tokens to embeddings) shows monotonic similarity.
- > The speech recognition task can be interpreted as a regurgitation task -- the language model is tasked with cleaning and repeating (in the same order) information that is present in the audio encoder output sequence.

## [2024] Moshi: a speech-text foundation model for real-time dialogue

- **Date**: 2025-09-05
- **Arxiv**: <https://arxiv.org/abs/2410.00037>
- **Paperpile**: <https://app.paperpile.com/view/?id=c7c6bc77-662f-4dae-a0bb-88f49c929788>

---

- TODO
