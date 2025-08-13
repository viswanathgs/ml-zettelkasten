# Speech/Audio Papers

**Start:** 2025-07-16  
**End:** TODO  

- [ ] [2016] WaveNet: A Generative Model for Raw Audio. <https://arxiv.org/abs/1609.03499>
- [ ] TODO wav2vec
- [ ] TODO wav2vec 2.0
- [ ] TODO Whisper
- [X] [2021] SoundStream: An End-to-End Neural Audio Codec. <https://arxiv.org/abs/2107.03312>
- [ ] [2023] SoundStorm: Efficient Parallel Audio Generation. <https://arxiv.org/abs/2305.09636>
- [X] [2023] AudioLM: a Language Modeling Approach to Audio Generation. <https://arxiv.org/abs/2209.03143>
- [ ] [2023] AudioPaLM: A Large Language Model That Can Speak and Listen. <https://arxiv.org/abs/2306.12925>
- [ ] [2023] MusicGen: Simple and Controllable Music Generation. <https://arxiv.org/abs/2306.05284>
- [X] [2023] Speech-Llama: Prompting Large Language Models with Speech Recognition Abilities. <https://arxiv.org/abs/2307.11795>
- [ ] [2024] Faster Speech-Llama Inference with Multi-token Prediction. <https://arxiv.org/abs/2409.08148>
- [ ] TODO moshi
- [ ] TODO sesame blog

## [2021] SoundStream: An End-to-End Neural Audio Codec

**Date:** 2025-07-16
**Arxiv:** <https://arxiv.org/abs/2107.03312>
**Paperpile:** <https://app.paperpile.com/view/?id=402f89dc-31cb-4a1b-a930-92b10377ac4c>

- Intro
  - Two broad categories of audio codecs: (1) waveform codecs, (2) parametric codecs.
  - (1) waveform codecs: time-domain waveform to time-frequency domain, and then coefficients are quantized and entropy coded. High-quality reconstruction at medium/high bitrates, but coding artifacts at low bitrates.
  - (2) parametric codecs: using a parametric model prior that describes the audio synthesis process. The encoder estimates the parameters of the model which are then quantized, and the decoder reconstructs a time-domain waveform using a synthesis model driven by the quantized parameters. Unlike waveform codes, the goal is not faithful reconstruction, but to generate audio that is perceptionally similar to the original.
- **Soundstream - e2e ML**
  - > SoundStream leverages  state-of-the-art  solutions  in  the  field of  neural  audio  synthesis,  and  introduces  a  new  learnable quantization module, to deliver audio at high perceptual quality, while operating at low-to-medium bitrates.
  - (1) **Encoder (fully convolutional):** takes raw audio waveform and produces a stream of embeddings at a lower sampling rate.
    - only causal conv: padding applied to the past and not to the future
    - 24kHz original waveform to 75Hz embeddings.
  - (2) **RVQ (Residual Vector Quantizer):** a varaible number of residual vector quantizers that take the embeddings and quantize.
  - (3) **Decoder (fully convolutional):** takes quantized embeddings and reconstructs an approximation of the original waveform.
    - Transposed conv for upsampling, but otherwise similar to the encoder reversed.
    - 75Hz quantized embeddings to 24kHz reconstructed waveform.
  - Trained with both reconstruction and adversarial losses.
  - Both encoder and decoder are causal, therefore the overall architectural latency is dictated just by the downsampling ratio of the encoder.
  - "Quantizer dropout" helps the model handle different bitrates.
  - Compared to mel-spectrogram features, the learned encoder had much better coding efficiency.
- **Residual Vector Quantizer (RVQ):**
  - Goal: Compress the output of the encoder $enc(x) \in R^{T \times D}$ to a target bitrate of $R$ bits per second (bps).
  - Naive Vector Quantizer (VQ) will lead to a prohibitively large codebook size. For example, if the target bitrate $R$ is 5000 bps, and the encoder output is at 50Hz (50 frames/sec), then this corresponds to 5000/50 = 100 bits allocated to each frame. This would mean a codebook / lookup table size of $2^{100}$ entries.
  - On the other hand, with Residual Vector Quantizer (RVQ), the same 100 bits for each frame can be allocated by using $N_q = 10$ vector quantizers to sequentially quantize the residuals, each with a codebook size of $2^{10}$ entries (much more manageable).
  - The codebook is initialized with k-means of the first batch embeddings and then updated using exponential moving average (rather than backprop): <https://chatgpt.com/share/689cad67-5ff4-8005-b471-7de54dc8f206>
    - > The codebook of each quantizer is trained with exponential moving  average  updates,  following  the  method  proposed  in VQ-VAE-2 [32]. To improve the usage of the codebooks we use two additional methods. First, instead of using a random initialization  for  the  codebook  vectors,  we  run  the  k-means algorithm  on  the  first  training  batch  and  use  the  learned centroids  as  initialization.  This  allows  the  codebook  to  be close to the distribution of its inputs and improves its usage. Second, as proposed in [34], when a codebook vector has not been assigned any input frame for several batches, we replace it with an input frame randomly sampled within the current batch. More precisely, we track the exponential moving average of the assignments to each vector (with a decay factor of 0.99) and replace the vectors of which this statistic falls below 2.
- **Quantizer dropout to build a single model that works for different bitrates:**
  - > Since the vector quantizers are trained jointly with the encoder/decoder, in principle a different SoundStream model should be trained for each target bitrate. Instead, having a single bitrate scalable model that can operate at several target bitrates is much more practical, since this reduces the memory footprint needed to store model parameters both at the encoder and decoder side.
  - This is done by randomly selecting the first $n_q$ of the total $N_q$ vector quantizers. In the example above, we had 10 vector quantizers of 2^10 entries each to achieve a total bitrate per frame of 100 bits. If we choose, say, the first 7 vector quantizers and drop the rest, then the bitrate per frame will be 70 bits.
  - During training, quantizer dropout is applied as above. During inference, for a target bitrate, we choose the required number of the first $n_q$ quantizers.
  - > A key advantage of our residual vector quantizer is that the dimensionality of the embeddings does not change with the bitrate. Indeed, the additive composition of  the  outputs  of  each  VQ  layer  progressively  refines  the quantized embeddings, while keeping the same shape. Hence, no architectural changes are needed in neither the encoder nor the decoder to accommodate different bitrates
- **Training objective:**
  - Definitions:
    - Generator: Soundstream generator $G(x) = dec(Q(enc(x)))$ takes the waveform $x$ through the encoder, the quantizer and the decoder.
    - $\hat{x} = G(x)$ is the reconstructed waveform.
    - Discriminator: $D(x)$ is the discriminator trained to distinguish between the original $x$ and the reconstructed $\hat{x}$ audio.
  - Loss $L = L_{D} + L_{G}^{adversarial} + L_{G}^{reconstruction}$
    - **Reconstruction loss** $L_{G}^{reconstruction}$: L1 or L2 loss between the spectrograms of the original $x$ and decoded $\hat{x}$ audio.
    - **Adversarial loss** $L_{G}^{adversarial}$: Hinge loss forcing the generator to produce outputs that the discriminator classifies as "real" (+1) and not as "fake" (-1). $max(0, 1 - D(G(x)))$.
      - **Hinge loss:** Used for maximum margin classification (such as in SVM). $L(y') = max(0, 1 - y * y')$, where $y$ is the ground-truth class (either +1 or -1), and $y'$ is the predicted logit ("raw" output of the classifier's decision function, not the predicted class label). When $y$ and $y'$ have the same sign (meaning y predicts the right class) and $|y'| \ge 1$ (correct class prediction and enough margin), the loss is 0. When they have the same sign, but $|y'| \lt 1$ (correct class prediction, but not enough margin), the loss decreases linearly with $y'$. When they have opposite signs (incorrect class prediction), the loss increases linearly with $y'$. <https://en.wikipedia.org/wiki/Hinge_loss>.
    - **Discriminator loss** $L_{D}$: Hinge loss over the logits of the discriminator forcing it to classify $x$ as real (+1) and $G(x)$ as fake (-1). $max(0, 1 - D(x)) + max(0, 1 + D(G(x)))$.
- Joint compression and enhancement:
  - Typically, audio compression and audio enhancement are done by separate modules.
  - In SoundStream codec training, compression and enhancement is merged into a single model as follows:
    - FiLM (Feature-wise Linear  Modulation) conditioning based on a boolean `denoise` parameter.
    - During training, when denoise is False, the input and target audio are the same. When denoise is True, the target is the cleaner/enhanced version of the input audio.
    - When the input itself is clean, the model is trained with input = target and denoise being both True and False. This is done to prevent SoundStream from adversely affecting clean audio when denoising is enabled.

## [2023] AudioLM: a Language Modeling Approach to Audio Generation

**Date:** 2025-08-10
**Arxiv:** <https://arxiv.org/abs/2209.03143>
**Paperpile:** <https://app.paperpile.com/view/?id=a0f4dbd9-e3c2-4c78-bdcc-8156542313b2>

- Intro:
  - > We  introduce  AudioLM,  a  framework  for  high- quality audio generation with long-term consistency. AudioLM maps the input audio to a sequence of discrete tokens and casts au- dio generation as a language modeling task in this representation space. We show how existing audio tokenizers provide different trade-offs between reconstruction quality and long-term structure, and  we  propose  a  hybrid  tokenization  scheme  to  achieve  both objectives. Namely, we leverage the discretized activations of a masked language model pre-trained on audio to capture long-term structure and the discrete codes produced by a neural audio codec to achieve high-quality synthesis. By training on large corpora of raw audio waveforms, AudioLM learns to generate natural and co- herent continuations given short prompts. When trained on speech, and  without  any  transcript  or  annotation,  AudioLM  generates syntactically and semantically plausible speech continuations while also maintaining speaker identity and prosody for unseen speakers. Furthermore, we demonstrate how our approach extends beyond speech by generating coherent piano music continuations, despite being  trained  without  any  symbolic  representation  of  music.
  - Achieves the objective of both high-quality audio generation as well as long-term coherent structure. Combines advances in neural audio compression (SoundStream), self-supervised speech pretraining (w2v-bert), and language modeling. Essentially, training a language model to generate both semantic and acoustic tokens simultaenously leads to high audio quality and long-term consistency.
    - **Semantic tokens (coarse):** constructed from a model pretrained with a self-supervised masked language modeling objective.
    - **Acoustic tokens (finer):** produced by SoundStream neural codec.
- Three components of the model:
  - **(1) Tokenizer:** Takes a single-channel audio sequence $x \in R^T$ and produces a sequence of discrete tokens $h = enc(x)$ of length $T' \ll T$.
  - **(2) Decoder-only transformer language model** that operates on $h$ to predict the next sequence of tokes $\hat{h}$ autoregressively. Since $T' \ll T$, the language model can now capture long-term dependency more efficiently (as self-attention complexity grows quadratically wrt sequence length).
  - **(3) Detokenizer:** Maps the sequence of predicted tokens $\hat{h}$ to produce the output audio waveform $\hat{x} = dec(\hat{x})$.
  - In AudioLM, (1) and (3) are pretrained and frozen (such as from SoundStream and w2v-BERT), and only (2) is trained.
- Details:
  - **Hybrid tokenization scheme** combining acoustic and semantic tokens:
    - **(i) Acoustic tokens** produced by SoundStream model (encoder + RVQ). See SoundStream notes for details. For a SoundStream model with $Q$ residual vector quantizers with $N$ bits allocated to each (such that the codebook size per quantizer is $2^N$), the raw audio waveform $x \in R^T$ is transformed to $y \in \{1,...,2^N\}^{T_A \times Q}$ discrete tokens, where $T/T_A$ is the downsampling factor of the SoundStream encoder.
    - **(ii) Semantic tokens** produced by an intermediate layer of w2v-BERT after applying k-means and using the centroid indices. With $K$ clusters, the raw audio waveform $x \in R^T$ is transformed to $z \in {1,...,K}^{T_S}$, where $T/T_S$ is the downsampling factor of the w2v-BERT encoder.
    - The combination of acoustic and semantic tokens helps reconcile conflicting requirements - fine-grained acoustic tokenization (more number of tokens for a fixed time window) helps reconstructing higher quality audio, whereas coarse-grained semantic tokenization (fewer number of tokens for a fixed time window) helps capture long-term dependency efficiently.
  - **Hierarchical modeling of semantic and acoustic tokens:** (Fig 2)
    - In what order should the semantic and acoustic tokens be presented to language model training? Key insights:
      - (a) Semantic tokens (which capture linguistic context) should only be dependent on past semantic tokens and not on past acoustic tokens (which capture context needed for audio synthesis).
      - (b) The acoustic tokens generated with $Q$ residual vector quantizers, the earlier vector quantizers (say, $Q'$) capture coarse acoustic info whereas the later ones (the remaining $Q-Q'$ vector quantizers) capture finer acoustic info (see SoundStream for details). Given this, the finer acoustic tokens should be independent of semantic tokens given coarse acoustic tokens.
    - Given the above insights, modeling is done in three stages:
      - **Stage 1 (Semantic modeling):** Autoregressive next-token prediction trained on just the semantic tokens $(z_1, z_2, ..., z_{T_S})$. There are no acoustic tokens here per insight (a) above.
      - **Stage 2 (Coarse acoustic modeling):** Autoregressive next-token prediction trained on the coarse acoustic tokens from the first $Q'$ vector quantizers $(y_1^1, ..., y_1^{Q'}, y_2^1, ..., y_2^{Q'}, y_{T_A}^1, ..., y_{T_A}^{Q'})$ conditioned on the semantic tokens $(z_1, z_2, ..., z_{T_S})$.
      - **Stage 3 (Fine acoustic modeling):** Autoregressive next-token prediction trained on the fine acoustic tokens from the remaining $Q-Q'$ vector quantizers, conditioned on the coarse acoustic tokens from the first $Q'$ vector quantizers. There are no semantic tokens per per insight (b) above.
      - Stages 2 & 3 can be merged into a single stage (to just model the entire semantic and acoustic token sequence at once), but breaking it down this way improves efficiency by limiting the max sequence length needed to be dealt with.
  - Inference:
    - Target application: Given an audio prompt $x$, generate continuations that maintain both semantic and acoustic coherence.
    - Steps:
      - **Semantic token generation:** First obtain semantic tokens corresponding to the prompt $x$ from w2v-BERT encoder and feed those into the transformer language model to generate semantic token completions.
      - **Coarse acoustic token generation:** Then, concatenate the entire semantic token sequence (corresponding to the prompt as well as those generated above) along with the coarse acoustic tokens corresponding to the prompt $x$ (obtained from SoundStream). Feed this as conditioning to the coarse acoustic model to generate the corase acoustic token completions.
      - **Fine acoustic token generation:** Concatenate the entire coarse acoustic token sequence (corresponding to the prompt as well as those generated above) along with the fine acoustic tokens corresponding to the prompt $x$ (obtained from SoundStream). Feed this as conditioning to the fine acoustic model to generate the fine acoustic token completions.
      - **SoundStream decoder for audio generation:** Finally, feed all the acoustic tokens to the SoundStream decoder to reconstruct the audio waveform output $\hat{x}$.
