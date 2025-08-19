# Interspeech 2023

**Created:** 2023-08

## People

- [X] Paola Garcia, JHU: <https://www.clsp.jhu.edu/faculty/paola-garcia/>
  - Diarization, children's speech
  - Also involved in icefall/lhotse/k2 at a high-level
- [X] Reinhold Haeb-Umbach, Paderborn University: <https://www.uni-paderborn.de/person/242>
  - Works mainly on "front-end", like source separation
  - Focused currently on multi-talker separation in meetings
- [X] Marc Delcroix, NTT: <https://scholar.google.com/citations?user=QG8aWfIAAAAJ&hl=en>
- [X] Shinji Watanabe, CMU: <https://www.ece.cmu.edu/directory/bios/Shinji%20Watanabe.html>
  - Was super aware of Meta's stuff including the recent CTRL<>speech reorg
- [X] Scott Wisdom, Google: <https://stwisdom.github.io/>
  - Works closely with Ron Weiss, mostly on audio stuff
- Sanjeev Khudanpur, JHU: <https://www.clsp.jhu.edu/faculty/sanjeev-khudanpur/>

## Papers of note

### [Delay-penalized CTC implemented based on Finite State Transducer](https://arxiv.org/abs/2305.11539)

**Tags:** latency, WFST

- The same idea as fast-emit/fast-elastic style regularization by identifying the first non-blank in a repeated sequence and adding a regualization term.
- But implemented as an FST instead by adding a new state that identifies the first non-blank.
- Therefore, unlike ours, doesn't need to be greedy.
- Built on [k2](https://github.com/k2-fsa/k2). Dan Povey is one of the authors. Fully differentiable, with CUDA autograd implementation of WFST.
- Relates to thoughts mentioned in the [generic modeling presentation from the offsite](https://docs.google.com/presentation/d/1Y7pUKlgtp0y-vyWGPtH17pvRVhx9jg92ycbGbjiBDBY/edit#slide=id.g25b4f0ab693_0_721)

![poster](https://i.imgur.com/27f45ad.jpg)

### [Enhancing the Unified Streaming and Non-streaming Model with Contrastive Learning](https://arxiv.org/abs/2306.00755)

**Tags:** knowledge distillation, pseudo-labeling

- Broadly, worth digging into the idea of unified streaming and non-streaming models as a way of knowledge distillation
- Separate streaming and non-streaming encoders, each with its own CTC loss, but contrastive loss to bridge representations. Also followed by an AED (Attention Encoder Decoder) component?
- [Google's dual-mode ASR](https://arxiv.org/abs/2010.06030) was raised in Q&A
  - The difference here is the Google approach uses knowledge distillation loss instead of contrastive loss
  - What happens at inference?

### [ZeroPrompt: Streaming Acoustic Encoders are Zero-Shot Masked LMs](https://arxiv.org/abs/2305.10649)

**Tags:** latency, implicit LM

- There's some latency optimization stuff here by adding zero-padding to the end of each chunk/segment to force it to emit early somehow, but tricks to counterat accuracy loss, but all without needing to retrain the model. How?
- Some connections to masked LM training, and there appears to be a risk that this would worsen the implicit LM
- Not a direct application, but there are parallels in latency measurement (first-emit, last-emit). Worth a read.

### [Improved Training for End-to-End Streaming Automatic Speech Recognition Model with Punctuation](https://arxiv.org/abs/2306.01296)

**Tags:** prompt-concatenation, implicit spaces

- "The acoustic model trained with long sequences by concatenating the input and target sequences can learn punctuation marks attached to the end of sentences more effectively"
  - Clear parallels with prompt-concatenation and implicit spacing (except that the focus in the paper is implicit punctuation)
  - Also, the goal is here the precise opposite of ours - make the model emit implicit punctuation as opposed to counteract implicit spaces
- There's some complexity with the prompt-concat CTC loss. There's another CTC loss computation corresponding to the concatenation (called "chunk CTC loss" in the paper) and the total loss is a weighted sum of the CTC losses of individual targets and the concatenation.
- The above makes me wonder why in our setting we actually compute CTC over the entire concatenated sequence. We have the "local" alignment info within each prompt. Can we provide this info (and have the loss be sum(CTC loss of each prompt)) instead of requiring the model to learn this? Likely counter implicit LM too?
  - Super tricky to implement in an efficient manner though as you can't batch anymore given that the prompt boundaries different for each batch item.
  - But can borrow ideas from ASG loss?
  - Could this be formulated via WFST more efficiently?

### [2-bit Conformer quantization for automatic speech recognition](https://arxiv.org/abs/2305.16619)

**Tags:** quantization, conformer

- A Google paper that productionizes a 2-bit quantized conformer model for ASR.
- While this is a bit far-fetched for us, some of the tricks here such as sub-channel quantization are worth reading up on.
  
![poster](https://imgur.com/gsFcZPv.jpg)

### [Knowledge Distillation from Non-streaming to Streaming ASR Encoder using Auxiliary Non-streaming Layer](https://www.isca-speech.org/archive/interspeech_2023/shim23_interspeech.html)

**Tags:** knowledge distillation

- Relates to the other paper about distilling from a non-streaming / full-context model to a streaming model, but done so without unified training
- Primarily aims to address two challenges of KD
  - Teacher has very different context, which makes the student "difficult to follow"
  - Different alignments between teacher and student, which further exacerbates this problem
- Uses a combination of losses (KD, distance, predictive coding) and auxiliary non-streaming layers to counter this.
- Not sure what's particularly novel here from the approaches we already have in pyspeech building blocks?

### [Improving RNN-Transducers with Acoustic LookAhead](https://arxiv.org/abs/2307.05006)

**Tags:** RNN-T, implicit LM

- Aims to counter the "hallucination" problem of RNN-T owing to its tight coupling with an LM, i.e., the model predictions overly leaning on the prediction network ignoring the input signal when the acoustic model has low confidence.
- They basically "lookahead" into the acoustic model by zeroing-out the output of the prediction network (text-encoder in the poster image).
- Feels somewhat of a bandaid approach?

![poster](https://imgur.com/DLZg7tK.jpg)

### [4D ASR: Joint modeling of CTC, Attention, Transducer, and Mask-Predict decoders](https://arxiv.org/abs/2212.10818)

**Tags:** CTC, RNN-T, decoder

- From Shinji Watanabe's group at CMU.
- Far-fetched for us, but the algorithm/implementation surrounding the fused decoder could make an interesting read.

![poster](https://imgur.com/Lcv5DAZ.jpg)

### [Blank-regularized CTC for Frame Skipping in Neural Transducer](https://arxiv.org/abs/2305.11558)

**Tags:** CTC, WFST

- Problem: Every non-blank output by the acoustic model adds work to the expensive decoder computations that occur downstream in ASR.
- Solution: Making the model not output a bunch of repeats but emit more blanks can save training and inference cost by needing to run the decoder less.
  - They do so by regularizing CTC to emit more blanks, and using this CTC-trained encoder's output to skip frames in RNN-T.
- Not immediately relevant for us - if anything this approach would exacerbate the "peaky" behavior of CTC even further. Also, we don't have the HCLG-like long chain of decoders as in ASR, likely just one.
- But, IMO, a pretty compelling demonstration of the power of decoupling the algorithm from its implementation. In this case, thinking through and formulating CTC or its variants at the higher-level abstraction of a graph/automata as opposed to in CUDA code.

![poster](https://imgur.com/tX1Bb4X.jpg)

### [Regarding Topology and Variant Frame Rates for Differentiable WFST-based End-to-End ASR](https://www.research.ed.ac.uk/en/publications/regarding-topology-and-variant-frame-rates-for-differentiable-wfs)

**Tags:** CTC, WFST

- Another paper similar to the previous one exploring variants in CTC's WFST topologies and looking into how this affects properties like output frame-rates, number of possible alignments, etc.
- Funnily, overheard the author complaining about the difficulty of getting differentiable WFST papers get accepted: "Reviewers ask why do you want to make WFST differentiable. Well, ask Dan Povey".

![poster](https://imgur.com/AL9K3N4.jpg)

### [DCTX-Conformer: Dynamic context carry-over for low latency unified streaming and non-streaming Conformer](https://arxiv.org/abs/2306.08175)

**Tags:** left-context, conformer
