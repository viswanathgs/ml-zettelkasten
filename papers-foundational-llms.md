# Foundational LLMs

- **Created**: 2025-01-04
- **Last Updated**: 2025-01-06
- **Status**: `Paused`

---

- [X] [2022] OPT: Open Pre-trained Transformer Language Models. <https://arxiv.org/abs/2205.01068>
- [X] [2022] LLaMa: Open and Efficient Foundation Language Models. <https://arxiv.org/abs/2302.13971>
- [ ] [2023] Llama 2: Open Foundation and Fine-Tuned Chat Models
- [ ] GPT 1-3
- [ ] llama3
- [ ] llama4
- [ ] GPT4

## [2022] OPT: Open Pre-trained Transformer Language Models

- **Date**: 2025-01-04
- **Arxiv**: <https://arxiv.org/abs/2205.01068>
- **Paperpile**: <https://app.paperpile.com/view/?id=4eefc94a-525e-4e84-88b4-2e80d7c04a56>

---

- Mainly replicating GPT3
- Fully Sharded Data Parallel (FSDP)

## [2022] Llama: Open and Efficient Foundation Language Models

- **Date**: 2025-01-04
- **Arxiv**: <https://arxiv.org/abs/2302.13971>
- **Paperpile**: <https://app.paperpile.com/view/?id=152b9407-9eca-4b65-8d4f-c753723826c9>

---

- Compared to GPT3, OPT, etc., smaller models trained on more tokens
- Pretraining data:
  - 1.4 trillion tokens
  - Tokenizer: Byte-Pair Encoding (BPE) using SentencePiece (<https://github.com/google/sentencepiece>) implementation
- Architecture updates on top of standard transformer:
  - RMSNorm normalization for inputs per layer - GPT3 inspired
  - SwiGLU activation (GLU, but swish instead of sigmoid) - PaLM inspired
  - Rotary positional embeddings (RoPE)
- AdamW optimizer with cosine LR scheduler
- Impl
  - Transformer impl from <https://github.com/facebookresearch/xformers> which includes flash attention
  - Improvements over default activation checkpointing to also cache some expensive linear layer activations.
- Evals
- Post-Training / Instruction Finetuning
- Section 7 (Related Work) has a good history of large LMs starting with Kneyser-Ney n-gram through BERT, GPT, etc.

## [2023] Llama 2: Open Foundation and Fine-Tuned Chat Models

- **Date**: 2025-01-06
- **Arxiv**: <https://arxiv.org/abs/2307.09288>
- **Paperpile**: <https://app.paperpile.com/view/?id=726a7eed-b9ac-4380-a737-ab1a17ed0b56>

---

- 1. Overview
  - Llama2 (pretrained): Differences from llama1
    - 40% larger pretraining corpus: 1.4 trillion tokens (llama1) -> 2 trillion tokens (llama2)
    - 2x context length
      - TODO: what is context length exactly?
    - Grouped-Query Attention (GQA) for inference scalability
      - TODO: follow up on the above
  - Llama2-chat (finetuned): supervised finetuning followed by RLHF (rejection sampling, PPO)
    - Understand fig 4 better - rejection sampling, PPO
- 2. Pretraining
  - 2 trillion tokens
  - Tokenizer: Same as llama1. BPE algorithm using SentencePiece implementation.
  - Most others are same as llama1
- 3. Finetuning
  - 3.1 Supervised Fine-Tuning (SFT)
    - Bootstrapping with instruction finetuning data.
    - Supervised in the sense that both the prompt and the response are provided by the annotator.
  - 3.2 Reinforcement Learning with Human Feedback (RLHF)
    - TODO
