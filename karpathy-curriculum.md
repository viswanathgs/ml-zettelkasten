# Karpathy Curriculum

- **Created**: 2025-10-14
- **Last Updated**: 2025-10-24
- **Status**: `In Progress`

---

- [ ] [micrograd](https://github.com/karpathy/micrograd)
- [ ] [minbpe](https://github.com/karpathy/minbpe)
- [ ] [llama2.c](https://github.com/karpathy/llama2.c)
- [ ] [llm.c](https://github.com/karpathy/llm.c)
- [ ] [nanochat](https://github.com/karpathy/nanochat)

---

## [minbpe](https://github.com/karpathy/minbpe)

- **Date**: 2025-10-24

---

- Original BPE (Byte Pair Encoding) algorithm in [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) [[papers-ml-fundamentals.md]], and then popularized by GPT-2.
  - Basic BPE implementation in <https://github.com/karpathy/minbpe/blob/master/minbpe/basic.py>
  - TODO See my independent impl at TODO
- Couple of additional things to handle:
  - RegexTokenizer: Preprocesses the input text by splitting it into categories (letters, numbers, puncutation) before tokenization. Avoids merges across cateogry boundaries. See regex based splitting and chunk handling in `train()` and `encode_ordinary()` of <https://github.com/karpathy/minbpe/blob/master/minbpe/regex.py>.
  - Handling special tokens like `<|promptstart|>` or `<|endoftext|>`, etc.
- > Tokenization is at the heart of a lot of weirdness in LLMs and I would advise that you do not brush it off. A lot of the issues that may look like issues with the neural network architecture actually trace back to tokenization. Here are just a few examples:
  >
  > Why can't LLM spell words? **Tokenization**.
  > Why can't LLM do super simple string processing tasks like reversing a string? **Tokenization**.
  > Why is LLM worse at non-English languages (e.g. Japanese)? **Tokenization**.
  > Why is LLM bad at simple arithmetic? **Tokenization**.
  > Why did GPT-2 have more than necessary trouble coding in Python? **Tokenization**.
  > Why did my LLM abruptly halt when it sees the string "<|endoftext|>"? **Tokenization**.
  > What is this weird warning I get about a "trailing whitespace"? **Tokenization**.
  > Why did the LLM break if I ask it about "SolidGoldMagikarp"? **Tokenization**.
  > Why should I prefer to use YAML over JSON with LLMs? **Tokenization**.
  > Why is LLM not actually end-to-end language modeling? **Tokenization**.
  > What is the real root of suffering? **Tokenization**.
