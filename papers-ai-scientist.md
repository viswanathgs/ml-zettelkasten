# AI Scientist

- **Created**: 2025-11-07
- **Last Updated**: 2025-11-09
- **Status**: `In Progress`

---

- [ ] AlphaTensor
- [X] [2023] FunSearch: Making new discoveries in mathematical sciences using Large Language Models - [paper](https://www.nature.com/articles/s41586-023-06924-6)
- [ ] [2025] AlphaEvolve: A coding agent for scientific and algorithmic discovery - [paper](https://arxiv.org/abs/2506.13131)
- [ ] TODO sakana
- [ ] TODO others?
- [ ] TODO math paper from shubho

---

## [2023] FunSearch: Making new discoveries in mathematical sciences using Large Language Models

- **Date**: 2025-11-09
- **Nature**: <https://www.nature.com/articles/s41586-023-06924-6>
- **Blog**: <https://deepmind.google/blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/>

---

- **Abstract**:
  - > Large language models (LLMs) have demonstrated tremendous capabilities in solving complex tasks, from quantitative reasoning to understanding natural language. However, LLMs sometimes suffer from confabulations (or hallucinations), which can result in them making plausible but incorrect statements1,2. This hinders the use of current large models in scientific discovery. Here we introduce FunSearch (short for searching in the function space), an evolutionary procedure based on pairing a pretrained LLM with a systematic evaluator. We demonstrate the effectiveness of this approach to surpass the best-known results in important problems, pushing the boundary of existing LLM-based approaches3. Applying FunSearch to a central problem in extremal combinatorics—the cap set problem—we discover new constructions of large cap sets going beyond the best-known ones, both in finite dimensional and asymptotic cases. This shows that it is possible to make discoveries for established open problems using LLMs. We showcase the generality of FunSearch by applying it to an algorithmic problem, online bin packing, finding new heuristics that improve on widely used baselines. In contrast to most computer search approaches, FunSearch searches for programs that describe how to solve a problem, rather than what the solution is. Beyond being an effective and scalable strategy, discovered programs tend to be more interpretable than raw solutions, enabling feedback loops between domain experts and FunSearch, and the deployment of such programs in real-world applications.
- **Hard to solve but easy to evaluate problems**
  - > Many problems in mathematical sciences are ‘easy to evaluate’, despite being typically ‘hard to solve’. For example, in computer science, NP-complete optimization problems admit a polynomial-time evaluation procedure (measuring the quality of the solution), despite the widespread belief that no polynomial-time algorithms to solve such problems exist. We focus in this paper on problems admitting an efficient ‘evaluate’ function, which measures the quality of a candidate solution.
  - > Our goal is to generate a ‘solve’ program, such that its outputs receive high scores from the ‘evaluate’ function (when executed on inputs of interest), and ultimately improve on the best-known solutions.
- **Evolutionary algorithms to go beyond LLM's limitations / training domain**
  - > synthesizing ‘solve’ programs for open problems requires finding new ideas that are verifiably correct. This is very hard for LLMs, as they tend to confabulate or ultimately fall short of going beyond existing results. To surpass the ‘nominal’ capabilities of LLMs, recent studies have combined them with evolutionary algorithms.
  - > FunSearch pushes the boundary of LLM-guided evolutionary procedures to a new level: the discovery of new scientific results for established open problems and the discovery of new algorithms. Surpassing state-of-the-art results on established open problems provides a clear indication that the discoveries are truly new, as opposed to being retrieved from the LLM’s training data.
- **FunSearch** (Fig 1):
  - Pretrained (frozen) LLM proposes solutions, evaluator verifies.
  - (1) Start with a program in the form of a skeleton (containing boilerplate code and potentially known structure about the problem), and only evolve the part governing the critical program logic
  - (2) Best-shot prompting: sample best performing programs and feed them back into prompts for the LLM to improve on.
  - (3) Maintain a large pool of diverse programs by using an island-based evolutionary method that encourages exploration and avoids local optima.
- **Outputting code rather than the solution directly**:
  - > Whereas most computer search techniques output directly what the solution is (for example, a list of vectors forming a cap set), FunSearch produces programs generating the solution. For structured problems, such programs tend to be more interpretable and concise.
- **Island-based evolutionary algorithm**:
  - > To encourage diversity, we adopt an islands model, also known as a multiple population and multiple-deme model, which is a genetic algorithm approach.
  - > Several islands, or subpopulations, are created and evolved independently. To sample from the program database, we first sample an island and then sample a program within that island, favouring higher-scoring and shorter programs (see Methods for the exact mechanism).
  - > Crucially, we let information flow between the islands by periodically discarding the programs in the worst half of the islands (corresponding to the ones whose best individuals have the lowest scores). We replace the programs in those islands with a new population, initialized by cloning one of the best individuals from the surviving islands.
- **Best-shot prompting**:
  - > We first sample k programs from a single island in the programs database, according to the procedure described above. Sampled programs are then sorted according to their score, and a version is assigned to each (‘v0’ for the lowest scoring program, ‘v1’ for the second lowest scoring and so on). These programs are then combined into a single prompt— with the version appended as a suffix to the function name; for example, in the case of Fig. 2a, this would be ‘priority_v0’, ‘priority_v1’, ...—and the header of the function we wish to generate (for example, ‘priority_vk’) is added to the end of the prompt.
  - > In practice, we set k = 2, as two functions lead to better results compared to just one, with diminishing returns beyond that.
  - > **Constructing a prompt by combining several programs (as opposed to only one) enables the LLM to spot patterns across the different programs and generalize those**.
- **Discussion**:
  - > We believe that the LLM used within FunSearch does not use much context about the problem; **the LLM should instead be seen as a source of diverse (syntactically correct) programs with occasionally interesting ideas**. When further constrained to operate on the crucial part of the algorithm with a program skeleton, **the LLM provides suggestions that marginally improve over existing ones in the population, which ultimately results in discovering new knowledge on open problems when combined with the evolutionary algorithm**.
  - > We note that FunSearch, at present, works best for problems having the following characteristics:
    - (1) availability of an efficient evaluator,
    - (2) a ‘rich’ scoring feedback quantifying the improvements (as opposed to a binary signal),
    - (3) ability to provide a skeleton with an isolated part to be evolved.
    - > For example, the problem of **generating proofs for theorems falls outside this scope, because it is unclear how to provide a rich enough scoring signal**.

[paper](https://www.nature.com/articles/s41586-023-06924-6)

## [2025] AlphaEvolve: A coding agent for scientific and algorithmic discovery

- **Date**: 2025-11-07
- **Arxiv**: <https://arxiv.org/abs/2506.13131>
- **Paperpile**: <https://app.paperpile.com/view/?id=0776fda3-5920-4f30-8b02-73049e8adb3f>
- **Blog**: <https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/>

---

- TODO
