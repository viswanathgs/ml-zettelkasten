# ML Hardware

- **Created**: 2025-11-23
- **Last Updated**: 2025-11-23
- **Status**: `In Progress`

---

- [ ] [2019] [Jeff Dean] The Deep Learning Revolution and Its Implications for Computer Architecture and Chip Design - [paper](https://arxiv.org/abs/1911.05289)
- [ ] [2020] [Sara Hooker] The Hardware Lottery - [paper](https://arxiv.org/abs/2009.06489)
- [ ] TODO TPU
- [ ] TODO Ampere arch

---

## [2020] The Hardware Lottery

- **Date**: 2025-11-23
- **Arxiv**: <https://arxiv.org/abs/2009.06489>
- **Paperpile**: <https://app.paperpile.com/view/?id=427d748b-faae-45e3-9412-6a4c346dd62f>
- **HTML**: <https://hardwarelottery.github.io/>

---

- **Paradox**:
  - > a crucial paradox: **machine learning researchers mostly ignore hardware despite the role it plays in determining what ideas succeed**.
  - > **our own intelligence is both algorithm and machine**. We do not inhabit multiple brains over the course of our lifetime. Instead, the notion of human intelligence is intrinsically associated with the physical 1400g of brain tissue and the patterns of connectivity between an estimated 85 billion neurons in your head.
  - > Today, in contrast to the necessary specialization in the very early days of computing, machine learning researchers tend to think of hardware, software and algorithm as three separate choices. This is largely due to a period in computer science history that radically changed the type of hardware that was made and incentivized hardware, software and machine learning research communities to evolve in isolation.
- **The Anna Karenina Principle**:
  - > The first sentence of Anna Karenina by Tolstoy reads “Happy families are all alike, every unhappy family is unhappy in it’s own way.” [19] Tolstoy is saying that it takes many different things for a marriage to be happy — financial stability, chemistry, shared values, healthy offspring. However, it only takes one of these aspects to not be present for a family to be unhappy. This has been popularized as the Anna Karenina principle — “a deficiency in any one of a number of factors dooms an endeavor to failure.” [20]
  - > Despite our preference to believe algorithms succeed or fail in isolation, history tells us that most computer science breakthroughs follow the Anna Kerenina principle. Successful breakthroughs are often distinguished from failures by benefiting from multiple criteria aligning surreptitiously. For computer science research, this often depends upon winning what this essay terms the hardware lottery — avoiding possible points of failure in downstream hardware and software choices.
  - > **“being too early is the same as being wrong”**. When Babbage passed away in 1871, there was no continuous path between his ideas and modern day computing. The concept of a stored program, modifiable code, memory and conditional branching were rediscovered a century later because the right tools existed to empirically show that the idea worked.
- **The Lost Decades**:
  - > Perhaps **the most salient example of the damage caused by not winning the hardware lottery is the delayed recognition of deep neural networks** as a promising direction of research. Most of the algorithmic components to make deep neural networks work had already been in place for a few decades: backpropagation (1963 [23], reinvented in 1976 [24], and again in 1988 [25]), deep convolutional neural networks (1979 [26], paired with backpropagation in 1989 [27]). However, it was only three decades later that convolutional neural networks were widely accepted as a promising research direction.
  - > This gap between algorithmic advances and empirical success is in large part due to incompatible hardware.
- **The Persistence of the Hardware Lottery**:
  - > unstructured pruning [65, 66, 67] and weight specific quantization [68] are very successful compression techniques in deep neural networks but are incompatible with current hardware and compilations kernels.
  - > It is a reasonable prediction that the next few generations of chips or specialized kernels will correct for present hardware bias against these techniques [69, 70]. Some of the first designs which facilitate sparsity have already hit the market [71]. In parallel, there is interesting research developing specialized software kernels to support unstructured sparsity [72, 73, 74].
  - > In many ways, hardware is catching up to the present state of machine learning research. **Hardware is only economically viable if the lifetime of the use case lasts more than three years**.
  - > Betting on ideas which have longevity is a key consideration for hardware developers. Thus, **co-design effort has focused almost entirely on optimizing an older generation of models with known commercial use cases**. For example, matrix multiplies are a safe target to optimize for because they are here to stay — anchored by the widespread use and adoption of deep neural networks in production systems. Allowing for unstructured sparsity and weight specific quantization are also safe targets because there is wide consensus that these will enable higher levels of compression.
  - > In 2019, a paper was published called “Machine learning is stuck in a rut.” [75] The authors consider the difficulty of training a new type of computer vision architecture called capsule networks on domain specialized hardware [76]. Capsule networks include novel components like squashing operations and routing by agreement. These architecture choices aimed to solve for key deficiencies in convolutional neural networks (lack of rotational invariance and spatial hierarchy understanding) but strayed from the typical architecture of neural networks as a sequence of matrix multiplies. As a result, while capsule networks operations can be implemented reasonably well on CPUs, performance falls off a cliff on accelerators like GPUs and TPUs which have been overly optimized for matrix multiplies.
  - > Whether or not you agree that capsule networks are the future of computer vision, the authors say something interesting about the difficulty of trying to train a new type of image classification architecture on domain specialized hardware. **Hardware design has prioritized delivering on commercial use cases, while built-in flexibility to accommodate the next generation of research ideas remains a distant secondary consideration**.
- **Biological Intelligence**:
  - > Perhaps more troubling is how far away we are from the type of intelligence humans demonstrate. **Human brains despite their complexity remain extremely energy efficient**. Our brain has over 85 billion neurons but runs on the energy equivalent of an electric shaver[9]. While deep neural networks may be scalable, it may be prohibitively expensive to do so in a regime of comparable intelligence to humans. An apt metaphor is that we appear to be trying to build a ladder to the moon.
  - > Biological examples of intelligence differ from deep neural networks in enough ways to suggest it is a risky bet to say that deep neural networks are the only way forward. **While general purpose algorithms like deep neural networks rely on global updates in order to learn a useful representation, our brains do not. Our own intelligence relies on decentralized local updates which surface a global signal in ways that are still not well understood** [85, 86, 87].
  - > In addition, **our brains are able to learn efficient representations from far fewer labelled examples than deep neural networks** [88]. For typical deep learning models the entire model is activated for every example which leads to a quadratic blow-up in training costs. In contrast, **evidence suggests that the brain does not perform a full forward and backward pass for all inputs. Instead, the brain simulates what inputs are expected against incoming sensory data. Based upon the certainty of the match, the brain simply infills**. What we see is largely virtual reality computed from memory [89, 90, 91]
  - > Humans have highly optimized and specific pathways developed in our biological hardware for different tasks [92, 93]. For example, it is easy for a human to walk and talk at the same time. However, it is far more cognitively taxing to attempt to read and talk [94]. This suggests **the way a network is organized and our inductive biases is as important as the overall size of the network**.
  - > **Our brains are able to fine-tune and retain humans skills across our lifetimes**. In contrast, **deep neural networks that are trained upon new data often evidence catastrophic forgetting**, where performance deteriorates on the original task because the new information interferes with previously learned behavior.
  - > The point of these examples is not to convince you that deep neural networks are not the way forward. But, rather that there are clearly other models of intelligence which suggest it may not be the only way. **It is possible that the next breakthrough will require a fundamentally different way of modelling the world with a different combination of hardware, software and algorithm**. We may very well be in the midst of a present day hardware lottery.
- > Scientific progress occurs when there is a confluence of factors which allows the scientist to overcome the “stickyness” of the existing paradigm. **The speed at which paradigm shifts have happened in AI research have been disproportionately determined by the degree of alignment between hardware, software and algorithm**. Thus, any attempt to avoid hardware lotteries must be concerned with making it cheaper and less time-consuming to explore different hardware-software-algorithm combinations.
