# [Neural Networks and Deep Learning, Michael Nielsen](https://neuralnetworksanddeeplearning.com/)

- **Created**: 2024-12-29
- **Last Updated**: 2025-01-17
- **Status**: `Done`

---

- [X] Chapter 1: intro <https://neuralnetworksanddeeplearning.com/chap1.html>
  - [X] rabbithole: mathoverflow intuitive crutches for higher dimensional thinking: <https://mathoverflow.net/questions/25983/intuitive-crutches-for-higher-dimensional-thinking>
- [X] Chapter 2: backprop <https://neuralnetworksanddeeplearning.com/chap2.html>
  - [X] rabbithole: Rumelhardt & Hinton 1986 backprop paper <https://www.iro.umontreal.ca/~vincentp/ift3395/lectures/backprop_old.pdf>
- [X] Chapter 3: cross-entropy, softmax, overfitting and regularization, weight init <https://neuralnetworksanddeeplearning.com/chap3.html>
  - [X] rabbithole: Efficient BackProp paper (1998) by Yann LeCun <https://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf>
    - Skimmed through the last few sections focused on second-order / Hessian based optimization strategies. Need to deeply understand L-BFGS, etc.
- [X] Chapter 4: universal function approximators <https://neuralnetworksanddeeplearning.com/chap4.html>
- [X] Chapter 5: vanishing gradient problem <https://neuralnetworksanddeeplearning.com/chap5.html>
- [X] Chapter 6: convolution, max pooling, and general deep learning overview <https://neuralnetworksanddeeplearning.com/chap6.html>
  - skimmed through
- [X] Appendix: Is there a simple algorithm for intelligence? <https://neuralnetworksanddeeplearning.com/sai.html>
  
## Notes

### Intention-driven user interfaces

I like the following paragraphs in the closing section of the book in chapter 6: <https://neuralnetworksanddeeplearning.com/chap6.html#on_the_future_of_neural_networks>. It's intersting thinking about this from EMG point-of-view.

> Intention-driven user interfaces: There's an old joke in which an impatient professor tells a confused student: "don't listen to what I say; listen to what I mean". Historically, computers have often been, like the confused student, in the dark about what their users mean. But this is changing. I still remember my surprise the first time I misspelled a Google search query, only to have Google say "Did you mean [corrected query]?" and to offer the corresponding search results. Google CEO Larry Page once described the perfect search engine as understanding exactly what [your queries] mean and giving you back exactly what you want.
>
> This is a vision of an intention-driven user interface. In this vision, instead of responding to users' literal queries, search will use machine learning to take vague user input, discern precisely what was meant, and take action on the basis of those insights.
>
> The idea of intention-driven interfaces can be applied far more broadly than search. Over the next few decades, thousands of companies will build products which use machine learning to make user interfaces that can tolerate imprecision, while discerning and acting on the user's true intent. We're already seeing early examples of such intention-driven interfaces: Apple's Siri; Wolfram Alpha; IBM's Watson; systems which can annotate photos and videos; and much more.
>
> Most of these products will fail. Inspired user interface design is hard, and I expect many companies will take powerful machine learning technology and use it to build insipid user interfaces. The best machine learning in the world won't help if your user interface concept stinks. But there will be a residue of products which succeed. Over time that will cause a profound change in how we relate to computers. Not so long ago - let's say, 2005 - users took it for granted that they needed precision in most interactions with computers. Indeed, computer literacy to a great extent meant internalizing the idea that computers are extremely literal; a single misplaced semi-colon may completely change the nature of an interaction with a computer. But over the next few decades I expect we'll develop many successful intention-driven user interfaces, and that will dramatically change what we expect when interacting with computers.

### Conway's Law

There's another discussion in the closing section in chapter 6 about Conway's Law: <https://neuralnetworksanddeeplearning.com/chap6.html#on_the_future_of_neural_networks>.

> Any organization that designs a system... will inevitably produce a design whose structure is a copy of the organization's communication structure.

The discussion is about how a field grows, from a nascent monolithic entity to having subfields and specializations. Taking medicine for example, early on there were no sub-specializations and practitioners like Hippocrates studied the entire body. And how a field grows and subspecializations form is dependent on the deep insights made prior. Using immunology as an example, without first germ theory of disease or knowing that antibodies existed, immunology couldn't have developed.
So you start with basic research, you make some discoveries, and your field specializes around those discoveries and social structure of the field forms based on that.

> "basic research is what I'm doing when I don't know what I'm doing"

Below is the full text of interest:

> Conway's law applies to the design and engineering of systems where we start out with a pretty good understanding of the likely constituent parts, and how to build them. It can't be applied directly to the development of artificial intelligence, because AI isn't (yet) such a problem: we don't know what the constituent parts are. Indeed, we're not even sure what basic questions to be asking. In others words, at this point AI is more a problem of science than of engineering. Imagine beginning the design of the 747 without knowing about jet engines or the principles of aerodynamics. You wouldn't know what kinds of experts to hire into your organization. As Wernher von Braun put it, "basic research is what I'm doing when I don't know what I'm doing". Is there a version of Conway's law that applies to problems which are more science than engineering?
>
> To gain insight into this question, consider the history of medicine. In the early days, medicine was the domain of practitioners like Galen and Hippocrates, who studied the entire body. But as our knowledge grew, people were forced to specialize. We discovered many deep new ideas**My apologies for overloading "deep". I won't define "deep ideas" precisely, but loosely I mean the kind of idea which is the basis for a rich field of enquiry. The backpropagation algorithm and the germ theory of disease are both good examples.: think of things like the germ theory of disease, for instance, or the understanding of how antibodies work, or the understanding that the heart, lungs, veins and arteries form a complete cardiovascular system. Such deep insights formed the basis for subfields such as epidemiology, immunology, and the cluster of inter-linked fields around the cardiovascular system. And so the structure of our knowledge has shaped the social structure of medicine. This is particularly striking in the case of immunology: realizing the immune system exists and is a system worthy of study is an extremely non-trivial insight. So we have an entire field of medicine - with specialists, conferences, even prizes, and so on - organized around something which is not just invisible, it's arguably not a distinct thing at all.
>
> This is a common pattern that has been repeated in many well-established sciences: not just medicine, but physics, mathematics, chemistry, and others. The fields start out monolithic, with just a few deep ideas. Early experts can master all those ideas. But as time passes that monolithic character changes. We discover many deep new ideas, too many for any one person to really master. As a result, the social structure of the field re-organizes and divides around those ideas. Instead of a monolith, we have fields within fields within fields, a complex, recursive, self-referential social structure, whose organization mirrors the connections between our deepest insights. And so the structure of our knowledge shapes the social organization of science. But that social shape in turn constrains and helps determine what we can discover. This is the scientific analogue of Conway's law.
