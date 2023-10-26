# Positional Embeddings
The transformer architecture has revolutionized the field of natural language processing, but it comes with a peculiar limitation: it lacks an intrinsic mechanism to account for the position or sequence order of elements in an input. In plain terms, a transformer model would produce the same output for two different permutations of the same input sequence. This is problematic because the sequence in which words or tokens appear carries significant meaning in language and other types of data.

This limitation arises because the architecture relies on [self-attention mechanisms](/attention.md), which, by their very design, are permutation-invariant—they treat all positions equally and thus are indifferent to the arrangement of elements in the sequence. Consequently, while transformers excel at recognizing patterns and relationships between elements, they are blind to the "where" and "when" of those elements within the sequence.

To address this shortcoming and make transformers aware of element positions, we use a specialized form of embeddings known as positional embeddings. These embeddings work alongside the standard word embeddings to grant transformers the capability to understand sequence order. By doing so, they complete the picture, allowing the model to interpret data in a way that respects both content and sequence.

As with all aspects of machine learning, the choice of position encoding typically involves tradeoffs between simplicity, flexibility, and efficiency. Here we explore a few of the most popular methods:

### Absolute Positional Encoding
Absolute position encodings are computed in the input layer and are summed with the input token embeddings. [Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762) proposed this for Transformers and it has been a popular choice in the followup works ([Radford et al., 2018](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf); [Devlin et al., 2018](https://arxiv.org/abs/1810.04805)). There are two common variations of the absolute position encodings - fixed and learned.

* [Fixed Positional Embeddings](nested/fixed-pos-embed.md)
* [Learned Positional Embeddings](nested/learned-pos-embed.md)

### Relative Positional Encoding
One drawback of absolute position encoding is that it requires fixed length of input sequence and does not directly capture relative positions to each word. To solve these problems several relative positions schemes have been proposed.
* [Rotary Positional Embedding](nested/rot-pos-embed.md)