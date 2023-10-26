# Attention
Imagine you're reading a dense academic paper, thrilling novel or LLM playbook. Your brain doesn't weigh each word or sentence equally. Some portions are scrutinized carefully, while others may be skimmed over. You naturally pay 'attention' to specific parts based on their relevance to your current focus or understanding. This selective focus allows you to better comprehend the text and keeps you from being overwhelmed by unnecessary details.

In the realm of machine learning, particularly in sequence-to-sequence tasks like machine translation or text summarization, a similar mechanism is invaluable. Early models like [RNNs](https://explained.ai/rnn/) and [LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) process sequences step-by-step, but they can struggle with long sequences and often lose track of important earlier tokens when focused on the more recent ones. The attention mechanism was introduced to combat these limitations. It essentially allows the model to "focus" on different parts of the input sequence when producing an output, much like how you would focus on certain parts of a text while reading.

### The Attention Mechanism Explained
Within this framework, three major components come into play: Query (**Q**), Key (**K**), and Value (**V**).

* **Query (Q)**: Picture this as the search term you'd type into a search engine—like asking your brain, "Hey, what should I focus on right now?" It's a vector representing your current area of focus.
* **Key (K)**: The Key vectors are akin to the titles of Wikipedia articles. They serve as guideposts, each corresponding to a specific token in the input sequence, hinting where you might find relevant information.
* **Value (V)**: These vectors are the meat of the matter—the article content, if you will. They offer the detailed information each token carries.

How do they all come together?
```python
class Attention(nn.Module):
    def __init__(self, word_size:int=512, embed_dim:int=64) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.dim_K = torch.tensor(embed_dim)
        self.query = nn.Linear(in_features=word_size, out_features=embed_dim, bias=True)
        self.key  = nn.Linear(in_features=word_size, out_features=embed_dim, bias=True)
        self.value = nn.Linear(in_features=word_size, out_features=embed_dim, bias=True)

    def self_attention(self, Q:Tensor, K:Tensor, V:Tensor) -> Tensor:
        K_T = torch.transpose(K, 0, 1)
        score = torch.matmul(Q, K_T)  / torch.sqrt(self.dim_K)
        score = torch.softmax(score, dim=-1)
        Z = torch.matmul(score, V)
        return Z

    def forward(self, x:Tensor) -> Tensor:
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        Z = self.self_attention(Q, K, V)
        return Z
```

1. **Score Generation:** Every Query is matched against all Keys via a *dot product* to calculate a score. It's like asking, "How relevant is this part of the text to my current focus?"
2. **Softmax Scaling:** These scores undergo Softmax normalization, turning these scores into probabilities that sum up to 1. Picture this as divvying up your concentration across various parts.
3. **Weighted Sum**: Finally, these attention weights are applied to the Values (you guessed it, another dot product), summing them up to form a single output vector. You can think of this as gathering all the most valuable sentences to form a cohesive summary of what you need to focus on.

### Self-Attention
Self-attention is a specific type of attention mechanism where the Query (Q), Key (K), and Value (V) all come from the same place, usually the output of the previous layer in your network. In layman's terms, self-attention enables tokens in the input sequence to look at other tokens in the same sequence to gather contextual clues.

Self-attention becomes particularly interesting when employed in autoregressive models like GPT (Generative Pre-trained Transformer). In such models, the generation of each new token is dependent only on the preceding tokens. Causal self-attention restricts the scope of attention in a way that each token only looks at those that precede it, and not the ones that follow. This is crucial for maintaining the sequence structure and generating sensible outputs. Imagine it like reading a book where you only consider the chapters or sentences you've already read to make a prediction about what comes next. You don't peek ahead; you stick with what you know so far.

### Limitations and Challenges
While attention mechanisms have revolutionized sequence-to-sequence tasks, they're not without their challenges:

* **Computational Cost:** Attention mechanisms can be computationally expensive, especially for very long sequences. As such there exists a rich body of literature extending and refining it in various ways. We'll explore a few of those approaches in the subsequent sections.
* **Interpretability:** While attention weights can give some insight into what the model is "focusing" on, this doesn't necessarily mean the model "understands" the text in the way humans do.