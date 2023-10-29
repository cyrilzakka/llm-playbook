# Top-K
When generating text, a language model can predict the next word based on the previous words in the sequence. One approach is to select the word with the highest probability, but this method—known as "greedy decoding"—often results in repetitive and incoherent text. This is where sampling techniques like Top-K sampling come into play.

The idea behind Top-K sampling is quite straightforward: instead of considering all possible next words in the vocabulary, limit the pool to the top-K most likely next words and sample from this narrowed distribution.

Here's how to do Top-K sampling in a simple NumPy function:

```python
import numpy as np

def top_k_sampling(logits, k):
    top_k_indices = np.argsort(logits)[-k:]  # Get indices of top-k logits
    top_k_logits = logits[top_k_indices]  # Get the top-k logits
    top_k_probs = np.exp(top_k_logits) / np.sum(np.exp(top_k_logits))  # Convert logits to probabilities
    selected_index = np.random.choice(top_k_indices, p=top_k_probs)  # Sample from the top-k indices based on the probabilities
    return selected_index
```

Here's the breakdown:

1. **Select Top-K:** Given the logits for the next word, we select the top-K logits, where K is a predetermined hyperparameter.
2. **Convert to Probabilities:** We then convert these logits to probabilities using the Softmax function.
3. **Sampling:** Finally, we sample the next word from this top-K distribution.

### When to Use Top-K Sampling
Top-K sampling is often used when you want a balance between randomness and relevance in the generated text. It allows the model to explore a bit, potentially generating more creative and diverse text while still being more coherent than random sampling.

### Limitations and Considerations
* Hyperparameter Tuning: The choice of K can significantly influence the results. A smaller K will make the output more focused but less creative, while a larger K will make the output more diverse but potentially less relevant.
* Not Adaptive: The value of K remains constant, meaning the method isn't adaptive to the context of the text being generated. This limitation has led to the development of more advanced sampling techniques like nucleus sampling.