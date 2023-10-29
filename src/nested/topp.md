# Top-P
While [Top-K sampling](./topp.md) restricts the sampling pool to the K most likely next words, Top-P sampling, also known as "nucleus sampling," adds a twist. Instead of specifying a set number of top candidates (K), you specify a probability mass (P) and sample only from the smallest group of words that have a collective probability greater than P.

Let's implement Top-P sampling using a NumPy function for better understanding:
```python
import numpy as np

def top_p_sampling(logits, p):
    sorted_indices = np.argsort(logits)  # Sort logits
    sorted_probs = np.exp(logits[sorted_indices]) / np.sum(np.exp(logits))  # Convert sorted logits to probabilities
    cum_probs = np.cumsum(sorted_probs)  # Calculate the cumulative probability
    valid_indices = np.where(cum_probs >= (1 - p))[0]  # Get valid indices where cumulative probability is above threshold
    if len(valid_indices) > 0:
        min_valid_index = valid_indices[0]
        mask = sorted_indices[min_valid_index:]  # Mask for valid logits
    else:
        mask = sorted_indices[-1:]  # If no valid indices, select the last one (highest probability)
    selected_index = np.random.choice(mask)  # Randomly select an index from the valid set
    return selected_index
```
Here's the step-by-step breakdown:

1. **Sort and Convert:** Sort the logits and convert them to probabilities.
2. **Cumulative Sum:** Calculate the cumulative sum of the sorted probabilities.
3. **Thresholding:** Identify the subset of words whose collective probability mass exceeds the given threshold (P).
4. **Sampling:** Randomly sample the next word from this set of valid candidates.

### When to Use Top-P Sampling
Top-P sampling is particularly useful when you want more adaptive and context-sensitive text generation. Unlike Top-K, which has a fixed number of candidates, Top-P allows for a variable number of candidates based on the context, making it more flexible.

### Limitations and Considerations
* Computational Cost: The sorting operation increases the computational cost slightly compared to Top-K sampling.
* Hyperparameter Sensitivity: The choice of P can significantly influence the generated text. A smaller P will make the text more random, while a larger P will make it more deterministic.
Top-P sampling provides an adaptive method for balancing the trade-off between diversity and informativeness in generated text. It has gained popularity in several NLP applications, from automated customer service to creative writing aids.